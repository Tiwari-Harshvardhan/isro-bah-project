#!/usr/bin/env python3
"""
================================================================================
 UrbanCool — Delhi Colony Monthly NDVI Pipeline
================================================================================
Computes monthly average NDVI (Jan 2018 - Dec 2025) for every colony polygon
in delhi_colonies.shp, using Sentinel-2 L2A data streamed directly from the
Microsoft Planetary Computer STAC API.

DESIGN PRINCIPLES (per project requirements):
  - No Sentinel imagery or GeoTIFFs are ever saved to disk permanently.
  - Zonal statistics are computed directly on in-memory NumPy arrays via
    rasterstats (array + affine transform) — no raster files are written
    at all, temporary or otherwise.
  - Only bands B04 (Red) and B08 (NIR) are ever read.
  - Reads are spatially restricted to the Delhi bounding box BEFORE any
    pixel data is pulled (odc-stac `bbox=` clips at the source/COG level).
  - Processing is strictly one month at a time. After each month:
        results are appended to the CSV immediately (checkpoint),
        all arrays/datasets are deleted and garbage-collected.
  - Re-running the script auto-detects already-completed months in the
    output CSV and resumes from the next unfinished month.
  - All network operations (STAC search, asset reads) are wrapped in a
    retry-with-backoff helper to survive transient HTTP errors, corrupted
    assets, or dropped connections.

OUTPUT:
    Delhi_Colony_Monthly_NDVI_2018_2025.csv
    Columns: zone, ward, colony, year, month, mean_ndvi
================================================================================
"""

import os
import gc
import sys
import time
import calendar
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd

import pystac_client
import planetary_computer as pc
import odc.stac

from rasterstats import zonal_stats

import psutil
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SHAPEFILE_PATH   = "delhi_colonies.shp"
OUTPUT_CSV       = "Delhi_Colony_Monthly_NDVI_2018_2025.csv"

STAC_API_URL     = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION       = "sentinel-2-l2a"

START_YEAR, START_MONTH = 2018, 1
END_YEAR, END_MONTH     = 2025, 12

# Images with cloud cover above this threshold are dropped in the
# "cloud filtering" step (step 2 of the pipeline).
CLOUD_COVER_THRESHOLD = 40   # percent

# Delhi is entirely within UTM Zone 43N — fixing the CRS explicitly keeps
# every month's raster grid aligned, which matters for consistent zonal stats.
TARGET_CRS       = "EPSG:32643"
TARGET_RESOLUTION = 10   # metres (native Sentinel-2 B04/B08 resolution)

BANDS = ["B04", "B08"]

# Network / robustness settings
MAX_RETRIES        = 5
RETRY_BACKOFF_BASE = 5     # seconds; actual wait = base * attempt_number

# Shapefile attribute columns expected to carry through to the output CSV.
ATTRIBUTE_COLUMNS = ["zone", "ward", "colony"]


# ==============================================================================
# SMALL UTILITIES
# ==============================================================================

def log(msg):
    """Timestamped, flush-immediately print so logs survive crashes/pipes."""
    print(msg, flush=True)


def retry_operation(func, *args, max_retries=MAX_RETRIES, desc="operation", **kwargs):
    """
    Generic retry-with-exponential-backoff wrapper.
    Used around every network-touching call (STAC search, odc.stac.load,
    dask .compute() triggering actual COG reads) so that transient HTTP
    errors, timeouts, or corrupted assets don't kill the whole run.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt == max_retries:
                log(f"  ✗ {desc} failed permanently after {max_retries} attempts: {e}")
                raise
            wait = RETRY_BACKOFF_BASE * attempt
            log(f"  ⚠ {desc} failed (attempt {attempt}/{max_retries}): {e}")
            log(f"    Retrying in {wait}s...")
            time.sleep(wait)
    raise last_exc  # unreachable, but keeps linters happy


def month_range(start_year, start_month, end_year, end_month):
    """Generate an ordered list of (year, month) tuples inclusive of both ends."""
    months = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def month_datetime_range(year, month):
    """Return an ISO 8601 'start/end' datetime string covering the full month."""
    last_day = calendar.monthrange(year, month)[1]
    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month:02d}-{last_day:02d}"
    return f"{start}/{end}"


def get_memory_usage_mb():
    """Current process RSS memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def load_completed_months(csv_path, n_colonies):
    """
    Inspect the existing output CSV (if any) and return the set of
    (year, month) tuples that are FULLY written — i.e. have exactly
    n_colonies rows. Partial months (from a crash mid-write) are treated
    as incomplete and will be recomputed and overwritten.
    """
    if not os.path.exists(csv_path):
        return set(), 0

    try:
        existing = pd.read_csv(csv_path)
    except Exception as e:
        log(f"⚠ Could not read existing CSV ({e}); starting fresh.")
        return set(), 0

    if existing.empty:
        return set(), 0

    counts = existing.groupby(["year", "month"]).size()
    completed = {
        (int(y), int(m)) for (y, m), c in counts.items() if c == n_colonies
    }
    return completed, len(existing)


def drop_incomplete_months_from_csv(csv_path, completed_months, n_colonies):
    """
    Rewrite the CSV keeping ONLY rows belonging to fully-completed months.
    This cleans up any partial month left behind by a crash, so that when
    we re-append that month's fresh results we don't get duplicate rows.
    """
    if not os.path.exists(csv_path):
        return

    existing = pd.read_csv(csv_path)
    if existing.empty:
        return

    mask = existing.apply(
        lambda r: (int(r["year"]), int(r["month"])) in completed_months, axis=1
    )
    cleaned = existing[mask]
    cleaned.to_csv(csv_path, index=False)


def append_month_to_csv(csv_path, df_month):
    """Append one month's results to the CSV, writing the header only once."""
    write_header = not os.path.exists(csv_path)
    df_month.to_csv(csv_path, mode="a", header=write_header, index=False)


# ==============================================================================
# STEP FUNCTIONS (each mirrors one stage of the required pipeline)
# ==============================================================================

def search_month_items(catalog, bbox_wgs84, year, month):
    """
    STEP 1: Search STAC for all Sentinel-2 L2A items intersecting the Delhi
    bounding box for the given month. No pixel data is touched here —
    this only returns metadata (item/asset references).
    """
    def _search():
        search = catalog.search(
            collections=[COLLECTION],
            bbox=bbox_wgs84,
            datetime=month_datetime_range(year, month),
        )
        return list(search.items())

    return retry_operation(_search, desc=f"STAC search {year}-{month:02d}")


def cloud_filter_items(items, threshold=CLOUD_COVER_THRESHOLD):
    """
    STEP 2: Drop items whose eo:cloud_cover exceeds the threshold.
    Items missing the property are conservatively kept (cloud cover
    unknown != cloud cover bad), but real Sentinel-2 items on Planetary
    Computer always carry this property.
    """
    filtered = [
        it for it in items
        if it.properties.get("eo:cloud_cover", 0) <= threshold
    ]
    return filtered


def load_and_composite(items, bbox_wgs84, year, month):
    """
    STEPS 3-4: Load ONLY bands B04/B08, spatially restricted to the Delhi
    bbox at read time (odc-stac clips at the COG/window level — full tiles
    are never downloaded), then collapse the time dimension via median
    to build the monthly cloud-filtered composite.

    Returns:
        b04 (np.ndarray float32), b08 (np.ndarray float32), affine transform, crs
    or (None, None, None, None) if nothing could be loaded.
    """
    def _load():
        # Sign each item's assets with a fresh SAS token via planetary_computer.
        signed_items = [pc.sign(it) for it in items]

        ds = odc.stac.load(
            signed_items,
            bands=BANDS,
            bbox=bbox_wgs84,          # <-- spatial restriction happens HERE,
                                       #     before any pixels are read.
            crs=TARGET_CRS,
            resolution=TARGET_RESOLUTION,
            resampling="bilinear",
            groupby="solar_day",      # merge same-day acquisitions
            chunks={"x": 2048, "y": 2048},  # dask-backed: lazy until .compute()
        )
        return ds

    ds = retry_operation(_load, desc=f"odc-stac load {year}-{month:02d}")

    if ds is None or "B04" not in ds or ds.sizes.get("time", 0) == 0:
        return None, None, None, None

    def _compute_median():
        # Median across the time axis = the monthly composite.
        # .compute() is where actual network reads of pixel windows happen —
        # wrapped in retry to survive transient read failures on individual
        # (possibly corrupted) COG assets.
        b04_med = ds["B04"].median(dim="time", skipna=True).compute()
        b08_med = ds["B08"].median(dim="time", skipna=True).compute()
        return b04_med, b08_med

    b04_med, b08_med = retry_operation(
        _compute_median, desc=f"composite compute {year}-{month:02d}"
    )

    # Pull out plain NumPy arrays + georeferencing info; drop xarray wrappers.
    geobox = ds.odc.geobox
    affine = geobox.affine
    crs = geobox.crs

    b04 = b04_med.values.astype("float32")
    b08 = b08_med.values.astype("float32")

    # Free the (potentially large) lazy dataset and intermediate DataArrays.
    del ds, b04_med, b08_med
    gc.collect()

    return b04, b08, affine, crs


def compute_ndvi(b04, b08):
    """
    STEP 5: NDVI = (B08 - B04) / (B08 + B04)
    Division-by-zero and Sentinel-2 nodata (0) pixels are masked to NaN
    so they contribute nothing to the zonal mean.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = b08 + b04
        ndvi = np.where(
            (denom == 0) | (b04 == 0) | (b08 == 0),
            np.nan,
            (b08 - b04) / denom,
        )
    return ndvi.astype("float32")


def compute_zonal_ndvi(gdf_target_crs, ndvi_array, affine):
    """
    STEP 6: Zonal statistics computed DIRECTLY on the in-memory NDVI array
    (rasterstats accepts a raw NumPy array + affine transform) — no raster
    file, temporary or permanent, is ever written to disk.
    """
    stats = zonal_stats(
        gdf_target_crs,
        ndvi_array,
        affine=affine,
        stats=["mean"],
        nodata=np.nan,
        all_touched=True,
        geojson_out=False,
    )
    return [s["mean"] for s in stats]


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    overall_start = time.time()

    log("=" * 60)
    log("UrbanCool — Delhi Colony Monthly NDVI Pipeline")
    log("=" * 60)

    # ---- Load colony shapefile -------------------------------------------
    log(f"Loading colony shapefile: {SHAPEFILE_PATH}")
    gdf = gpd.read_file(SHAPEFILE_PATH)
    n_colonies = len(gdf)
    log(f"✓ Loaded {n_colonies} colony polygons")

    # Verify required attribute columns exist; fall back gracefully if not.
    for col in ATTRIBUTE_COLUMNS:
        if col not in gdf.columns:
            log(f"⚠ Column '{col}' not found in shapefile attributes — filling with NaN.")
            gdf[col] = np.nan

    # Ensure a defined CRS (assume EPSG:4326 if missing — adjust if you know otherwise).
    if gdf.crs is None:
        log("⚠ Shapefile has no CRS defined — assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")

    # Bounding box for STAC search (must be lon/lat WGS84).
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    bbox_wgs84 = list(gdf_wgs84.total_bounds)  # [minx, miny, maxx, maxy]
    log(f"✓ Delhi bounding box (WGS84): {[round(b, 4) for b in bbox_wgs84]}")

    # Pre-reproject colonies ONCE into the raster's target CRS for zonal stats.
    gdf_target = gdf.to_crs(TARGET_CRS)
    # Keep only what we need, in stable order, so row order always matches gdf_target order.
    attrs = gdf_target[ATTRIBUTE_COLUMNS].reset_index(drop=True)
    gdf_geom_only = gdf_target[["geometry"]].reset_index(drop=True)

    # ---- Open the Planetary Computer STAC catalog -------------------------
    log(f"Connecting to Planetary Computer STAC API...")
    catalog = retry_operation(
        pystac_client.Client.open,
        STAC_API_URL,
        modifier=pc.sign_inplace,
        desc="STAC catalog connection",
    )
    log("✓ Connected.")

    # ---- Build the full month list and detect resume point ----------------
    all_months = month_range(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    total_months = len(all_months)

    completed_months, existing_rows = load_completed_months(OUTPUT_CSV, n_colonies)
    if completed_months:
        log(f"✓ Found existing CSV with {existing_rows} rows "
            f"({len(completed_months)} months already complete).")
        # Clean out any partial/dangling month before resuming.
        drop_incomplete_months_from_csv(OUTPUT_CSV, completed_months, n_colonies)
    else:
        log("No existing valid checkpoint found — starting from scratch.")

    months_to_process = [m for m in all_months if m not in completed_months]

    if not months_to_process:
        log("All months already complete. Nothing to do.")
        return

    log(f"Months remaining: {len(months_to_process)} / {total_months}")
    log("")

    # ---- Main per-month loop ------------------------------------------------
    month_durations = []

    for idx_in_run, (year, month) in enumerate(
        tqdm(months_to_process, desc="Overall progress", unit="month")
    ):
        month_start_time = time.time()
        month_label = f"{year}-{month:02d}"

        # overall index across the FULL 96-month timeline (for reporting)
        overall_index = all_months.index((year, month)) + 1

        log("=" * 60)
        log(f"Processing {month_label}")
        log("=" * 60)

        try:
            # ---- STEP 1: Search ----
            log("Searching STAC...")
            items = search_month_items(catalog, bbox_wgs84, year, month)
            log(f"✓ Found {len(items)} images")

            # ---- STEP 2: Cloud filter ----
            log("Cloud filtering...")
            clean_items = cloud_filter_items(items)
            log(f"✓ {len(clean_items)} images remain "
                f"(cloud cover <= {CLOUD_COVER_THRESHOLD}%)")

            if len(clean_items) == 0:
                # Gracefully handle zero-image months: write NaN for every
                # colony so the month is marked "complete" and skipped on resume.
                log("⚠ No usable images this month — writing NaN placeholder rows.")
                mean_ndvi_values = [np.nan] * n_colonies
            else:
                # ---- STEPS 3-4: Load (bbox-restricted) + monthly composite ----
                log("Loading imagery (bbox-restricted, bands B04/B08 only)...")
                b04, b08, affine, raster_crs = load_and_composite(
                    clean_items, bbox_wgs84, year, month
                )

                if b04 is None:
                    log("⚠ Composite could not be built — writing NaN placeholder rows.")
                    mean_ndvi_values = [np.nan] * n_colonies
                else:
                    log("Creating monthly median composite... ✓")

                    # ---- STEP 5: NDVI ----
                    log("Computing NDVI...")
                    ndvi = compute_ndvi(b04, b08)
                    del b04, b08
                    gc.collect()

                    # ---- STEP 6: Zonal statistics ----
                    log("Running zonal statistics...")
                    mean_ndvi_values = compute_zonal_ndvi(gdf_geom_only, ndvi, affine)
                    log(f"✓ Finished {n_colonies} colonies")

                    del ndvi
                    gc.collect()

            # ---- STEP 7: Assemble + append to CSV ----
            df_month = attrs.copy()
            df_month["year"] = year
            df_month["month"] = month
            df_month["mean_ndvi"] = mean_ndvi_values
            df_month = df_month[["zone", "ward", "colony", "year", "month", "mean_ndvi"]]

            log("Appending to CSV...")
            append_month_to_csv(OUTPUT_CSV, df_month)
            current_rows = existing_rows + sum(
                1 for _ in range(n_colonies)
            )  # rows just written this month (n_colonies)
            existing_rows += n_colonies
            log(f"✓ CSV now contains {existing_rows} rows")

            del df_month, mean_ndvi_values
            gc.collect()
            log("Memory released.")

        except Exception as e:
            # Any unrecoverable error for this month: log it clearly and
            # move on. Nothing for this month gets written, so on the next
            # run it will be correctly detected as incomplete and retried.
            log(f"✗ FAILED to process {month_label}: {e}")
            log("  Skipping this month for now — it will be retried on next run.")

        # ---- Timing / progress reporting ----
        elapsed_month = time.time() - month_start_time
        month_durations.append(elapsed_month)
        avg_month_time = sum(month_durations) / len(month_durations)

        months_left_total = total_months - overall_index
        est_remaining_seconds = months_left_total * avg_month_time
        est_remaining_minutes = est_remaining_seconds / 60

        completion_pct = 100 * overall_index / total_months
        mem_mb = get_memory_usage_mb()

        log(f"Elapsed time: {elapsed_month:.1f} seconds")
        log(f"Estimated remaining time: {est_remaining_minutes:.1f} minutes")
        log("-" * 60)
        log(f"Overall progress: Month {overall_index} / {total_months}")
        log(f"Overall completion: {completion_pct:.1f}%")

        if est_remaining_seconds >= 3600:
            h = int(est_remaining_seconds // 3600)
            m = int((est_remaining_seconds % 3600) // 60)
            log(f"Estimated finish: {h}h {m}m remaining")
        else:
            m = int(est_remaining_seconds // 60)
            log(f"Estimated finish: {m}m remaining")

        log(f"Current CSV size: {existing_rows} rows")
        log(f"Current memory usage: {mem_mb:.1f} MB")
        log("")

    total_elapsed = time.time() - overall_start
    log("=" * 60)
    log(f"Pipeline run complete. Total elapsed: {total_elapsed/60:.1f} minutes")
    log(f"Final CSV: {OUTPUT_CSV}")
    log("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n⚠ Interrupted by user. Progress up to the last completed month "
            "is safely saved in the CSV. Re-run the script to resume.")
        sys.exit(1)