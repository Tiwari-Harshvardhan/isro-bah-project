#!/usr/bin/env python3
"""
================================================================================
 UrbanCool — Delhi Zone Built-Up Percent Pipeline (2018-2022)
================================================================================
Computes, for every ZONE (colonies dissolved into their parent zone) in
delhi_colonies.shp, the percentage of land classified as "Built-up" by the
ESA WorldCover 10m global land cover product, hosted on Microsoft Planetary
Computer's STAC API.

IMPORTANT DATA-AVAILABILITY NOTE:
  ESA WorldCover only has TWO real snapshots: 2020 and 2021. There is no
  monthly, or even annual, built-up layer covering 2018-2025 the way
  Sentinel-2/Landsat do. This script therefore:
    1. Fetches the two REAL years (2020, 2021) from actual satellite-derived
       land cover pixels.
    2. Derives 2018, 2019, and 2022 by linear extrapolation along the trend
       line between the two real years:
           value(year) = v2020 + (v2021 - v2020) * (year - 2020)
       (2020 and 2021 themselves fall out of this same formula exactly.)
    3. Expands every year into 12 monthly rows (built-up % doesn't change
       month to month, so each month within a year repeats that year's
       value), written out in strict chronological order:
           2018-01, 2018-02, ..., 2018-12, 2019-01, ..., 2022-12

  A `method` column marks each row "computed" (real pixels) or
  "extrapolated" (projected), so real and estimated values are never
  ambiguous.

PIPELINE (Stage 1 — real data, the only stage that touches the network):
    Dissolve colonies -> zones
    -> Search ESA WorldCover STAC for 2020, then 2021
    -> bbox-restricted read of the "map" (classification) band
    -> Categorical zonal statistics per zone (pixel counts per class)
    -> built_up_percent = built-up pixel count / total valid pixel count
    -> Checkpoint each year immediately to a small anchor file
    -> Release memory

PIPELINE (Stage 2 — pure arithmetic, no network calls):
    For each (year, month) 2018-01 .. 2022-12, in order:
    -> Compute per-zone value via the trend formula above
    -> Append one row per zone to the final CSV
    -> Checkpoint immediately

RESUMABILITY:
  Both stages independently detect what's already been written and skip it.
  If your connection drops mid-fetch in Stage 1, or the script is
  interrupted mid-write in Stage 2, simply re-running the script picks up
  exactly where it left off — nothing is recomputed or duplicated.

No raster files (temporary or permanent) are ever written to disk. Zonal
statistics run directly on in-memory NumPy arrays via rasterstats.
================================================================================
"""

import os
import gc
import sys
import time
import warnings

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

SHAPEFILE_PATH   = "/content/drive/MyDrive/urbancool/delhi/delhi_colonies.shp"

# Stage 1 output: the two real, satellite-derived anchor years (internal checkpoint file).
ANCHOR_CSV       = "/content/drive/MyDrive/urbancool/delhi/Delhi_Zone_BuiltUp_Anchors_2020_2021.csv"

# Stage 2 output: the final monthly deliverable.
OUTPUT_CSV       = "/content/drive/MyDrive/urbancool/delhi/Delhi_Zone_Monthly_BuiltUp_Percent_2018_2022.csv"

STAC_API_URL     = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION       = "esa-worldcover"

# The only two years ESA WorldCover actually provides.
ANCHOR_YEARS     = [2020, 2021]

# Full monthly output range requested.
START_YEAR, START_MONTH = 2018, 1
END_YEAR, END_MONTH     = 2022, 12

TARGET_CRS        = "EPSG:32643"   # Delhi UTM Zone 43N
TARGET_RESOLUTION = 10             # metres (native WorldCover resolution)
BAND               = "map"         # ESA WorldCover classification band
BUILT_UP_CLASS     = 50            # LCCS code for "Built-up"

MAX_RETRIES        = 5
RETRY_BACKOFF_BASE = 5   # seconds

ATTRIBUTE_COLUMNS_INPUT = ["zone"]   # dissolve key from the colony shapefile

INCLUDE_METHOD_COLUMN = True
CSV_COLUMNS = ["zone", "year", "month", "built_up_percent"]
if INCLUDE_METHOD_COLUMN:
    CSV_COLUMNS = CSV_COLUMNS + ["method"]


# ==============================================================================
# SMALL UTILITIES
# ==============================================================================

def log(msg):
    print(msg, flush=True)


def retry_operation(func, *args, max_retries=MAX_RETRIES, desc="operation", **kwargs):
    """Retry-with-exponential-backoff wrapper for every network-touching call."""
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
    raise last_exc


def month_range(start_year, start_month, end_year, end_month):
    """Chronologically ordered list of (year, month): Jan->Dec, year by year."""
    months = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def get_memory_usage_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


# ==============================================================================
# STAGE 1 — fetch real built-up % for 2020 and 2021
# ==============================================================================

def load_anchor_checkpoint(n_zones):
    """Return (dataframe, set of years already fully present)."""
    if not os.path.exists(ANCHOR_CSV):
        return pd.DataFrame(columns=["zone", "year", "built_up_percent"]), set()
    try:
        df = pd.read_csv(ANCHOR_CSV)
    except Exception as e:
        log(f"⚠ Could not read anchor checkpoint ({e}); starting fresh.")
        return pd.DataFrame(columns=["zone", "year", "built_up_percent"]), set()

    complete_years = set()
    for year in ANCHOR_YEARS:
        if len(df[df["year"] == year]) == n_zones:
            complete_years.add(year)
    return df, complete_years


def append_anchor_year(df_year):
    write_header = not os.path.exists(ANCHOR_CSV)
    df_year.to_csv(ANCHOR_CSV, mode="a", header=write_header, index=False)


def fetch_anchor_year(catalog, bbox_wgs84, zones_gdf, year):
    """
    Search + load + zonal-stat ESA WorldCover for a single real year.
    Returns a list of built_up_percent values in zones_gdf row order.
    """
    def _search():
        search = catalog.search(
            collections=[COLLECTION],
            bbox=bbox_wgs84,
            datetime=f"{year}-01-01/{year}-12-31",
        )
        return list(search.items())

    log("Searching STAC...")
    items = retry_operation(_search, desc=f"ESA WorldCover search {year}")
    log(f"✓ Found {len(items)} tile(s)")

    if len(items) == 0:
        log(f"⚠ No ESA WorldCover tiles found for {year} — writing NaN for all zones.")
        return [np.nan] * len(zones_gdf)

    def _load():
        signed_items = [pc.sign(it) for it in items]
        return odc.stac.load(
            signed_items,
            bands=[BAND],
            bbox=bbox_wgs84,           # spatial restriction at read time
            crs=TARGET_CRS,
            resolution=TARGET_RESOLUTION,
            resampling="nearest",      # categorical data -> never interpolate classes
            chunks={"x": 2048, "y": 2048},
        )

    log("Loading classification raster (bbox-restricted)...")
    ds = retry_operation(_load, desc=f"odc-stac load {year}")

    if ds is None or BAND not in ds:
        log(f"⚠ Could not load classification band for {year} — writing NaN for all zones.")
        return [np.nan] * len(zones_gdf)

    def _compute():
        # If multiple tiles cover the AOI, take the first valid classification
        # per pixel (WorldCover tiles don't overlap in time, just mosaic).
        arr = ds[BAND]
        if "time" in arr.dims:
            arr = arr.max(dim="time", skipna=True)
        return arr.compute()

    classification = retry_operation(_compute, desc=f"raster compute {year}")

    geobox = ds.odc.geobox
    affine = geobox.affine
    class_array = classification.values.astype("float32")

    del ds, classification
    gc.collect()

    log("Running categorical zonal statistics...")
    stats = zonal_stats(
        zones_gdf, class_array, affine=affine,
        categorical=True, nodata=0, all_touched=False, geojson_out=False,
    )

    built_up_percents = []
    for s in stats:
        total_pixels = sum(s.values())
        built_up_pixels = s.get(BUILT_UP_CLASS, 0)
        pct = (built_up_pixels / total_pixels * 100.0) if total_pixels > 0 else np.nan
        built_up_percents.append(pct)

    del class_array
    gc.collect()

    log(f"✓ Finished {len(zones_gdf)} zones")
    return built_up_percents


def run_stage1(catalog, bbox_wgs84, zones_gdf):
    """Ensure both anchor years (2020, 2021) are present in ANCHOR_CSV."""
    n_zones = len(zones_gdf)
    log("")
    log("########## STAGE 1: Fetching real built-up data (2020, 2021) ##########")

    _, complete_years = load_anchor_checkpoint(n_zones)
    years_to_fetch = [y for y in ANCHOR_YEARS if y not in complete_years]

    if not years_to_fetch:
        log("✓ Both anchor years already present — skipping Stage 1 entirely.")
        return

    for year in years_to_fetch:
        t0 = time.time()
        log("=" * 60)
        log(f"Processing ESA WorldCover {year}  [REAL / COMPUTED]")
        log("=" * 60)
        try:
            values = fetch_anchor_year(catalog, bbox_wgs84, zones_gdf, year)
            df_year = pd.DataFrame({
                "zone": zones_gdf["zone"].values,
                "year": year,
                "built_up_percent": values,
            })
            append_anchor_year(df_year)
            log(f"✓ Anchor checkpoint updated for {year} ({n_zones} zones)")
            del df_year, values
            gc.collect()
        except Exception as e:
            log(f"✗ FAILED to fetch {year}: {e}")
            log("  This year will be retried the next time you run the script.")

        log(f"Elapsed: {time.time() - t0:.1f} seconds")
        log(f"Memory usage: {get_memory_usage_mb():.1f} MB")
        log("")


# ==============================================================================
# STAGE 2 — expand into monthly rows, interpolating/extrapolating as needed
# ==============================================================================

def month_is_complete(existing_df, year, month, n_zones):
    if existing_df.empty:
        return False
    sub = existing_df[(existing_df["year"] == year) & (existing_df["month"] == month)]
    return len(sub) == n_zones


def load_existing_output():
    if os.path.exists(OUTPUT_CSV):
        try:
            return pd.read_csv(OUTPUT_CSV)
        except Exception as e:
            log(f"⚠ Could not read existing output CSV ({e}); starting fresh.")
    return pd.DataFrame(columns=CSV_COLUMNS)


def append_month_to_output(df_month):
    write_header = not os.path.exists(OUTPUT_CSV)
    df_month.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)


def run_stage2(zones_gdf):
    """Expand the two real anchor years into full monthly rows for 2018-2022,
    in strict chronological order, interpolating/extrapolating as needed."""
    n_zones = len(zones_gdf)
    log("")
    log("########## STAGE 2: Building monthly CSV (2018-2022, no network calls) ##########")

    anchor_df = pd.read_csv(ANCHOR_CSV)
    v2020 = anchor_df[anchor_df["year"] == 2020].set_index("zone")["built_up_percent"]
    v2021 = anchor_df[anchor_df["year"] == 2021].set_index("zone")["built_up_percent"]

    # Align to the zones_gdf order so output rows are always in the same
    # stable zone ordering every month.
    v2020 = v2020.reindex(zones_gdf["zone"].values)
    v2021 = v2021.reindex(zones_gdf["zone"].values)

    def value_for_year(year):
        """Linear trend through the two real anchors; exact at 2020/2021,
        extrapolated linearly outside that range. Clipped to a valid
        0-100% range since raw extrapolation has no such guarantee."""
        raw = v2020 + (v2021 - v2020) * (year - 2020)
        return np.clip(raw.to_numpy(), 0.0, 100.0)

    all_months = month_range(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    total_months = len(all_months)

    existing_df = load_existing_output()
    months_to_run = [
        (y, m) for (y, m) in all_months
        if not month_is_complete(existing_df, y, m, n_zones)
    ]

    if not months_to_run:
        log("✓ All months already written — nothing to do.")
        return

    row_count = len(existing_df)

    for (year, month) in tqdm(months_to_run, desc="Writing monthly rows", unit="month"):
        overall_index = all_months.index((year, month)) + 1
        values = value_for_year(year)
        method = "computed" if year in ANCHOR_YEARS else "extrapolated"

        df_month = pd.DataFrame({
            "zone": zones_gdf["zone"].values,
            "year": year,
            "month": month,
            "built_up_percent": values,
        })
        if INCLUDE_METHOD_COLUMN:
            df_month["method"] = method
        df_month = df_month[CSV_COLUMNS]

        append_month_to_output(df_month)
        row_count += n_zones

        if overall_index % 12 == 0 or overall_index == total_months:
            # Log once per year-boundary to keep output readable (60 months
            # would otherwise print 60 near-identical blocks).
            log(f"✓ Through {year}-{month:02d}  "
                f"[{method}]  |  Month {overall_index}/{total_months}  "
                f"({100*overall_index/total_months:.1f}%)  |  "
                f"CSV size: {row_count} rows")

        del df_month
        gc.collect()

    log(f"Final memory usage: {get_memory_usage_mb():.1f} MB")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    overall_start = time.time()

    log("=" * 60)
    log("UrbanCool — Delhi Zone Built-Up Percent (2018-2022)")
    log("=" * 60)

    # ---- Load + dissolve shapefile into zone-level polygons -----------------
    log(f"Loading colony shapefile: {SHAPEFILE_PATH}")
    gdf = gpd.read_file(SHAPEFILE_PATH)
    log(f"✓ Loaded {len(gdf)} colony polygons")

    if "zone" not in gdf.columns:
        log("✗ ERROR: shapefile has no 'zone' column — cannot dissolve into zones.")
        sys.exit(1)

    if gdf.crs is None:
        log("⚠ Shapefile has no CRS — assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")

    # ============================================================
    # FIX: Repair invalid geometries before dissolving
    # ============================================================
    log("Checking geometry validity...")
    invalid_count = (~gdf.geometry.is_valid).sum()
    if invalid_count > 0:
        log(f"⚠ Found {invalid_count} invalid geometries — repairing with make_valid()...")
        gdf["geometry"] = gdf.geometry.make_valid()
        # Re-check after repair
        still_invalid = (~gdf.geometry.is_valid).sum()
        if still_invalid > 0:
            log(f"⚠ {still_invalid} geometries remain invalid after make_valid() — "
                f"buffer(0) may fix them.")
            gdf["geometry"] = gdf.geometry.buffer(0)
    else:
        log("✓ All geometries are valid.")

    log("Dissolving colonies into zone-level polygons...")
    zones = gdf.dissolve(by="zone", as_index=False)[["zone", "geometry"]]
    n_zones = len(zones)
    log(f"✓ {n_zones} zones")

    zones_wgs84 = zones.to_crs("EPSG:4326")
    bbox_wgs84 = list(zones_wgs84.total_bounds)
    log(f"✓ Delhi bounding box (WGS84): {[round(b, 4) for b in bbox_wgs84]}")

    zones_target = zones.to_crs(TARGET_CRS).reset_index(drop=True)

    # ---- Connect to STAC (only needed if Stage 1 has work to do) ------------
    log("Connecting to Planetary Computer STAC API...")
    catalog = retry_operation(
        pystac_client.Client.open, STAC_API_URL,
        modifier=pc.sign_inplace, desc="STAC catalog connection",
    )
    log("✓ Connected.")

    # ---- Stage 1: real anchor years -----------------------------------------
    run_stage1(catalog, bbox_wgs84, zones_target)

    # ---- Stage 2: monthly expansion with interpolation/extrapolation --------
    run_stage2(zones_target)

    total_elapsed = time.time() - overall_start
    log("=" * 60)
    log(f"Pipeline complete. Total elapsed: {total_elapsed/60:.1f} minutes")
    log(f"Final CSV: {OUTPUT_CSV}")
    log("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n⚠ Interrupted. Whatever was already checkpointed (anchor years "
            "and/or monthly rows) is safe. Re-run the script to resume exactly "
            "where it left off.")
        sys.exit(1)