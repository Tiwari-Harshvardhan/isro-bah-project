#!/usr/bin/env python3
"""
================================================================================
 UrbanCool — Delhi Colony Monthly NDVI Pipeline (Half-Compute / Interpolated)
================================================================================
Designed to run on the Microsoft Planetary Computer JupyterHub
(https://planetarycomputer.microsoft.com/compute) — physically co-located
with the Sentinel-2 data, so reads are fast, and the terminal session can
survive you closing your laptop entirely (see "RUNNING THIS ON PC HUB" below).

STRATEGY (halves real satellite computation):
  Of the 96 months (Jan 2018 - Dec 2025), only every OTHER month is actually
  computed from Sentinel-2 imagery (~48 months = exactly 6/year):
        Jan(compute) Feb(skip) Mar(compute) Apr(skip) May(compute) ...
  The skipped months are filled in afterwards by INTERPOLATING between their
  two real neighbours (e.g. Feb = average of real Jan and real Mar).

  Why interpolation instead of pure forward-extrapolation (projecting Feb
  from Jan alone)? Because by the time we need Feb, Mar has *also* already
  been computed (Pass 1 computes all anchor months first) — so using both
  real neighbours costs nothing extra and is virtually always more accurate
  for a smooth seasonal signal like NDVI than a one-sided projection.
  The very last month in the whole range (Dec 2025) has no "next" anchor,
  so it falls back to linear trend extrapolation from the last two anchors.

  A `method` column ("computed" / "interpolated") is included so real vs.
  estimated values are always distinguishable. Set INCLUDE_METHOD_COLUMN =
  False below if you want strictly the original 6-column schema.

PIPELINE PER COMPUTED MONTH (unchanged from the full pipeline):
    Search STAC -> Scene-level cloud filter -> Load B04, B08, SCL
    -> Pixel-level cloud masking using SCL -> Monthly median composite
    -> NDVI -> Vegetation quality filtering -> Zonal stats -> Append to CSV
  No Sentinel imagery or GeoTIFFs are ever saved to disk, temporarily or
  permanently. Zonal stats run directly on in-memory arrays via rasterstats.

================================================================================
RUNNING THIS ON PLANETARY COMPUTER'S JUPYTERHUB
================================================================================
1. Go to https://planetarycomputer.microsoft.com/compute and launch a server
   (the default "Pangeo Notebook" environment already has pystac-client,
   planetary-computer, odc-stac, geopandas, and rasterstats preinstalled;
   `pip install rasterstats psutil tqdm` if rasterstats/psutil are missing).
2. Upload delhi_colonies.shp (+ .dbf/.shx/.prj) into your Hub home directory
   via the Jupyter file browser (drag & drop).
3. Open a Terminal from the Jupyter Launcher (not a notebook cell) and run:

       tmux new -s ndvi
       python delhi_colony_ndvi_pipeline_alt.py

   `tmux` keeps the process running on Microsoft's server even if you close
   your laptop or lose your internet connection. Reconnect later from any
   device by opening a new Terminal on the Hub and running:

       tmux attach -t ndvi

   (If tmux isn't installed, `nohup python delhi_colony_ndvi_pipeline_alt.py
   > run.log 2>&1 &` works too — just check run.log for progress instead of
   live terminal output.)
4. When finished, download Delhi_Colony_Monthly_NDVI_2018_2025.csv from the
   Jupyter file browser back to your laptop.
================================================================================
"""

import os
import gc
import sys
import time
import calendar
import warnings
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import geopandas as gpd

import pystac_client
import planetary_computer as pc
import odc.stac
import xarray as xr

from rasterstats import zonal_stats

import psutil
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SHAPEFILE_PATH   = "/content/drive/MyDrive/urbancool/delhi/delhi_colonies.shp"
OUTPUT_CSV       = "/content/drive/MyDrive/urbancool/delhi/Delhi_Colony_Monthly_NDVI_2018_2025.csv"

STAC_API_URL     = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION       = "sentinel-2-l2a"

START_YEAR, START_MONTH = 2018, 1
END_YEAR, END_MONTH     = 2025, 12

# Scene-level cloud cover threshold (percent)
SCENE_CLOUD_COVER_MAX = 40

# Target CRS and resolution
TARGET_CRS            = "EPSG:32643"   # Delhi is entirely UTM Zone 43N
TARGET_RESOLUTION     = 10             # metres (native B04/B08 resolution)

# Sentinel-2 bands
BANDS                 = ["B04", "B08", "SCL"]  # Now includes SCL for cloud masking

# SCL (Scene Classification Layer) pixel values
# Source: https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
SCL_CLASSES = {
    'NO_DATA': 0,
    'SATURATED_OR_DEFECTIVE': 1,
    'DARK_AREA': 2,
    'CLOUD_SHADOW': 3,
    'VEGETATION': 4,
    'NOT_VEGETATED': 5,
    'WATER': 6,
    'UNCLASSIFIED': 7,
    'CLOUD_MEDIUM_PROBABILITY': 8,
    'CLOUD_HIGH_PROBABILITY': 9,
    'THIN_CIRRUS': 10,
    'SNOW_ICE': 11
}

# Pixel values to mask out (clouds, shadows, cirrus, snow)
CLOUD_MASK_VALUES = [
    SCL_CLASSES['CLOUD_SHADOW'],           # 3
    SCL_CLASSES['CLOUD_MEDIUM_PROBABILITY'], # 8
    SCL_CLASSES['CLOUD_HIGH_PROBABILITY'],   # 9
    SCL_CLASSES['THIN_CIRRUS'],            # 10
    SCL_CLASSES['SNOW_ICE'],               # 11
    SCL_CLASSES['NO_DATA'],                # 0
    SCL_CLASSES['SATURATED_OR_DEFECTIVE'], # 1
    SCL_CLASSES['DARK_AREA'],              # 2
]

# Keep only vegetation pixels (optional - set to None to keep all valid pixels)
# VEGETATION_ONLY = [SCL_CLASSES['VEGETATION'], SCL_CLASSES['NOT_VEGETATED']]
VEGETATION_ONLY = None  # Keep all non-cloud pixels for NDVI

# NDVI quality thresholds
NDVI_MIN_VALID = -0.2   # Below this is likely water/no-data
NDVI_MAX_VALID = 1.0    # Above this is likely snow/cloud

MAX_RETRIES        = 5
RETRY_BACKOFF_BASE = 5     # seconds

ATTRIBUTE_COLUMNS = ["zone", "ward", "colony"]

# Include a "method" column ("computed"/"interpolated") in the output CSV.
INCLUDE_METHOD_COLUMN = True

# Include quality metrics
INCLUDE_QUALITY_METRICS = True

CSV_COLUMNS = ATTRIBUTE_COLUMNS + ["year", "month", "mean_ndvi"]
if INCLUDE_METHOD_COLUMN:
    CSV_COLUMNS = CSV_COLUMNS + ["method"]
if INCLUDE_QUALITY_METRICS:
    CSV_COLUMNS = CSV_COLUMNS + ["pixel_count", "valid_pixel_count", "cloud_pixel_percent"]

# ==============================================================================
# SMALL UTILITIES
# ==============================================================================

@dataclass
class MonthResult:
    """Container for monthly processing results."""
    mean_ndvi: np.ndarray
    pixel_count: np.ndarray
    valid_pixel_count: np.ndarray
    cloud_pixel_percent: np.ndarray


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
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}"


def get_memory_usage_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def load_existing_csv():
    """Load the existing output CSV if present, else return an empty frame."""
    if os.path.exists(OUTPUT_CSV):
        try:
            return pd.read_csv(OUTPUT_CSV)
        except Exception as e:
            log(f"⚠ Could not read existing CSV ({e}); starting fresh.")
    return pd.DataFrame(columns=CSV_COLUMNS)


def month_is_complete(existing_df, year, month, n_colonies):
    sub = existing_df[(existing_df["year"] == year) & (existing_df["month"] == month)]
    return len(sub) == n_colonies


def rewrite_csv_dropping_incomplete(existing_df, valid_months, n_colonies):
    """Keep only months that are fully present; rewrite the CSV clean."""
    if existing_df.empty:
        return existing_df
    keep_mask = existing_df.apply(
        lambda r: (int(r["year"]), int(r["month"])) in valid_months, axis=1
    )
    cleaned = existing_df[keep_mask].copy()
    cleaned.to_csv(OUTPUT_CSV, index=False)
    return cleaned


def append_month_to_csv(df_month):
    write_header = not os.path.exists(OUTPUT_CSV)
    df_month.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)


# ==============================================================================
# CLOUD MASKING FUNCTIONS
# ==============================================================================

def create_pixel_cloud_mask(scl_array: np.ndarray) -> np.ndarray:
    """
    Create a boolean mask from SCL where True = clear pixel (keep).
    Uses CLOUD_MASK_VALUES to identify pixels to discard.
    """
    mask = np.ones_like(scl_array, dtype=bool)
    for cloud_value in CLOUD_MASK_VALUES:
        mask = mask & (scl_array != cloud_value)

    # Optional: keep only vegetation pixels
    if VEGETATION_ONLY is not None:
        veg_mask = np.zeros_like(scl_array, dtype=bool)
        for veg_value in VEGETATION_ONLY:
            veg_mask = veg_mask | (scl_array == veg_value)
        mask = mask & veg_mask

    return mask


def apply_cloud_mask_to_bands(
    b04: np.ndarray,
    b08: np.ndarray,
    scl: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Apply pixel-level cloud mask to B04 and B08.
    Returns: masked_b04, masked_b08, cloud_mask, cloud_percent
    """
    # Create cloud mask
    cloud_mask = create_pixel_cloud_mask(scl)

    # Calculate cloud percentage
    total_pixels = scl.size
    cloud_pixels = np.sum(~cloud_mask)
    cloud_percent = (cloud_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    # Apply mask (set cloud pixels to NaN)
    b04_masked = b04.copy().astype(np.float32)
    b08_masked = b08.copy().astype(np.float32)
    b04_masked[~cloud_mask] = np.nan
    b08_masked[~cloud_mask] = np.nan

    return b04_masked, b08_masked, cloud_mask, cloud_percent


def filter_valid_ndvi(ndvi: np.ndarray) -> np.ndarray:
    """
    Filter NDVI values based on quality thresholds.
    Sets invalid values to NaN.
    """
    ndvi_filtered = ndvi.copy()
    ndvi_filtered[(ndvi < NDVI_MIN_VALID) | (ndvi > NDVI_MAX_VALID)] = np.nan
    return ndvi_filtered


# ==============================================================================
# SENTINEL-2 / NDVI PIPELINE STEPS
# ==============================================================================

def search_month_items(catalog, bbox_wgs84, year, month):
    def _search():
        search = catalog.search(
            collections=[COLLECTION],
            bbox=bbox_wgs84,
            datetime=month_datetime_range(year, month),
        )
        return list(search.items())
    return retry_operation(_search, desc=f"STAC search {year}-{month:02d}")


def cloud_filter_items(items, threshold=SCENE_CLOUD_COVER_MAX):
    """Filter items by scene-level cloud cover."""
    return [it for it in items if it.properties.get("eo:cloud_cover", 0) <= threshold]


def load_and_composite(items, bbox_wgs84, year, month):
    """
    Load B04, B08, and SCL bands.
    Apply pixel-level cloud masking.
    Return median composites and metadata.
    """
    def _load():
        signed_items = [pc.sign(it) for it in items]
        return odc.stac.load(
            signed_items,
            bands=BANDS,  # Now includes SCL
            bbox=bbox_wgs84,
            crs=TARGET_CRS,
            resolution=TARGET_RESOLUTION,
            resampling="bilinear",
            groupby="solar_day",
            chunks={"time": 1, "x": 2048, "y": 2048},
        )

    ds = retry_operation(_load, desc=f"odc-stac load {year}-{month:02d}")

    if ds is None or "B04" not in ds or ds.sizes.get("time", 0) == 0:
        return None, None, None, None

    # Get dimensions
    n_time = ds.sizes.get("time", 0)
    log(f"  Loaded {n_time} images with bands: {list(ds.data_vars)}")

    # Process each image with cloud masking
    b04_masked_list = []
    b08_masked_list = []
    cloud_percentages = []

    for t_idx in range(n_time):
        # Extract bands for this time step
        b04 = ds["B04"].isel(time=t_idx).values.astype(np.float32)
        b08 = ds["B08"].isel(time=t_idx).values.astype(np.float32)
        scl = ds["SCL"].isel(time=t_idx).values.astype(np.int8)

        # Apply pixel-level cloud mask
        b04_masked, b08_masked, _, cloud_pct = apply_cloud_mask_to_bands(b04, b08, scl)

        # Only keep images with enough clear pixels (at least 10% clear)
        if cloud_pct < 90:  # At least 10% clear pixels
            b04_masked_list.append(b04_masked)
            b08_masked_list.append(b08_masked)
            cloud_percentages.append(cloud_pct)

        # Clean up
        del b04, b08, scl, b04_masked, b08_masked

    if not b04_masked_list:
        log(f"  No images with sufficient clear pixels after cloud masking")
        return None, None, None, None

    log(f"  Kept {len(b04_masked_list)} images after pixel-level cloud masking")
    avg_cloud = np.mean(cloud_percentages) if cloud_percentages else 0
    log(f"  Average cloud cover after masking: {avg_cloud:.1f}%")

    # Stack masked bands
    b04_stacked = np.stack(b04_masked_list, axis=0)
    b08_stacked = np.stack(b08_masked_list, axis=0)

    # Compute median composite (ignoring NaN)
    b04_med = np.nanmedian(b04_stacked, axis=0).astype(np.float32)
    b08_med = np.nanmedian(b08_stacked, axis=0).astype(np.float32)

    # Count valid pixels used in median
    valid_counts = np.sum(~np.isnan(b04_stacked), axis=0)

    # Get geobox
    geobox = ds.odc.geobox
    affine = geobox.affine

    # Clean up
    del ds, b04_stacked, b08_stacked
    gc.collect()

    return b04_med, b08_med, affine, valid_counts


def compute_ndvi_with_quality(
    b04: np.ndarray,
    b08: np.ndarray,
    valid_counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute NDVI with quality metrics.
    Returns: ndvi, pixel_count, valid_pixel_count
    """
    # Compute NDVI
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = b08 + b04
        ndvi = np.where(
            (denom == 0) | (b04 == 0) | (b08 == 0) | np.isnan(b04) | np.isnan(b08),
            np.nan,
            (b08 - b04) / denom,
        )

    ndvi = ndvi.astype(np.float32)

    # Filter invalid NDVI values
    ndvi_filtered = filter_valid_ndvi(ndvi)

    # Quality metrics
    pixel_count = np.full_like(ndvi_filtered, ndvi.size, dtype=np.int32)
    valid_pixel_count = valid_counts.copy()

    return ndvi_filtered, pixel_count, valid_pixel_count


def compute_zonal_stats_with_quality(
    gdf_target_crs,
    ndvi_array: np.ndarray,
    pixel_count: np.ndarray,
    valid_pixel_count: np.ndarray,
    affine
) -> Tuple[List[float], List[int], List[int], List[float]]:
    """
    Compute zonal statistics with quality metrics for each polygon.
    Returns: mean_ndvi, pixel_counts, valid_counts, cloud_percentages
    """
    # Compute mean NDVI per polygon
    stats = zonal_stats(
        gdf_target_crs, ndvi_array, affine=affine,
        stats=["mean"], nodata=np.nan, all_touched=True, geojson_out=False,
    )
    mean_ndvi = [s["mean"] for s in stats]

    # Count pixels per polygon
    pixel_stats = zonal_stats(
        gdf_target_crs, pixel_count, affine=affine,
        stats=["count"], nodata=0, all_touched=True, geojson_out=False,
    )
    pixel_counts = [s["count"] for s in pixel_stats]

    # Count valid pixels per polygon
    valid_stats = zonal_stats(
        gdf_target_crs, valid_pixel_count, affine=affine,
        stats=["count"], nodata=0, all_touched=True, geojson_out=False,
    )
    valid_counts = [s["count"] for s in valid_stats]

    # Calculate cloud percentage per polygon
    cloud_percents = [
        100 * (1 - valid_counts[i] / pixel_counts[i]) if pixel_counts[i] > 0 else 100
        for i in range(len(pixel_counts))
    ]

    return mean_ndvi, pixel_counts, valid_counts, cloud_percents


def compute_one_month(
    catalog,
    bbox_wgs84,
    year,
    month,
    n_colonies
) -> Optional[MonthResult]:
    """
    Full real-satellite pipeline for one month with pixel-level cloud masking.
    Returns MonthResult object, or None if no usable imagery.
    """
    log("Searching STAC...")
    items = search_month_items(catalog, bbox_wgs84, year, month)
    log(f"✓ Found {len(items)} images")

    log("Scene-level cloud filtering...")
    clean_items = cloud_filter_items(items)
    log(f"✓ {len(clean_items)} images remain (scene cloud <= {SCENE_CLOUD_COVER_MAX}%)")

    if len(clean_items) == 0:
        log("⚠ No usable images this month — writing NaN placeholder rows.")
        return None

    log("Loading imagery (bbox-restricted, bands B04/B08/SCL)...")
    b04, b08, affine, valid_counts = load_and_composite(clean_items, bbox_wgs84, year, month)

    if b04 is None:
        log("⚠ Composite could not be built — writing NaN placeholder rows.")
        return None

    log("Computing NDVI with quality metrics...")
    ndvi, pixel_count, valid_pixel_count = compute_ndvi_with_quality(b04, b08, valid_counts)

    log("Running zonal statistics...")
    mean_ndvi, pix_counts, valid_counts, cloud_pcts = compute_zonal_stats_with_quality(
        GDF_GEOM_ONLY, ndvi, pixel_count, valid_pixel_count, affine
    )
    log(f"✓ Finished {n_colonies} colonies")

    # Clean up
    del b04, b08, ndvi, pixel_count, valid_pixel_count
    gc.collect()

    # Convert to arrays
    return MonthResult(
        mean_ndvi=np.array(mean_ndvi),
        pixel_count=np.array(pix_counts),
        valid_pixel_count=np.array(valid_counts),
        cloud_pixel_percent=np.array(cloud_pcts)
    )


# ==============================================================================
# MAIN
# ==============================================================================

GDF_GEOM_ONLY = None  # set in main(); module-level so compute_one_month can use it


def main():
    global GDF_GEOM_ONLY
    overall_start = time.time()

    log("=" * 60)
    log("UrbanCool — Delhi Colony Monthly NDVI (Half-Compute, Interpolated)")
    log("WITH PIXEL-LEVEL CLOUD MASKING USING SCL")
    log("=" * 60)

    # ---- Load colony shapefile -------------------------------------------
    log(f"Loading colony shapefile: {SHAPEFILE_PATH}")
    gdf = gpd.read_file(SHAPEFILE_PATH)
    n_colonies = len(gdf)
    log(f"✓ Loaded {n_colonies} colony polygons")

    for col in ATTRIBUTE_COLUMNS:
        if col not in gdf.columns:
            log(f"⚠ Column '{col}' not found — filling with NaN.")
            gdf[col] = np.nan

    if gdf.crs is None:
        log("⚠ Shapefile has no CRS — assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")

    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    bbox_wgs84 = list(gdf_wgs84.total_bounds)
    log(f"✓ Delhi bounding box (WGS84): {[round(b, 4) for b in bbox_wgs84]}")

    gdf_target = gdf.to_crs(TARGET_CRS)
    attrs = gdf_target[ATTRIBUTE_COLUMNS].reset_index(drop=True)
    GDF_GEOM_ONLY = gdf_target[["geometry"]].reset_index(drop=True)

    # ---- Connect to STAC ----------------------------------------------------
    log("Connecting to Planetary Computer STAC API...")
    catalog = retry_operation(
        pystac_client.Client.open, STAC_API_URL,
        modifier=pc.sign_inplace, desc="STAC catalog connection",
    )
    log("✓ Connected.")

    # ---- Build month schedule: alternating anchor/estimate ------------------
    all_months = month_range(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    total_months = len(all_months)

    # Even index (0,2,4,...) = ANCHOR (really computed). Odd index = ESTIMATED.
    anchor_months = [m for i, m in enumerate(all_months) if i % 2 == 0]
    estimated_months = [m for i, m in enumerate(all_months) if i % 2 == 1]

    log(f"Total months: {total_months} | "
        f"To compute: {len(anchor_months)} | To interpolate: {len(estimated_months)}")

    # ---- Resume detection -----------------------------------------------------
    existing_df = load_existing_csv()
    valid_month_set = set()
    if not existing_df.empty:
        for (y, m) in all_months:
            if month_is_complete(existing_df, y, m, n_colonies):
                valid_month_set.add((y, m))
        existing_df = rewrite_csv_dropping_incomplete(existing_df, valid_month_set, n_colonies)
        log(f"✓ Resuming: {len(valid_month_set)} months already complete "
            f"({len(existing_df)} rows).")
    else:
        log("No existing checkpoint found — starting from scratch.")

    row_count = len(existing_df)
    month_durations = []

    # ==========================================================================
    # PASS 1 — compute anchor months from real Sentinel-2 imagery
    # ==========================================================================
    log("")
    log("########## PASS 1: Computing anchor months from satellite data ##########")
    anchors_to_run = [m for m in anchor_months if m not in valid_month_set]

    for (year, month) in tqdm(anchors_to_run, desc="Pass 1 (compute)", unit="month"):
        t0 = time.time()
        overall_index = all_months.index((year, month)) + 1
        log("=" * 60)
        log(f"Processing {year}-{month:02d}  [COMPUTED anchor month]")
        log("=" * 60)

        try:
            result = compute_one_month(catalog, bbox_wgs84, year, month, n_colonies)

            if result is None:
                # No usable imagery - write NaN rows
                log("⚠ No usable data - writing NaN placeholder rows.")
                df_month = attrs.copy()
                df_month["year"] = year
                df_month["month"] = month
                df_month["mean_ndvi"] = np.nan
                if INCLUDE_METHOD_COLUMN:
                    df_month["method"] = "computed"
                if INCLUDE_QUALITY_METRICS:
                    df_month["pixel_count"] = 0
                    df_month["valid_pixel_count"] = 0
                    df_month["cloud_pixel_percent"] = 100
                df_month = df_month[CSV_COLUMNS]
            else:
                df_month = attrs.copy()
                df_month["year"] = year
                df_month["month"] = month
                df_month["mean_ndvi"] = result.mean_ndvi
                if INCLUDE_METHOD_COLUMN:
                    df_month["method"] = "computed"
                if INCLUDE_QUALITY_METRICS:
                    df_month["pixel_count"] = result.pixel_count
                    df_month["valid_pixel_count"] = result.valid_pixel_count
                    df_month["cloud_pixel_percent"] = result.cloud_pixel_percent
                df_month = df_month[CSV_COLUMNS]

            log("Appending to CSV...")
            append_month_to_csv(df_month)
            row_count += n_colonies
            log(f"✓ CSV now contains {row_count} rows")

            valid_month_set.add((year, month))
            del df_month
            if result is not None:
                del result
            gc.collect()
            log("Memory released.")

        except Exception as e:
            log(f"✗ FAILED to process {year}-{month:02d}: {e}")
            log("  Skipping — will retry on next run.")

        elapsed = time.time() - t0
        month_durations.append(elapsed)
        avg = sum(month_durations) / len(month_durations)
        remaining = (len(anchors_to_run) - anchors_to_run.index((year, month)) - 1) * avg

        log(f"Elapsed time: {elapsed:.1f} seconds")
        log(f"Estimated remaining time (Pass 1): {remaining/60:.1f} minutes")
        log("-" * 60)
        log(f"Overall progress: Month {overall_index} / {total_months}")
        log(f"Overall completion: {100*overall_index/total_months:.1f}%")
        log(f"Current CSV size: {row_count} rows")
        log(f"Current memory usage: {get_memory_usage_mb():.1f} MB")
        log("")

    # ==========================================================================
    # PASS 2 — interpolate the skipped months between real neighbours
    # ==========================================================================
    log("")
    log("########## PASS 2: Interpolating skipped months (no network calls) ##########")

    # Reload the CSV fresh so Pass 2 sees every anchor written in Pass 1.
    anchor_df = pd.read_csv(OUTPUT_CSV)
    anchor_df = anchor_df[anchor_df["year"].notna()]

    def get_anchor_values(year, month, include_quality=False):
        """Return the mean_ndvi array for a given already-computed month."""
        sub = anchor_df[(anchor_df["year"] == year) & (anchor_df["month"] == month)]
        if len(sub) != n_colonies:
            return None
        # Rows were always appended in `attrs` order, so position == colony order.
        result = {"mean_ndvi": sub["mean_ndvi"].to_numpy()}
        if include_quality and INCLUDE_QUALITY_METRICS:
            if "pixel_count" in sub.columns:
                result["pixel_count"] = sub["pixel_count"].to_numpy()
                result["valid_pixel_count"] = sub["valid_pixel_count"].to_numpy()
                result["cloud_pixel_percent"] = sub["cloud_pixel_percent"].to_numpy()
        return result

    estimates_to_run = [m for m in estimated_months if m not in valid_month_set]

    for (year, month) in tqdm(estimates_to_run, desc="Pass 2 (interpolate)", unit="month"):
        t0 = time.time()
        overall_index = all_months.index((year, month)) + 1
        idx = all_months.index((year, month))
        prev_month = all_months[idx - 1] if idx - 1 >= 0 else None
        next_month = all_months[idx + 1] if idx + 1 < total_months else None

        log("=" * 60)
        log(f"Processing {year}-{month:02d}  [INTERPOLATED month]")
        log("=" * 60)

        prev_data = get_anchor_values(*prev_month, include_quality=True) if prev_month else None
        next_data = get_anchor_values(*next_month, include_quality=True) if next_month else None

        # Interpolate NDVI
        if prev_data is not None and next_data is not None:
            log(f"Interpolating between {prev_month[0]}-{prev_month[1]:02d} "
                f"and {next_month[0]}-{next_month[1]:02d}...")
            mean_ndvi = (prev_data["mean_ndvi"] + next_data["mean_ndvi"]) / 2.0
            # Interpolate quality metrics
            if INCLUDE_QUALITY_METRICS:
                pixel_count = (prev_data["pixel_count"] + next_data["pixel_count"]) / 2.0
                valid_pixel_count = (prev_data["valid_pixel_count"] + next_data["valid_pixel_count"]) / 2.0
                cloud_pixel_percent = (prev_data["cloud_pixel_percent"] + next_data["cloud_pixel_percent"]) / 2.0
        elif prev_data is not None:
            # Edge case: no next anchor
            prev_prev_month = all_months[idx - 3] if idx - 3 >= 0 else None
            prev_prev_data = get_anchor_values(*prev_prev_month, include_quality=True) if prev_prev_month else None
            if prev_prev_data is not None:
                log("No next anchor available — extrapolating linear trend from last two anchors...")
                mean_ndvi = prev_data["mean_ndvi"] + (prev_data["mean_ndvi"] - prev_prev_data["mean_ndvi"])
                if INCLUDE_QUALITY_METRICS:
                    pixel_count = prev_data["pixel_count"] + (prev_data["pixel_count"] - prev_prev_data["pixel_count"])
                    valid_pixel_count = prev_data["valid_pixel_count"] + (prev_data["valid_pixel_count"] - prev_prev_data["valid_pixel_count"])
                    cloud_pixel_percent = prev_data["cloud_pixel_percent"] + (prev_data["cloud_pixel_percent"] - prev_prev_data["cloud_pixel_percent"])
            else:
                log("No next anchor available — carrying forward previous anchor value...")
                mean_ndvi = prev_data["mean_ndvi"]
                if INCLUDE_QUALITY_METRICS:
                    pixel_count = prev_data["pixel_count"]
                    valid_pixel_count = prev_data["valid_pixel_count"]
                    cloud_pixel_percent = prev_data["cloud_pixel_percent"]
        elif next_data is not None:
            log("No previous anchor available — carrying backward next anchor value...")
            mean_ndvi = next_data["mean_ndvi"]
            if INCLUDE_QUALITY_METRICS:
                pixel_count = next_data["pixel_count"]
                valid_pixel_count = next_data["valid_pixel_count"]
                cloud_pixel_percent = next_data["cloud_pixel_percent"]
        else:
            log("⚠ Neither neighbour available — writing NaN placeholder rows.")
            mean_ndvi = np.full(n_colonies, np.nan)
            if INCLUDE_QUALITY_METRICS:
                pixel_count = np.full(n_colonies, 0)
                valid_pixel_count = np.full(n_colonies, 0)
                cloud_pixel_percent = np.full(n_colonies, 100)

        # Build DataFrame
        df_month = attrs.copy()
        df_month["year"] = year
        df_month["month"] = month
        df_month["mean_ndvi"] = mean_ndvi
        if INCLUDE_METHOD_COLUMN:
            df_month["method"] = "interpolated"
        if INCLUDE_QUALITY_METRICS:
            df_month["pixel_count"] = pixel_count
            df_month["valid_pixel_count"] = valid_pixel_count
            df_month["cloud_pixel_percent"] = cloud_pixel_percent
        df_month = df_month[CSV_COLUMNS]

        log("Appending to CSV...")
        append_month_to_csv(df_month)
        row_count += n_colonies
        log(f"✓ CSV now contains {row_count} rows")

        del df_month
        gc.collect()

        elapsed = time.time() - t0
        log(f"Elapsed time: {elapsed:.2f} seconds (no network I/O)")
        log("-" * 60)
        log(f"Overall progress: Month {overall_index} / {total_months}")
        log(f"Overall completion: {100*overall_index/total_months:.1f}%")
        log(f"Current CSV size: {row_count} rows")
        log("")

    total_elapsed = time.time() - overall_start
    log("=" * 60)
    log(f"Pipeline run complete. Total elapsed: {total_elapsed/60:.1f} minutes")
    log(f"Final CSV: {OUTPUT_CSV}  ({row_count} rows)")
    log(f"  Computed months:     {len(anchor_months)}")
    log(f"  Interpolated months: {len(estimated_months)}")
    log("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n⚠ Interrupted. Progress up to the last completed month is saved. "
            "Re-run to resume (if using tmux, your job continues even if you "
            "just close the browser tab without pressing Ctrl+C).")
        sys.exit(1)