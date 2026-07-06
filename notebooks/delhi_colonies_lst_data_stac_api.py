#!/usr/bin/env python3
"""
================================================================================
 UrbanCool — Delhi Colony Monthly LST Pipeline (Landsat Collection 2)
================================================================================
Extracts monthly Daytime Land Surface Temperature for Delhi colonies
from 2018-2025 using Landsat Collection 2 Level-2 Surface Temperature.

UPDATED: Optimizations applied for cloudy monsoon climates (Delhi specific)
================================================================================
"""

import os
import gc
import sys
import time
import calendar
import random
import warnings
from typing import Optional, Tuple, List
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
OUTPUT_CSV       = "/content/drive/MyDrive/urbancool/delhi/Delhi_Colony_Monthly_LST_2018_2025.csv"

STAC_API_URL     = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Landsat Collection 2 Level-2 collections - separate for Landsat 8 and 9
LANDSAT_COLLECTIONS = [
    "landsat-8-c2-l2",  # Landsat 8 Collection 2 Level-2
    "landsat-9-c2-l2",  # Landsat 9 Collection 2 Level-2
]

# Allowed platforms
ALLOWED_PLATFORMS = ["landsat-8", "landsat-9"]

START_YEAR, START_MONTH = 2018, 1
END_YEAR, END_MONTH     = 2025, 12

# Target CRS and resolution
TARGET_CRS        = "EPSG:32643"   # UTM Zone 43N for Delhi
TARGET_RESOLUTION = 30              # 30m native Landsat resolution

# Landsat Bands for Collection 2 Level-2
ST_BAND          = "ST_B10"        # Surface Temperature band
QA_BAND          = "QA_PIXEL"      # Pixel Quality Assessment band

# Landsat Collection 2 Temperature Scale Factor
TEMPERATURE_MULTIPLIER = 0.00341802
TEMPERATURE_ADD_OFFSET = 149.0
TEMPERATURE_KELVIN_OFFSET = 273.15

# QC Bitmask values for Landsat QA_PIXEL
QA_BITS = {
    'fill': 0,
    'dilated_cloud': 1,
    'cirrus': 2,
    'cloud': 3,
    'cloud_shadow': 4,
    'snow': 5,
    'clear': 6,
    'water': 7,
    'cloud_confidence': 8,       # 2-bit field: bits 8-9
    'cloud_shadow_confidence': 10,  # 2-bit field: bits 10-11
    'snow_ice_confidence': 12,      # 2-bit field: bits 12-13
    'cirrus_confidence': 14,        # 2-bit field: bits 14-15
}

CLOUD_CONFIDENCE_NONE = 0
CLOUD_CONFIDENCE_LOW = 1
CLOUD_CONFIDENCE_MEDIUM = 2
CLOUD_CONFIDENCE_HIGH = 3

# --- FIXED: More permissive QA mask settings for heavy cloud areas ---
MASK_BITS = [
    'cloud',
    'dilated_cloud',
    'cirrus',
    'snow',
    # 'cloud_shadow',  # keep commented out if you want to preserve monsoon coverage
]

# Allow up to medium cloud confidence to extract max available data points
CLOUD_CONFIDENCE_MAX = CLOUD_CONFIDENCE_MEDIUM

# --- FIXED: Lower threshold for cloudy Delhi environments ---
# Reduced from 30 to 10 to allow scenes with sparse gaps between clouds
MIN_VALID_PIXEL_PERCENT = 5

# QC mask threshold for zonal statistics
MIN_GOOD_PIXELS = 3

# Zonal statistics setting
ALL_TOUCHED_FOR_ZONAL = False

# Retry settings with jitter for rate limiting
MAX_RETRIES = 10
RETRY_BACKOFF_BASE = 5
RETRY_BACKOFF_MAX = 120

# Chunk size for memory efficiency
CHUNK_SIZE = 256

ATTRIBUTE_COLUMNS = ["zone", "ward", "colony"]

# Output columns
CSV_COLUMNS = ATTRIBUTE_COLUMNS + [
    "year", "month",
    "mean_lst_day_celsius",
    "pixel_count",
    "valid_pixel_count",
    "invalid_pixel_percent"
]

# ==============================================================================
# UTILITIES
# ==============================================================================

@dataclass
class MonthResult:
    """Container for monthly LST processing results."""
    mean_lst_day: np.ndarray
    pixel_count: np.ndarray
    valid_pixel_count: np.ndarray
    invalid_percent: np.ndarray


def log(msg):
    print(msg, flush=True)


def retry_operation(func, *args, max_retries=MAX_RETRIES, desc="operation", **kwargs):
    """Retry-with-exponential-backoff and jitter for network operations."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            error_msg = str(e).lower()

            is_rate_limit = (
                "rate limit" in error_msg or
                "429" in error_msg or
                "too many requests" in error_msg
            )

            if is_rate_limit:
                wait = min(RETRY_BACKOFF_BASE * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX)
                jitter = random.uniform(0, wait * 0.1)
                total_wait = wait + jitter

                if attempt < max_retries:
                    log(f"  ⚠ Rate limit hit (attempt {attempt}/{max_retries})")
                    log(f"    Waiting {total_wait:.1f}s before retry...")
                    time.sleep(total_wait)
                    continue
                else:
                    log(f"  ✗ Rate limit persists after {max_retries} attempts")
                    raise
            else:
                if attempt == max_retries:
                    log(f"  ✗ {desc} failed permanently after {max_retries} attempts: {e}")
                    raise
                wait = min(RETRY_BACKOFF_BASE * attempt, RETRY_BACKOFF_MAX)
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
    if os.path.exists(OUTPUT_CSV):
        try:
            df = pd.read_csv(OUTPUT_CSV)
            valid_rows = df.dropna(subset=['mean_lst_day_celsius'])
            completed_months = set()
            if not valid_rows.empty:
                for _, row in valid_rows.iterrows():
                    completed_months.add((int(row['year']), int(row['month'])))
            log(f"✓ Loaded checkpoint: {len(valid_rows)} valid rows, {len(completed_months)} months with data")
            return df, completed_months
        except Exception as e:
            log(f"⚠ Could not read existing CSV ({e}); starting fresh.")
    return pd.DataFrame(columns=CSV_COLUMNS), set()


def append_month_to_csv(df_month):
    has_valid = df_month['mean_lst_day_celsius'].notna().any()
    if not has_valid:
        log("  No valid data for this month - skipping CSV append")
        return
    write_header = not os.path.exists(OUTPUT_CSV)
    df_month.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
    valid_count = df_month['mean_lst_day_celsius'].notna().sum()
    log(f"  ✓ Appended {valid_count} valid rows to CSV")


# ==============================================================================
# QC / CLOUD MASKING FOR LANDSAT LST
# ==============================================================================

def create_qa_mask(qa_array: np.ndarray) -> np.ndarray:
    """Create a boolean mask from Landsat QA_PIXEL band."""
    mask = np.ones_like(qa_array, dtype=bool)

    for bit_name in MASK_BITS:
        bit_position = QA_BITS[bit_name]
        bad_pixels = (qa_array >> bit_position) & 1
        mask = mask & (bad_pixels == 0)

    cloud_confidence = (qa_array >> QA_BITS['cloud_confidence']) & 0b11
    mask = mask & (cloud_confidence <= CLOUD_CONFIDENCE_MAX)

    return mask


def apply_qa_mask_to_lst(
    lst_array: np.ndarray,
    qa_array: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Apply QA mask to LST array."""
    quality_mask = create_qa_mask(qa_array)

    total_pixels = qa_array.size
    invalid_pixels = np.sum(~quality_mask)
    invalid_percent = (invalid_pixels / total_pixels) * 100 if total_pixels > 0 else 100

    lst_masked = lst_array.copy().astype(np.float32)
    lst_masked[~quality_mask] = np.nan

    return lst_masked, invalid_percent


def scale_lst_to_celsius(lst_dn: np.ndarray) -> np.ndarray:
    """Convert Landsat Collection 2 ST_B10 from DN to degrees Celsius."""
    lst_kelvin = lst_dn.astype(np.float32) * TEMPERATURE_MULTIPLIER + TEMPERATURE_ADD_OFFSET
    lst_celsius = lst_kelvin - TEMPERATURE_KELVIN_OFFSET
    return lst_celsius


# ==============================================================================
# STAC CLIENT AND DATA LOADING
# ==============================================================================

def search_month_items(catalog, bbox_wgs84, year, month):
    """Search all Landsat collections for a given month."""
    all_items = []

    for collection in LANDSAT_COLLECTIONS:
        def _search(coll=collection):
            search = catalog.search(
                collections=[coll],
                bbox=bbox_wgs84,
                datetime=month_datetime_range(year, month),
                query={
                    "eo:cloud_cover": {"lt": 80},
                    "platform": {"in": ALLOWED_PLATFORMS}
                }
            )
            items = list(search.items())
            log(f"  Found {len(items)} images from {coll}")
            return items

        try:
            items = retry_operation(
                _search,
                desc=f"Landsat search {year}-{month:02d} ({collection})"
            )
            all_items.extend(items)
        except Exception as e:
            log(f"  Collection {collection} search failed: {e}")

    return all_items


def load_and_composite_lst(items, bbox_wgs84, year, month):
    """Load ST_B10 and QA_PIXEL bands with QC masking."""
    if not items:
        return None, None, None, None, None

    def _load():
        signed_items = [pc.sign(it) for it in items]

        # Load LST with bilinear interpolation
        ds = odc.stac.load(
            signed_items,
            bands=[ST_BAND, QA_BAND],
            bbox=bbox_wgs84,
            crs=TARGET_CRS,
            resolution=TARGET_RESOLUTION,
            resampling="bilinear",
            groupby="solar_day",
            chunks={"time": 1, "x": CHUNK_SIZE, "y": CHUNK_SIZE},
        )

        # QA loaded separately with nearest neighbor
        ds_qa = odc.stac.load(
            signed_items,
            bands=[QA_BAND],
            bbox=bbox_wgs84,
            crs=TARGET_CRS,
            resolution=TARGET_RESOLUTION,
            resampling="nearest",
            groupby="solar_day",
            chunks={"time": 1, "x": CHUNK_SIZE, "y": CHUNK_SIZE},
        )

        return ds, ds_qa

    ds, ds_qa = retry_operation(_load, desc=f"odc-stac load {year}-{month:02d}")

    if ds is None or ST_BAND not in ds or ds.sizes.get("time", 0) == 0:
        return None, None, None, None, None

    n_time = ds.sizes.get("time", 0)
    log(f"  Loaded {n_time} Landsat scenes")

    lst_list = []
    scene_invalid_percents = []

    for t_idx in range(n_time):
        lst = ds[ST_BAND].isel(time=t_idx).values.astype(np.float32)
        qa = ds_qa[QA_BAND].isel(time=t_idx).values.astype(np.int16)

        lst_masked, scene_invalid_pct = apply_qa_mask_to_lst(lst, qa)

        # --- FIXED: Added real-time clear pixel diagnostics ---
        clear_pixel_pct = 100 - scene_invalid_pct
        log(f"    Scene {t_idx+1}: {clear_pixel_pct:.1f}% clear pixels")

        if scene_invalid_pct <= (100 - MIN_VALID_PIXEL_PERCENT):
            lst_list.append(lst_masked)
            scene_invalid_percents.append(scene_invalid_pct)
        else:
            log(f"      Rejected (threshold: {MIN_VALID_PIXEL_PERCENT}% clear required)")

        del lst, qa, lst_masked

    if not lst_list:
        log(f"  No scenes with {MIN_VALID_PIXEL_PERCENT}% clear pixels")
        return None, None, None, None, None

    log(f"  Kept {len(lst_list)} scenes after QA filtering")
    avg_invalid = np.mean(scene_invalid_percents) if scene_invalid_percents else 0
    log(f"  Average invalid pixels per scene: {avg_invalid:.1f}%")

    lst_stacked = np.stack(lst_list, axis=0)
    lst_med = np.nanmedian(lst_stacked, axis=0).astype(np.float32)
    valid_count = np.sum(~np.isnan(lst_stacked), axis=0)

    geobox = ds.odc.geobox
    affine = geobox.affine

    del ds, ds_qa, lst_stacked
    gc.collect()

    return lst_med, valid_count, scene_invalid_percents, affine, len(lst_list)


def compute_zonal_stats_lst(
    gdf_target_crs,
    lst_array: np.ndarray,
    affine
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute zonal statistics for LST with quality metrics."""
    stats = zonal_stats(
        gdf_target_crs,
        lst_array,
        affine=affine,
        stats=["mean", "count"],
        nodata=np.nan,
        all_touched=ALL_TOUCHED_FOR_ZONAL,
        geojson_out=False,
    )

    mean_lst = np.array([s["mean"] for s in stats], dtype=np.float32)
    valid_pixels = np.array([s["count"] for s in stats], dtype=np.int32)

    ones_array = np.ones_like(lst_array, dtype=np.int32)
    stats_total = zonal_stats(
        gdf_target_crs,
        ones_array,
        affine=affine,
        stats=["count"],
        nodata=0,
        all_touched=ALL_TOUCHED_FOR_ZONAL,
        geojson_out=False,
    )
    total_pixels = np.array([s["count"] for s in stats_total], dtype=np.int32)

    invalid_percent = np.zeros_like(total_pixels, dtype=np.float32)
    for i in range(len(total_pixels)):
        if total_pixels[i] > 0:
            invalid_percent[i] = 100 * (1 - valid_pixels[i] / total_pixels[i])
        else:
            invalid_percent[i] = 100

    mean_lst[valid_pixels < MIN_GOOD_PIXELS] = np.nan

    return mean_lst, total_pixels, valid_pixels, invalid_percent


def compute_one_month(
    catalog,
    bbox_wgs84,
    year,
    month,
    n_colonies
) -> Optional[MonthResult]:
    """Full pipeline for one month of Landsat LST data."""

    log("Searching STAC for Landsat 8/9 imagery...")
    items = search_month_items(catalog, bbox_wgs84, year, month)

    if len(items) == 0:
        log("  ⚠ No Landsat 8/9 images found for this month")
        return None

    log(f"✓ Found {len(items)} Landsat 8/9 images")
    log("Loading imagery (bbox-restricted, ST_B10 + QA_PIXEL)...")

    lst_med, valid_count, scene_invalid_pcts, affine, n_scenes = load_and_composite_lst(
        items, bbox_wgs84, year, month
    )

    if lst_med is None:
        log("  ⚠ Composite could not be built")
        return None

    log(f"✓ Built composite from {n_scenes} scenes")
    log("Converting LST to Celsius...")
    lst_celsius = scale_lst_to_celsius(lst_med)

    log("Running zonal statistics...")
    mean_lst, total_pixels, valid_pixels, invalid_percent = compute_zonal_stats_lst(
        GDF_GEOM_ONLY, lst_celsius, affine
    )

    valid_count_total = np.sum(~np.isnan(mean_lst))
    log(f"✓ {valid_count_total} colonies have valid LST data")

    if valid_count_total == 0:
        log("  ⚠ No valid LST data extracted - all colonies masked out")
        return None

    del lst_med, lst_celsius
    gc.collect()

    return MonthResult(
        mean_lst_day=mean_lst,
        pixel_count=total_pixels,
        valid_pixel_count=valid_pixels,
        invalid_percent=invalid_percent
    )


# ==============================================================================
# MAIN
# ==============================================================================

GDF_GEOM_ONLY = None


def main():
    global GDF_GEOM_ONLY
    overall_start = time.time()

    log("=" * 60)
    log("UrbanCool — Delhi Colony Monthly LST (Landsat Collection 2)")
    log("30m resolution · Correct temperature scaling · QC masking")
    log(f"Collections: {', '.join(LANDSAT_COLLECTIONS)}")
    log("=" * 60)

    # ---- Load shapefile -------------------------------------------
    log(f"Loading colony shapefile: {SHAPEFILE_PATH}")

    if not os.path.exists(SHAPEFILE_PATH):
        log(f"✗ ERROR: Shapefile not found at {SHAPEFILE_PATH}")
        sys.exit(1)

    try:
        gdf = gpd.read_file(SHAPEFILE_PATH)
        n_colonies = len(gdf)
        log(f"✓ Loaded {n_colonies} colony polygons")
    except Exception as e:
        log(f"✗ Failed to load shapefile: {e}")
        sys.exit(1)

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

    # ---- Connect to STAC -------------------------------------------
    log("Connecting to Planetary Computer STAC API...")
    try:
        catalog = retry_operation(
            pystac_client.Client.open, STAC_API_URL,
            modifier=pc.sign_inplace, desc="STAC catalog connection",
        )
        log("✓ Connected.")
    except Exception as e:
        log(f"✗ Failed to connect to STAC: {e}")
        sys.exit(1)

    # ---- Build month list -------------------------------------------
    all_months = month_range(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    total_months = len(all_months)
    log(f"Total months: {total_months}")

    # ---- Resume detection -------------------------------------------
    existing_df, completed_months = load_existing_csv()

    months_to_run = [m for m in all_months if m not in completed_months]
    log(f"Months to process: {len(months_to_run)}")

    if len(months_to_run) == 0:
        log("\n✓ All months already processed!")
        sys.exit(0)

    row_count = len(existing_df)
    month_durations = []

    # ---- Process each month -----------------------------------------
    log("\n" + "=" * 60)
    log("STARTING LANDSAT LST EXTRACTION")
    log("=" * 60)

    for (year, month) in tqdm(months_to_run, desc="Processing months", unit="month"):
        t0 = time.time()
        overall_index = all_months.index((year, month)) + 1
        log("=" * 60)
        log(f"Processing {year}-{month:02d}")
        log("=" * 60)

        try:
            result = compute_one_month(catalog, bbox_wgs84, year, month, n_colonies)

            if result is None:
                log("  ⚠ No usable data for this month — skipping")
            else:
                df_month = attrs.copy()
                df_month["year"] = year
                df_month["month"] = month
                df_month["mean_lst_day_celsius"] = result.mean_lst_day
                df_month["pixel_count"] = result.pixel_count
                df_month["valid_pixel_count"] = result.valid_pixel_count
                df_month["invalid_pixel_percent"] = result.invalid_percent

                df_month = df_month[CSV_COLUMNS]

                if df_month['mean_lst_day_celsius'].notna().any():
                    append_month_to_csv(df_month)
                    completed_months.add((year, month))
                    row_count += n_colonies
                    log(f"✓ CSV now contains {row_count} rows")
                else:
                    log("  ⚠ No valid data to append")

                del df_month
                if result is not None:
                    del result

            gc.collect()

        except Exception as e:
            log(f"✗ FAILED to process {year}-{month:02d}: {e}")
            log("  Skipping — will retry on next run.")

        elapsed = time.time() - t0
        month_durations.append(elapsed)
        if len(month_durations) > 0:
            avg = sum(month_durations) / len(month_durations)
            remaining = (len(months_to_run) - months_to_run.index((year, month)) - 1) * avg
            log(f"Average time per month: {avg:.1f}s")
            log(f"Estimated remaining: {remaining/60:.1f} minutes")

        log("-" * 60)
        log(f"Overall progress: Month {overall_index} / {total_months}")
        log(f"Overall completion: {100*overall_index/total_months:.1f}%")
        log(f"Current CSV size: {row_count} rows")
        log(f"Memory usage: {get_memory_usage_mb():.1f} MB")
        log("")

    # ---- Final summary -------------------------------------------
    total_elapsed = time.time() - overall_start
    log("=" * 60)
    log(f"Pipeline complete. Total elapsed: {total_elapsed/60:.1f} minutes")
    log(f"Final CSV: {OUTPUT_CSV}")

    if os.path.exists(OUTPUT_CSV):
        final_df = pd.read_csv(OUTPUT_CSV)
        valid_rows = final_df.dropna(subset=['mean_lst_day_celsius'])

        if not valid_rows.empty:
            valid_months = (
                valid_rows[["year", "month"]]
                .drop_duplicates()
                .shape[0]
            )
            log(f"✓ {len(valid_rows)} valid rows")
            log(f"✓ {valid_months} months with valid LST data")
            log(f"✓ {valid_rows['colony'].nunique()} colonies with data")
            log(f"  Mean LST: {valid_rows['mean_lst_day_celsius'].mean():.1f}°C")
            log(f"  Min LST: {valid_rows['mean_lst_day_celsius'].min():.1f}°C")
            log(f"  Max LST: {valid_rows['mean_lst_day_celsius'].max():.1f}°C")

            log("\n  Sample data (first 5 rows):")
            sample_cols = ['colony', 'year', 'month', 'mean_lst_day_celsius', 'invalid_pixel_percent']
            print(valid_rows[sample_cols].head(5).to_string())
        else:
            log("  ⚠ No valid LST data found in the CSV.")
    else:
        log("✗ Output CSV was not created")

    log("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n  ⚠ Interrupted. Progress up to the last completed month is saved.")
        sys.exit(1)