import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import sys
import warnings
import math
import fiona
import json
import re
from shapely.geometry import shape
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path("/home/gmiddle/rs_paper/bfp/")
INPUT_FILENAME = "san_fransisco.geojson"
OUTPUT_FILENAME = "san_francisco_footprints.gpkg"
OUTPUT_LAYER = "san_francisco_footprints"

# CRS: WGS 84 / UTM Zone 10N (Meter-based for San Francisco)
TARGET_CRS = "EPSG:32610"

INPUT_PATH = BASE_DIR / INPUT_FILENAME
OUTPUT_PATH = BASE_DIR / OUTPUT_FILENAME

# --- Engine & Warning Config ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=fiona.errors.FionaDeprecationWarning)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Silence underlying libraries
logging.getLogger("fiona").setLevel(logging.ERROR)
logging.getLogger("fiona.ogrext").setLevel(logging.ERROR)
logging.getLogger("osgeo").setLevel(logging.ERROR)

def scrape_features_from_text(text_content):
    """Scrapes valid JSON 'Feature' objects from a raw string."""
    logger.info("Starting 'Scorched Earth' feature scraping...")
    feature_start_pattern = re.compile(r'\{\s*\"type\"\s*:\s*\"Feature\"')
    decoder = json.JSONDecoder()
    valid_features = []
    pos = 0
    
    while True:
        match = feature_start_pattern.search(text_content, pos)
        if not match: break
            
        start_idx = match.start()
        try:
            obj, end_idx = decoder.raw_decode(text_content, idx=start_idx)
            if "geometry" in obj and "properties" in obj:
                valid_features.append(obj)
            pos = end_idx
        except json.JSONDecodeError:
            pos = start_idx + 1
            
        if len(valid_features) % 50000 == 0 and len(valid_features) > 0:
            logger.info(f"Scraped {len(valid_features)} features so far...")

    logger.info(f"Scraping complete. Recovered {len(valid_features)} valid features.")
    return valid_features

def fix_invalid_polys_once(gdf_in: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One-pass make_valid repair."""
    gdf_in = gdf_in[gdf_in.geometry.notna()].copy()
    gdf_in = gdf_in[~gdf_in.is_empty].copy()

    v = gdf_in.is_valid
    n_bad = int((~v).sum())
    
    if n_bad == 0: return gdf_in
        
    logger.warning(f"Repairing {n_bad} invalid geometries...")
    gdf_out = gdf_in.copy()
    gdf_out.loc[~v, "geometry"] = gdf_out.loc[~v, "geometry"].make_valid()
    
    # Post-repair filter
    gdf_out = gdf_out[gdf_out.geometry.notna()].copy()
    gdf_out = gdf_out[~gdf_out.is_empty].copy()
    gdf_out = gdf_out[gdf_out.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    
    return gdf_out

def process_san_francisco():
    logger.info(f"Input: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        logger.error(f"File not found: {INPUT_PATH}")
        sys.exit(1)

    gdf = None

    # --- 1. Load Data (Hierarchy of Desperation) ---
    try:
        logger.info("Attempt A: Standard Fiona ingest...")
        gdf = gpd.read_file(INPUT_PATH, engine="fiona")
        logger.info("Success: Standard ingest worked.")
    except Exception:
        logger.warning("Attempt A failed. Trying raw text recovery...")
        
        raw_text = ""
        try:
            # Try utf-8 first, fallback to latin1
            try:
                with open(INPUT_PATH, "r", encoding="utf-8") as f:
                    raw_text = f.read()
            except UnicodeDecodeError:
                with open(INPUT_PATH, "r", encoding="latin1") as f:
                    raw_text = f.read()
            logger.info(f"Loaded raw file into memory ({len(raw_text)} bytes).")
        except Exception as e_read:
            logger.error(f"Fatal: Could not read file: {e_read}")
            sys.exit(1)

        try:
            logger.info("Attempt B: Parsing whole file as JSON...")
            data = json.loads(raw_text)
            gdf = gpd.GeoDataFrame.from_features(data["features"])
            logger.info("Success: Whole JSON parse worked.")
        except Exception:
            logger.warning("Attempt B failed. Using Scraper (Attempt C)...")
            features = scrape_features_from_text(raw_text)
            if not features:
                logger.error("Fatal: Scraper found 0 valid features.")
                sys.exit(1)
            gdf = gpd.GeoDataFrame.from_features(features)
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)

    # --- 2. Schema Normalization ---
    if gdf.geometry.name != "geometry":
        gdf = gdf.rename(columns={gdf.geometry.name: "geometry"}).set_geometry("geometry")
        
    logger.info(f"Reprojecting to {TARGET_CRS} (Meter-based)...")
    gdf = gdf.to_crs(TARGET_CRS)

    # Height Logic: Rename hgt_maxcm -> height AND convert CM to Meters
    logger.info("Standardizing height column (hgt_maxcm [cm] -> height [m])...")
    
    if "hgt_maxcm" in gdf.columns:
        # Coerce to numeric
        raw_height = pd.to_numeric(gdf["hgt_maxcm"], errors="coerce")
        
        # Convert CM to Meters
        gdf["height"] = (raw_height / 100.0).astype("float64")
        
        # Log stats
        null_count = gdf["height"].isna().sum()
        logger.info(f"Height stats: {len(gdf) - null_count} valid, {null_count} NULL")
    else:
        logger.warning("Column 'hgt_maxcm' not found! 'height' will be all NULL.")
        gdf["height"] = np.nan

    # --- 3. Filter ---
    logger.info("Filtering non-polygons...")
    gdf = gdf[gdf.geometry.notna() & ~gdf.is_empty]
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    # --- 4. Write ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists(): OUTPUT_PATH.unlink()

    logger.info(f"Writing {len(gdf)} features to {OUTPUT_PATH}")
    try:
        gdf.to_file(OUTPUT_PATH, driver="GPKG", layer=OUTPUT_LAYER, index=False)
    except Exception as e:
        logger.warning(f"Write failed ({e}). Retrying with repair...")
        gdf = fix_invalid_polys_once(gdf)
        if OUTPUT_PATH.exists(): OUTPUT_PATH.unlink()
        gdf.to_file(OUTPUT_PATH, driver="GPKG", layer=OUTPUT_LAYER, index=False)
    
    logger.info("Success.")

if __name__ == "__main__":
    process_san_francisco()