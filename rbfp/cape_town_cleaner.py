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
INPUT_FILENAME = "cape_town.geojson"
OUTPUT_FILENAME = "cape_town_footprints.gpkg"
OUTPUT_LAYER = "cape_town_footprints"

# CRS: WGS 84 / UTM zone 34S (Meter-based for Cape Town)
TARGET_CRS = "EPSG:32734"

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
    """
    Scrapes valid JSON 'Feature' objects from a raw string, ignoring
    file structure (commas, brackets, headers) entirely.
    """
    logger.info("Starting 'Scorched Earth' feature scraping...")
    
    # regex to find the start of a feature
    # matches {"type": "Feature" or {"type":"Feature" allowing for whitespace/quotes
    feature_start_pattern = re.compile(r'\{\s*\"type\"\s*:\s*\"Feature\"')
    
    decoder = json.JSONDecoder()
    valid_features = []
    pos = 0
    total_len = len(text_content)
    
    while True:
        # Find next potential start of a JSON object
        match = feature_start_pattern.search(text_content, pos)
        if not match:
            break
            
        start_idx = match.start()
        
        try:
            # raw_decode parses one valid JSON object and returns (obj, end_index)
            # It stops exactly where the object ends, ignoring missing commas after it.
            obj, end_idx = decoder.raw_decode(text_content, idx=start_idx)
            
            # double check it actually has geometry (sanity check)
            if "geometry" in obj and "properties" in obj:
                valid_features.append(obj)
            
            # Advance pointer
            pos = end_idx
            
        except json.JSONDecodeError:
            # If this block looked like a feature but wasn't valid JSON, skip past it
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

def process_cape_town():
    logger.info(f"Input: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        logger.error(f"File not found: {INPUT_PATH}")
        sys.exit(1)

    gdf = None

    # --- 1. Load Data (The Hierarchy of Desperation) ---
    
    # Attempt A: Standard Ingest
    try:
        logger.info("Attempt A: Standard Fiona ingest...")
        gdf = gpd.read_file(INPUT_PATH, engine="fiona")
        logger.info("Success: Standard ingest worked.")
    except Exception:
        logger.warning("Attempt A failed. Trying raw text recovery...")
        
        # Load raw content - Force 'latin1' to ensure we read EVERY byte without decoding errors
        raw_text = ""
        try:
            with open(INPUT_PATH, "r", encoding="latin1") as f:
                raw_text = f.read()
            logger.info(f"Loaded raw file into memory ({len(raw_text)} bytes).")
        except Exception as e_read:
            logger.error(f"Fatal: Could not read file even as latin1: {e_read}")
            sys.exit(1)

        # Attempt B: Whole JSON Parse
        try:
            logger.info("Attempt B: Parsing whole file as JSON...")
            data = json.loads(raw_text)
            gdf = gpd.GeoDataFrame.from_features(data["features"])
            logger.info("Success: Whole JSON parse worked.")
        except Exception:
            logger.warning("Attempt B failed (JSON syntax corrupted).")
            
            # Attempt C: The Scraper (Nuclear Option)
            logger.info("Attempt C: Scraper Mode (extracting individual features from garbage)...")
            features = scrape_features_from_text(raw_text)
            
            if not features:
                logger.error("Fatal: Scraper found 0 valid features. The file might not contain GeoJSON.")
                sys.exit(1)
                
            logger.info("Building GeoDataFrame from scraped features...")
            gdf = gpd.GeoDataFrame.from_features(features)
            
            # GeoJSON is always WGS84
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)

    # --- 2. Schema Normalization ---
    if gdf.geometry.name != "geometry":
        gdf = gdf.rename(columns={gdf.geometry.name: "geometry"}).set_geometry("geometry")
        
    logger.info(f"Reprojecting to {TARGET_CRS} (Meter-based)...")
    gdf = gdf.to_crs(TARGET_CRS)

    # Height Logic: Rename BLD_HGT -> height
    logger.info("Standardizing height column (BLD_HGT -> height)...")
    
    if "BLD_HGT" in gdf.columns:
        # Coerce to numeric, errors become NaN
        gdf["height"] = pd.to_numeric(gdf["BLD_HGT"], errors="coerce").astype("float64")
        
        # Log stats
        null_count = gdf["height"].isna().sum()
        logger.info(f"Height stats: {len(gdf) - null_count} valid, {null_count} NULL")
    else:
        logger.warning("Column 'BLD_HGT' not found! 'height' will be all NULL.")
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
    process_cape_town()
