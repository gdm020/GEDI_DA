import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import sys
import json
import re
import warnings
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path("/home/gmiddle/rs_paper/bfp/")
INPUT_FILE = BASE_DIR / "austin.geojson"
OUTPUT_FILE = Path("/home/gmiddle/rs_paper/clean_fp/austin.gpkg")
TARGET_CRS = "EPSG:32614"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

def scrape_features_from_text(text_content):
    """Fallback parser for malformed/large GeoJSON."""
    logger.info("Engaging raw text scraper...")
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
            print(f"Scraped {len(valid_features)} features...", end="\r")

    print("") 
    logger.info(f"Scraping complete. Recovered {len(valid_features)} features.")
    return valid_features

def run():
    logger.info("--- PROCESSING AUSTIN (WITH FALLBACK) ---")
    
    if not INPUT_FILE.exists():
        logger.error(f"Missing input file: {INPUT_FILE}")
        return

    gdf = None

    # 1. Robust Loading
    try:
        logger.info("Attempt 1: Standard GeoPandas load...")
        gdf = gpd.read_file(INPUT_FILE)
    except Exception:
        logger.warning("Attempt 1 failed. Reading raw text...")
        try:
            with open(INPUT_FILE, "r", encoding="utf-8") as f: raw_text = f.read()
        except UnicodeDecodeError:
            with open(INPUT_FILE, "r", encoding="latin1") as f: raw_text = f.read()
        
        try:
            logger.info("Attempt 2: JSON Parse...")
            data = json.loads(raw_text)
            gdf = gpd.GeoDataFrame.from_features(data["features"])
        except Exception:
            logger.warning("Attempt 2 failed. Using Scraper...")
            features = scrape_features_from_text(raw_text)
            if not features:
                logger.error("Fatal: Scraper found no features.")
                return
            gdf = gpd.GeoDataFrame.from_features(features)
            if gdf.crs is None: gdf.set_crs("EPSG:4326", inplace=True)

    initial_count = len(gdf)
    logger.info(f"Loaded {initial_count} features.")

    # 2. Normalize Columns (Fix the 'max_height' vs 'MAX_HEIGHT' issue)
    gdf.columns = [c.lower() for c in gdf.columns]
    
    # 3. Calculate Height (The Fix)
    logger.info("Calculating height (Priority: max_height -> elevation - base_elevation)...")

    # Ensure columns exist and are numeric
    cols_needed = ['max_height', 'elevation', 'base_elevation']
    for c in cols_needed:
        if c not in gdf.columns:
            # Create if missing (fill with NaN)
            logger.warning(f"Column '{c}' missing, filling with NaN.")
            gdf[c] = np.nan
        else:
            # Coerce to numeric
            gdf[c] = pd.to_numeric(gdf[c], errors='coerce')

    # Step A: Start with max_height
    gdf['final_height_ft'] = gdf['max_height']

    # Step B: Calculate fallback (Elevation - Base)
    # This assumes both are in feet, matching Austin metadata standards
    fallback_height = gdf['elevation'] - gdf['base_elevation']

    # Step C: Fill missing max_height with fallback
    # We fill where max_height is NaN
    gdf['final_height_ft'] = gdf['final_height_ft'].fillna(fallback_height)

    # Step D: Convert to Meters (1 ft = 0.3048 m)
    gdf['height'] = gdf['final_height_ft'] * 0.3048

    # 4. Data Integrity Report
    valid_count = gdf['height'].notna().sum()
    null_count = initial_count - valid_count
    
    # Check how many were saved by the fallback
    original_valid = gdf['max_height'].notna().sum()
    rescued_count = valid_count - original_valid

    print("\n" + "="*50)
    print("DATA INTEGRITY REPORT: AUSTIN")
    print("-" * 50)
    print(f"Total Features:          {initial_count}")
    print(f"Valid 'max_height':      {original_valid}")
    print(f"Rescued via Fallback:    {rescued_count}")
    print(f"Total Valid Heights:     {valid_count}")
    print(f"Final Nulls:             {null_count}")
    print(f"Retention Rate:          {(valid_count/initial_count)*100:.2f}%")
    print("="*50 + "\n")

    # 5. Clean & Save
    logger.info("Filtering valid geometries...")
    gdf = gdf[gdf['height'].notna()] # Drop only if we truly failed to find ANY height
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    
    # Keep strictly necessary columns
    gdf = gdf[["geometry", "height"]]

    if gdf.crs.to_string() != TARGET_CRS:
        logger.info(f"Reprojecting to {TARGET_CRS}...")
        gdf.to_crs(TARGET_CRS, inplace=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {OUTPUT_FILE}...")
    
    try:
        gdf.to_file(OUTPUT_FILE, driver="GPKG", layer="austin")
        logger.info("Success.")
    except Exception as e:
        logger.warning(f"Save failed ({e}). Attempting repair...")
        gdf.geometry = gdf.geometry.buffer(0)
        gdf.to_file(OUTPUT_FILE, driver="GPKG", layer="austin")
        logger.info("Success (after repair).")

if __name__ == "__main__":
    run()