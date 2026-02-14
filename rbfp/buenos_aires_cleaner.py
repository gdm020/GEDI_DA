import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import sys
import shutil
import zipfile
from pathlib import Path

# --- Configuration ---
ZIP_PATH = Path("/home/gmiddle/rs_paper/bfp/tejido.zip")
TEMP_DIR = Path("/home/gmiddle/rs_paper/bfp/temp_ba_extract")
OUTPUT_FILE = Path("/home/gmiddle/rs_paper/clean_fp/buenos_aires.gpkg")
TARGET_CRS = "EPSG:32721"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def clean_height_value(val):
    if pd.isna(val): return np.nan
    try:
        s_val = str(val).replace(",", ".").strip()
        f_val = float(s_val)
        if 0 < f_val < 600: return f_val
        return np.nan
    except:
        return np.nan

def run():
    logger.info("--- BUENOS AIRES EXTRACTOR ---")
    
    # 1. Unzip
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Unzipping {ZIP_PATH} to temp folder...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(TEMP_DIR)
    except Exception as e:
        logger.error(f"Failed to unzip: {e}")
        return

    # 2. Find the .shp file (Recursive search)
    shp_files = list(TEMP_DIR.rglob("*.shp"))
    if not shp_files:
        logger.error("❌ No .shp file found inside the zip!")
        return
    
    # Take the first one found (usually the correct one)
    target_shp = shp_files[0]
    logger.info(f"Found shapefile: {target_shp.name}")

    try:
        # 3. Read (Latin1)
        logger.info("Reading shapefile...")
        gdf = gpd.read_file(target_shp, encoding="latin1")
        
        initial_count = len(gdf)
        logger.info(f"Loaded {initial_count:,} features.")

        # 4. Clean Height
        # Check 'altura' or uppercase 'ALTURA'
        col_map = {c.lower(): c for c in gdf.columns}
        if "altura" not in col_map:
            logger.error(f"❌ Column 'altura' not found. Available: {list(gdf.columns)}")
            return

        h_col = col_map["altura"]
        logger.info(f"Cleaning column '{h_col}'...")
        gdf["height"] = gdf[h_col].apply(clean_height_value)

        # 5. Stats
        null_count = gdf["height"].isna().sum()
        print(f"\n" + "="*40)
        print(f"BUENOS AIRES STATS")
        print(f"Valid Heights: {initial_count - null_count:,}")
        print(f"Null Heights:  {null_count:,} ({null_count/initial_count:.1%})")
        print(f"="*40 + "\n")

        # 6. Save
        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf[["geometry", "height"]]
        
        if gdf.crs.to_string() != TARGET_CRS:
            logger.info(f"Reprojecting to {TARGET_CRS}...")
            gdf.to_crs(TARGET_CRS, inplace=True)

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to {OUTPUT_FILE}...")
        gdf.to_file(OUTPUT_FILE, driver="GPKG", layer="buenos_aires")
        logger.info("✅ Success.")

    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
    finally:
        # Cleanup
        if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    run()