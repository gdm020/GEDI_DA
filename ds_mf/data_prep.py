import geopandas as gpd
import pandas as pd
import logging
import sys
import shutil
from pathlib import Path

# --- Configuration ---
INPUT_DIR = Path("/home/gmiddle/rs_paper/data/")
OUTPUT_DIR = Path("/home/gmiddle/rs_paper/data_prep/")

# Ensure Output Directory Exists
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def get_universal_columns(files):
    """Scans all headers to find the intersection of columns."""
    if not files:
        return []
    
    # Read first file's columns to start
    first_gdf = gpd.read_file(files[0], rows=1)
    common_cols = set(first_gdf.columns)
    
    logger.info(f"Scanning {len(files)} files for common schema...")
    
    for fp in files[1:]:
        gdf = gpd.read_file(fp, rows=1)
        current_cols = set(gdf.columns)
        
        # Intersect
        common_cols = common_cols.intersection(current_cols)
    
    # Sort for consistency
    return sorted(list(common_cols))

def process_files():
    # 1. Find all input files
    gpkg_files = list(INPUT_DIR.glob("*_data.gpkg"))
    if not gpkg_files:
        logger.error("No input files found in data folder.")
        return

    # 2. Identify Universal Columns
    universal_cols = get_universal_columns(gpkg_files)
    
    # Ensure 'geometry' is kept if it wasn't strictly in the set (it usually is)
    if "geometry" not in universal_cols:
        universal_cols.append("geometry")
        
    logger.info(f" Found {len(universal_cols)} Universal Columns.")
    logger.info(f"   Columns: {universal_cols}")

    # 3. Process Each City
    for fp in gpkg_files:
        logger.info(f"\nProcessing {fp.name}...")
        
        try:
            # Load Data
            gdf = gpd.read_file(fp)
            initial_count = len(gdf)
            
            # Select ONLY universal columns
            gdf = gdf[universal_cols]
            
            # Remove rows with ANY null values ("data full rows")
            gdf_clean = gdf.dropna()
            final_count = len(gdf_clean)
            
            dropped_count = initial_count - final_count
            
            if final_count == 0:
                logger.warning(f"  WARNING: {fp.name} resulted in 0 rows! (All rows had at least one null).")
                continue

            # Save to New Folder
            output_path = OUTPUT_DIR / fp.name
            
            # Saving as GPKG (Better for ML than GeoJSON)
            gdf_clean.to_file(output_path, driver="GPKG")
            
            logger.info(f"   Original: {initial_count:,} | Cleaned: {final_count:,} | Dropped: {dropped_count:,}")
            logger.info(f"   Saved to {output_path}")
            
        except Exception as e:
            logger.error(f" Failed to process {fp.name}: {e}")

    logger.info("\n--- Data Prep Complete ---")

if __name__ == "__main__":
    process_files()
