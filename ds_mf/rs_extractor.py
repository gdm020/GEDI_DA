import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import sys
import zipfile
import shutil
import rasterio
import os
from pathlib import Path
from shapely.geometry import box

# --- Configuration ---
# Source: Morphologically processed footprints
FOOTPRINT_DIR = Path("/home/gmiddle/rs_paper/morphologically_processed/")
# Source: RS imagery zips
RS_DIR = Path("/home/gmiddle/rs_paper/rs_data/")
# Output: Destination for the combined dataset
OUTPUT_DIR = Path("/home/gmiddle/rs_paper/data/")

# Mapping RS zip name patterns (keys) to Footprint filenames (values)
CITY_MAP = {
    "amsterdam": "amsterdam",
    "auckland": "auckland", 
    "aust": "austin",       
    "berlin": "berlin",
    "buenos_aires": "buenos_aires",
    "capetown": "cape_town",
    "madrid": "madrid",
    "san_francisco": "san_francisco",
    "sendai": "sendai",
    "warsaw": "warsaw"
}

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def find_footprint_file(city_key):
    """Finds the .gpkg for the city in the morphologically_processed folder."""
    candidates = list(FOOTPRINT_DIR.glob(f"{city_key}.gpkg"))
    if not candidates:
        candidates = list(FOOTPRINT_DIR.glob(f"{city_key}_footprints.gpkg"))
    return candidates[0] if candidates else None

def sanitize_name(name):
    """Cleans strings to be SQLite/GeoPackage friendly."""
    if not name: return "unknown"
    return name.replace(" ", "_").replace("-", "_").replace(".", "").lower()

def process_city(rs_name, fp_name):
    logger.info(f"\n==================================================")
    logger.info(f"Processing City: {fp_name.upper()} (RS Key: {rs_name})")
    
    # 1. Setup Paths
    fp_path = find_footprint_file(fp_name)
    if not fp_path:
        logger.error(f"Could not find footprint file for {fp_name} in {FOOTPRINT_DIR}. Skipping.")
        return

    output_path = OUTPUT_DIR / f"{fp_name}_data.gpkg"
    
    # 2. Locate RS Zip Files
    rs_zips = list(RS_DIR.glob(f"*{rs_name}*.zip"))
    if not rs_zips:
        logger.error(f"No RS zip files found matching '*{rs_name}*'. Skipping.")
        return
    
    logger.info(f"Input Footprints: {fp_path.name}")
    logger.info(f"Output Target:    {output_path}")

    # 3. Load Footprints
    try:
        try:
            gdf = gpd.read_file(fp_path, engine="pyogrio")
        except ImportError:
            gdf = gpd.read_file(fp_path)
    except Exception as e:
        logger.error(f"Failed to read footprints: {e}")
        return

    logger.info(f"Loaded {len(gdf)} footprints. Calculating centroids...")
    centroids_local = gdf.centroid
    
    # 4. Extract Zips
    temp_dir = RS_DIR / f"temp_{rs_name}"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    # List to hold all RS dataframes before merging
    # Each item will be a DataFrame of extracted values for ONE image
    rs_data_chunks = []

    try:
        for z in rs_zips:
            logger.info(f"Unzipping {z.name}...")
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

        # Find raster files recursively
        extensions = ["*.tif", "*.tiff", "*.jp2", "*.img"]
        raster_files = []
        for ext in extensions:
            raster_files.extend(list(temp_dir.rglob(ext)))

        if not raster_files:
            logger.warning("No raster files found.")
            return

        logger.info(f"Found {len(raster_files)} images. Starting extraction...")

        for i, r_path in enumerate(raster_files):
            try:
                with rasterio.open(r_path) as src:
                    # --- DYNAMIC REPROJECTION ---
                    raster_crs = src.crs
                    if raster_crs:
                        # Project centroids to match THIS image for accurate sampling
                        centroids_proj = centroids_local.to_crs(raster_crs)
                    else:
                        logger.warning(f"Image {r_path.name} has no CRS. Skipping.")
                        continue

                    # --- OVERLAP CHECK ---
                    l, b, r, t = src.bounds
                    tile_box = box(l, b, r, t)
                    
                    # Spatial index intersection
                    possible_matches_idx = list(centroids_proj.sindex.intersection(tile_box.bounds))
                    if not possible_matches_idx:
                        continue 

                    subset_centroids = centroids_proj.iloc[possible_matches_idx]
                    precise_matches = subset_centroids[subset_centroids.within(tile_box)]
                    
                    if precise_matches.empty:
                        continue

                    # --- SAMPLING ---
                    match_indices = precise_matches.index
                    match_coords = [(pt.x, pt.y) for pt in precise_matches]
                    
                    # Sample all bands for these points
                    # samples is list of arrays: [ [b1_val, b2_val...], ... ]
                    samples = list(src.sample(match_coords))
                    if not samples: continue
                    samples_arr = np.array(samples) 

                    # --- BUILD DATAFRAME FOR THIS IMAGE ---
                    # We create a mini-dataframe for this image's contributions
                    # Columns will be uniquely named: {filename}_{bandname}
                    
                    image_data = {}
                    file_stem = sanitize_name(r_path.stem)
                    descriptions = src.descriptions
                    
                    for band_idx in range(src.count):
                        # Get band description or generic name
                        band_desc = descriptions[band_idx]
                        if not band_desc:
                            band_desc = f"b{band_idx + 1}"
                        
                        clean_band = sanitize_name(band_desc)
                        
                        # Unique Column Name: filename + band
                        col_name = f"{file_stem}_{clean_band}"
                        
                        image_data[col_name] = samples_arr[:, band_idx]

                    # Create DataFrame indexed by the original footprint IDs
                    img_df = pd.DataFrame(image_data, index=match_indices)
                    rs_data_chunks.append(img_df)
                    
                    if (i + 1) % 5 == 0:
                        logger.info(f"Processed {i + 1}/{len(raster_files)} images...")
            
            except Exception as e:
                logger.warning(f"Failed to process {r_path.name}: {e}")

        # 5. MERGE ALL DATA
        if rs_data_chunks:
            logger.info(f"Merging {len(rs_data_chunks)} data chunks...")
            # Concatenate all RS chunks horizontally (axis=1) matching on index
            # This handles the fragmentation issue by doing one big join
            all_rs_data = pd.concat(rs_data_chunks, axis=1)
            
            # Combine with original footprints
            logger.info("Joining with original footprints...")
            final_gdf = pd.concat([gdf, all_rs_data], axis=1)
            
            # 6. SAVE
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving combined dataset to {output_path}...")
            final_gdf.to_file(output_path, driver="GPKG", layer=f"{fp_name}_data")
            logger.info(f"Success. Total columns: {len(final_gdf.columns)}")
            
        else:
            logger.warning("No RS data extracted for this city.")

    finally:
        # Cleanup temp dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def run():
    for rs_key, fp_key in CITY_MAP.items():
        process_city(rs_key, fp_key)

if __name__ == "__main__":
    run()