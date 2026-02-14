import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import sys
import zipfile
import shutil
import os
import rasterio
from pathlib import Path
from shapely.geometry import box

# --- Configuration ---
BASE_DIR = Path("/home/gmiddle/rs_paper/bfp/auckland")
OUTPUT_FILENAME = "auckland_footprints.gpkg"
OUTPUT_LAYER = "auckland_footprints"

# CRS: NZTM2000 (Standard for NZ Lidar)
TARGET_CRS = "EPSG:2193"

# Zip Filenames (from your notes)
ZIP_BUILDINGS = "lds-nz-building-outlines-GTiff-SHP.zip"
ZIP_LIDAR = "lds-new-zealand-2layers-GTiff-SHP.zip"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def resolve_file(pattern):
    """Finds a file in BASE_DIR matching a pattern."""
    files = list(BASE_DIR.glob(pattern))
    return files[0] if files else None

def unzip_data():
    """Handles the complex unzipping logic."""
    logger.info("--- Phase 1: Unzipping ---")
    
    # 1. Unzip Buildings
    bldg_zip = resolve_file(ZIP_BUILDINGS)
    if not bldg_zip:
        logger.error(f"Missing building zip: {ZIP_BUILDINGS}")
        sys.exit(1)
        
    logger.info(f"Unzipping buildings: {bldg_zip.name}")
    with zipfile.ZipFile(bldg_zip, 'r') as z:
        z.extractall(BASE_DIR)

    # 2. Unzip Lidar (DSM/DEM)
    lidar_zip = resolve_file(ZIP_LIDAR)
    if not lidar_zip:
        logger.error(f"Missing lidar zip: {ZIP_LIDAR}")
        sys.exit(1)

    logger.info(f"Unzipping lidar layers: {lidar_zip.name}")
    with zipfile.ZipFile(lidar_zip, 'r') as z:
        z.extractall(BASE_DIR)
        
    logger.info("Unzip complete.")

def sample_raster_series(gdf, raster_dir_name, target_col):
    """
    Iterates over a folder of TIFs. For each TIF, finds intersecting
    buildings and samples the raster value at their centroids.
    """
    # Find the directory
    # We search recursively because the unzip structure might vary slightly
    candidates = list(BASE_DIR.rglob(raster_dir_name))
    if not candidates:
        logger.error(f"Could not find raster folder: {raster_dir_name}")
        return gdf
    
    raster_dir = candidates[0]
    tifs = list(raster_dir.glob("*.tif"))
    
    logger.info(f"Sampling {len(tifs)} tiles from {raster_dir.name} into column '{target_col}'...")
    
    # Ensure spatial index exists
    if gdf.sindex is None:
        logger.info("Building spatial index...")
        gdf.sindex
        
    # Initialize column with NaN
    if target_col not in gdf.columns:
        gdf[target_col] = np.nan

    count_sampled = 0
    
    for i, tif in enumerate(tifs):
        if i % 50 == 0 and i > 0:
            logger.info(f"Processed {i}/{len(tifs)} tiles...")
            
        try:
            with rasterio.open(tif) as src:
                # 1. Check bounds overlap
                # Create a polygon for the raster bounds
                # (left, bottom, right, top)
                b = src.bounds
                tile_box = box(b.left, b.bottom, b.right, b.top)
                
                # 2. Find buildings centroids that intersect this tile
                # We use the spatial index for speed
                possible_matches_index = list(gdf.sindex.intersection(tile_box.bounds))
                possible_matches = gdf.iloc[possible_matches_index]
                
                # Further filter precise intersection with centroid
                # (The sindex is bounding box based, so we check actual geometry)
                # We only care if the centroid is in the tile
                centroids = possible_matches.centroid
                precise_matches = centroids[centroids.within(tile_box)]
                
                if precise_matches.empty:
                    continue
                    
                # 3. Sample the raster
                # rasterio.sample expects list of (x,y)
                coords = [(pt.x, pt.y) for pt in precise_matches]
                
                # Sample returns a generator of arrays (1 value per band)
                # We assume band 1
                values = [val[0] for val in src.sample(coords)]
                
                # 4. Assign values back to DataFrame
                # Use the index of precise_matches to update the main gdf
                gdf.loc[precise_matches.index, target_col] = values
                count_sampled += len(values)
                
        except Exception as e:
            logger.warning(f"Failed to read tile {tif.name}: {e}")

    logger.info(f"Finished sampling {target_col}. Total points updated: {count_sampled}")
    return gdf

def process_auckland():
    # 1. Unzip
    unzip_data()
    
    # 2. Load Buildings
    logger.info("--- Phase 2: Vector Ingest ---")
    shp_path = list(BASE_DIR.rglob("nz-building-outlines.shp"))
    if not shp_path:
        logger.error("Could not find nz-building-outlines.shp")
        sys.exit(1)
        
    logger.info(f"Reading shapefile: {shp_path[0]}")
    gdf = gpd.read_file(shp_path[0], engine="fiona") # Robust read
    
    logger.info(f"Initial CRS: {gdf.crs}")
    if gdf.crs.to_string() != TARGET_CRS:
        logger.info(f"Reprojecting to {TARGET_CRS}...")
        gdf = gdf.to_crs(TARGET_CRS)
    
    # Calculate Centroids once (in Projected CRS) for sampling
    # We do this implicitly in the sampling function using .centroid property
    
    # 3. Sample DSM (Top Surface)
    # Folder name from your notes: "auckland-lidar-1m-dsm-2013"
    logger.info("--- Phase 3: DSM Sampling ---")
    gdf = sample_raster_series(gdf, "auckland-lidar-1m-dsm-2013", "dsm_val")
    
    # 4. Sample DEM (Ground Surface)
    # Folder name from your notes: "new-zealand-lidar-1m-dem"
    logger.info("--- Phase 4: DEM Sampling ---")
    gdf = sample_raster_series(gdf, "new-zealand-lidar-1m-dem", "dem_val")
    
    # 5. Calculate Height
    logger.info("--- Phase 5: Height Calculation ---")
    # Clean nodata values (often -9999 or extremely large/small)
    # Filter reasonable range for Lidar values (e.g. -100m to 3000m)
    mask_valid_dsm = (gdf["dsm_val"] > -100) & (gdf["dsm_val"] < 4000)
    mask_valid_dem = (gdf["dem_val"] > -100) & (gdf["dem_val"] < 4000)
    
    gdf.loc[~mask_valid_dsm, "dsm_val"] = np.nan
    gdf.loc[~mask_valid_dem, "dem_val"] = np.nan
    
    # height = DSM - DEM
    gdf["height"] = gdf["dsm_val"] - gdf["dem_val"]
    
    # QA Check
    valid_heights = gdf["height"].notna().sum()
    logger.info(f"Calculated height for {valid_heights} / {len(gdf)} buildings.")
    logger.info(f"Mean Height: {gdf['height'].mean():.2f}m")

    # 6. Clean & Write
    logger.info("--- Phase 6: Finalizing ---")
    
    # Filter geometry
    gdf = gdf[gdf.geometry.notna() & ~gdf.is_empty]
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    
    output_path = BASE_DIR / OUTPUT_FILENAME
    
    # Remove intermediate columns to keep output clean? 
    # Usually better to keep dsm/dem for debug, but strict schema reqs usually imply dropping them.
    # We'll keep them for now as this is a complex derivative product.
    
    if output_path.exists():
        output_path.unlink()
        
    logger.info(f"Writing to {output_path}...")
    try:
        gdf.to_file(output_path, driver="GPKG", layer=OUTPUT_LAYER)
        logger.info("Success.")
    except Exception as e:
        logger.error(f"Write failed: {e}")

if __name__ == "__main__":
    process_auckland()