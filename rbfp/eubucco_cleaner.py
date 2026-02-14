import geopandas as gpd
import pandas as pd
import logging
import sys
import zipfile
import shutil
import os
import fiona
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path("/home/gmiddle/rs_paper/bfp/")

CITY_CONFIGS = {
    "berlin":    {"zip_pattern": "*Berlin.gpkg.zip",    "output_name": "berlin.gpkg",    "epsg": "EPSG:32633"},
    "madrid":    {"zip_pattern": "*Madrid.gpkg.zip",    "output_name": "madrid.gpkg",    "epsg": "EPSG:32630"},
    "amsterdam": {"zip_pattern": "*Amsterdam.gpkg.zip", "output_name": "amsterdam.gpkg", "epsg": "EPSG:32631"},
    "warsaw":    {"zip_pattern": "*Warszawa.gpkg.zip",  "output_name": "warsaw.gpkg",    "epsg": "EPSG:32634"}
}

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def resolve_file(pattern):
    files = list(BASE_DIR.glob(pattern))
    return files[0] if files else None

def read_and_merge_layers(gpkg_path, engine="pyogrio"):
    """
    Smart loader: Checks all layers. If multiple valid polygon layers exist,
    it merges them. Skips non-spatial layers if possible.
    """
    layers = fiona.listlayers(gpkg_path)
    logger.info(f"   Found layers: {layers}")
    
    gdfs = []
    for layer_name in layers:
        try:
            # Read layer (check bounds or geometry type if needed, but simple read is usually fine)
            sub_gdf = gpd.read_file(gpkg_path, layer=layer_name, engine=engine)
            
            # Sanity check: does it have geometry?
            if sub_gdf.geometry.notna().any():
                # Normalize columns if needed here, but usually EUBUCCO is consistent
                gdfs.append(sub_gdf)
            else:
                logger.warning(f"   Skipping layer '{layer_name}': No valid geometry.")
        except Exception as e:
            logger.warning(f"   Could not read layer '{layer_name}': {e}")

    if not gdfs:
        raise ValueError("No valid geometry layers found in file.")

    if len(gdfs) == 1:
        return gdfs[0]
    
    logger.info(f"   Merging {len(gdfs)} layers into one...")
    # CRS Consistency check
    base_crs = gdfs[0].crs
    for i, g in enumerate(gdfs[1:]):
        if g.crs != base_crs:
             gdfs[i+1] = g.to_crs(base_crs)
             
    merged_gdf = pd.concat(gdfs, ignore_index=True)
    return merged_gdf

def process_cities():
    logger.info(f"Scanning directory: {BASE_DIR}")
    
    # Try using pyogrio for speed, fall back to fiona
    try:
        import pyogrio
        engine = "pyogrio" 
    except ImportError:
        engine = "fiona"

    for city, config in CITY_CONFIGS.items():
        logger.info(f"\n--- Processing {city.upper()} ---")
        
        # 1. Check if finalized output already exists (Optional safety, or overwrite)
        # For now, we assume we want to regenerate/repair.
        
        zip_path = resolve_file(config["zip_pattern"])
        if not zip_path:
            logger.warning(f"ZIP matching '{config['zip_pattern']}' not found.")
            continue
            
        target_path = BASE_DIR / config["output_name"]
        
        # Use a temporary extraction folder
        temp_extract_dir = BASE_DIR / f"temp_{city}"
        if temp_extract_dir.exists(): shutil.rmtree(temp_extract_dir)
        temp_extract_dir.mkdir()

        try:
            # --- EXTRACT ---
            logger.info(f"Unzipping {zip_path.name}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(temp_extract_dir)
            
            # Find the internal gpkg
            extracted_gpkgs = list(temp_extract_dir.glob("*.gpkg"))
            if not extracted_gpkgs:
                logger.error("No GPKG found in zip.")
                continue
            
            source_gpkg = extracted_gpkgs[0] # Assume the largest/main one is correct
            
            # --- READ & MERGE ---
            logger.info("Reading and merging layers...")
            gdf = read_and_merge_layers(source_gpkg, engine=engine)
            
            # --- REPROJECT ---
            target_epsg = config["epsg"]
            
            # Handle missing CRS
            if not gdf.crs:
                logger.warning(f"CRS missing. Forcing {target_epsg}...")
                gdf.set_crs(target_epsg, inplace=True)
            else:
                current_crs = gdf.crs.to_string()
                if current_crs != target_epsg:
                    logger.info(f"Reprojecting {current_crs} -> {target_epsg}")
                    gdf = gdf.to_crs(target_epsg)
                else:
                    logger.info("CRS matches.")

            # --- SAFE WRITE ---
            # Write to a temp file first!
            temp_output_file = BASE_DIR / f"temp_write_{city}.gpkg"
            if temp_output_file.exists(): temp_output_file.unlink()
            
            logger.info(f"Saving to temp file: {temp_output_file.name}...")
            gdf.to_file(temp_output_file, driver="GPKG", layer=city, engine=engine)
            
            # If successful, rename to final
            if target_path.exists():
                target_path.unlink()
            
            temp_output_file.rename(target_path)
            logger.info(f"Success! Final file: {target_path.name}")

        except Exception as e:
            logger.error(f"Failed to process {city}: {e}")
        
        finally:
            # Cleanup extraction temp dir
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)

    logger.info("\n--- Batch Processing Complete ---")

if __name__ == "__main__":
    process_cities()