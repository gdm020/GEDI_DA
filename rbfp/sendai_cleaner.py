import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import sys
import warnings
import fiona
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path("/home/gmiddle/rs_paper/bfp/")
INPUT_FILENAME = "sendai.gpkg"
OUTPUT_FILENAME = "sendai_footprints.gpkg"
OUTPUT_LAYER = "sendai_footprints"
SOURCE_LAYER_NAME = "main.building_lod0"

# CRS: WGS 84 / UTM Zone 54N (Meter-based for Sendai)
TARGET_CRS = "EPSG:32654"

INPUT_PATH = BASE_DIR / INPUT_FILENAME
OUTPUT_PATH = BASE_DIR / OUTPUT_FILENAME

# --- Engine & Warning Config ---
# Use Pyogrio if available for speed, else Fiona
try:
    import pyogrio
    ENGINE = "pyogrio"
except ImportError:
    ENGINE = "fiona"

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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

def process_sendai():
    logger.info(f"Input: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        logger.error(f"File not found: {INPUT_PATH}")
        sys.exit(1)

    gdf = None

    # 1. Ingest
    try:
        logger.info(f"Attempting to read layer '{SOURCE_LAYER_NAME}'...")
        gdf = gpd.read_file(INPUT_PATH, layer=SOURCE_LAYER_NAME, engine=ENGINE)
        logger.info("Ingest successful.")
    except Exception as e:
        logger.warning(f"Failed to read specific layer '{SOURCE_LAYER_NAME}': {e}")
        logger.info("Listing available layers...")
        try:
            layers = fiona.listlayers(INPUT_PATH)
            logger.info(f"Available layers: {layers}")
            if len(layers) > 0:
                logger.info(f"Falling back to first layer: {layers[0]}")
                gdf = gpd.read_file(INPUT_PATH, layer=layers[0], engine=ENGINE)
            else:
                logger.error("No layers found in GeoPackage.")
                sys.exit(1)
        except Exception as e2:
            logger.error(f"Fatal error reading GeoPackage: {e2}")
            sys.exit(1)

    # 2. Schema Normalization
    if gdf.geometry.name != "geometry":
        gdf = gdf.rename(columns={gdf.geometry.name: "geometry"}).set_geometry("geometry")
        
    logger.info(f"Reprojecting to {TARGET_CRS} (Meter-based)...")
    if not gdf.crs:
        logger.warning("Source CRS missing. Assuming EPSG:4326 (common for GML/CityGML conversions).")
        gdf.set_crs("EPSG:4326", inplace=True)
        
    gdf = gdf.to_crs(TARGET_CRS)

    # Height Logic: Rename measured_height -> height
    logger.info("Standardizing height column (measured_height -> height)...")
    
    if "measured_height" in gdf.columns:
        # Coerce to numeric
        gdf["height"] = pd.to_numeric(gdf["measured_height"], errors="coerce").astype("float64")
        
        # Log stats
        null_count = gdf["height"].isna().sum()
        logger.info(f"Height stats: {len(gdf) - null_count} valid, {null_count} NULL")
    else:
        logger.warning("Column 'measured_height' not found! 'height' will be all NULL.")
        gdf["height"] = np.nan

    # 3. Filter
    logger.info("Filtering non-polygons...")
    gdf = gdf[gdf.geometry.notna() & ~gdf.is_empty]
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    # 4. Write
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
    process_sendai()