import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import sys
import warnings
import gc
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = Path("/home/gmiddle/rs_paper/clean_fp/")
OUTPUT_DIR = Path("/home/gmiddle/rs_paper/morphologically_processed/")
NEIGHBOR_RADII = [100, 250, 500, 1000]
CHUNK_SIZE = 50000  # Process neighbors in batches of 50k rows to save RAM

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculate_geometry_metrics(gdf):
    """Calculates single-building morphological features with memory cleanup."""
    logger.info("   [1/3] Calculating geometric features...")
    
    # ensure index is reset for direct numpy assignment later
    gdf = gdf.reset_index(drop=True)

    # 1. Basic Geometry
    # We use .values to work with numpy arrays where possible for speed/mem
    gdf["area"] = gdf.area.astype(np.float32)
    gdf["perimeter"] = gdf.length.astype(np.float32)
    
    # 2. Convex Hull (Heavy operation, delete immediately)
    hulls = gdf.convex_hull
    gdf["convex_hull_area"] = hulls.area.astype(np.float32)
    gdf["convexity"] = (hulls.length / gdf["perimeter"]).astype(np.float32)
    del hulls
    gc.collect()
    
    # 3. Bounding Box
    bounds = gdf.bounds
    gdf["width"] = (bounds["maxx"] - bounds["minx"]).astype(np.float32)
    gdf["height_bbox"] = (bounds["maxy"] - bounds["miny"]).astype(np.float32)
    del bounds
    gc.collect()
    
    # 4. Shape Descriptors (Vectorized Numpy ops)
    # Compactness
    area = gdf["area"].values
    perim = gdf["perimeter"].values
    
    gdf["compactness"] = ((4 * np.pi * area) / (perim ** 2)).astype(np.float32)
    gdf["IPQ"] = gdf["compactness"] # Alias
    
    # Fractality
    # 1 - [log(area) / (2 * log(perimeter))]
    with np.errstate(divide='ignore', invalid='ignore'):
        fractality = 1 - (np.log(area) / (2 * np.log(perim)))
        # Fix infinite/nan cases
        fractality[~np.isfinite(fractality)] = np.nan
        gdf["fractality"] = fractality.astype(np.float32)
        
        # Cooke JC
        cooke = (perim / (4 * np.sqrt(area))) - 1
        cooke[~np.isfinite(cooke)] = np.nan
        gdf["cooke_jc_index"] = cooke.astype(np.float32)
    
    # Rectangularity
    envel_area = gdf["width"] * gdf["height_bbox"]
    gdf["rectangularity"] = (area / envel_area).astype(np.float32)
    
    # Aspect Ratio & Eccentricity
    w = gdf["width"].values
    h = gdf["height_bbox"].values
    gdf["aspect_ratio"] = (h / w).astype(np.float32)
    gdf["diagonal"] = np.sqrt(w**2 + h**2).astype(np.float32)
    
    min_dim = np.minimum(w, h)
    max_dim = np.maximum(w, h)
    with np.errstate(divide='ignore', invalid='ignore'):
        ecc = np.sqrt(1 - (min_dim**2 / max_dim**2))
        ecc[~np.isfinite(ecc)] = 0
        gdf["eccentricity"] = ecc.astype(np.float32)

    # 5. Vertex Count
    # Using a list comprehension is often faster/lighter than .apply for simple geom ops
    def get_pts(g):
        if g is None or g.is_empty: return 0
        if g.geom_type == 'Polygon': return len(g.exterior.coords)
        if g.geom_type == 'MultiPolygon': return sum(len(p.exterior.coords) for p in g.geoms)
        return 0
    
    gdf["vertex_count"] = [get_pts(g) for g in gdf.geometry]
    gdf["vertex_count"] = gdf["vertex_count"].astype(np.int32)

    # 6. Extent (Requires reprojection - do it safely)
    try:
        # Reproject only geometry to WGS84 temporarily
        wgs_bounds = gdf.geometry.to_crs("EPSG:4326").bounds
        gdf["lat_dif"] = (wgs_bounds["maxy"] - wgs_bounds["miny"]).astype(np.float32)
        gdf["long_dif"] = (wgs_bounds["maxx"] - wgs_bounds["minx"]).astype(np.float32)
        del wgs_bounds
    except Exception:
        gdf["lat_dif"] = np.nan
        gdf["long_dif"] = np.nan
    
    gc.collect()
    return gdf

def calculate_neighborhood_metrics_chunked(gdf, radii):
    """
    Calculates neighborhood stats using a chunked approach.
    Crucial for RAM safety: never computes *all* neighbor lists at once.
    """
    logger.info("   [2/3] Calculating neighborhood features (Chunked)...")
    
    n_total = len(gdf)
    
    # 1. Build Tree (this is usually safe for RAM, it's the query that hurts)
    centroids = gdf.centroid
    coords = np.column_stack([centroids.x, centroids.y])
    tree = cKDTree(coords)
    
    # Free up the geoseries immediately
    del centroids
    gc.collect()

    # 2. Iterate by Radius
    for r in radii:
        logger.info(f"      Processing Radius: {r}m")
        
        # Pre-allocate numpy arrays for results (Float32 saves RAM)
        # Using arrays is much safer than appending to lists 1 million times
        res_count = np.zeros(n_total, dtype=np.int32)
        res_mean  = np.full(n_total, np.nan, dtype=np.float32)
        res_std   = np.full(n_total, np.nan, dtype=np.float32)
        res_min   = np.full(n_total, np.nan, dtype=np.float32)
        res_max   = np.full(n_total, np.nan, dtype=np.float32)
        res_cv    = np.full(n_total, np.nan, dtype=np.float32)

        # 3. Chunked Query
        # Process indices in batches
        for start_idx in tqdm(range(0, n_total, CHUNK_SIZE), desc=f"      Radius {r}m", leave=False):
            end_idx = min(start_idx + CHUNK_SIZE, n_total)
            
            # Identify coordinates for this batch
            batch_coords = coords[start_idx:end_idx]
            
            # Query ONLY this batch
            # returns list of lists, but only for CHUNK_SIZE items
            batch_neighbors = tree.query_ball_point(batch_coords, r)
            
            # Calculate stats for the batch
            # We iterate manually because neighbors are ragged (diff lengths)
            for local_i, global_i in enumerate(range(start_idx, end_idx)):
                indices = batch_neighbors[local_i]
                
                # Filter out self-match (the building itself)
                # indices is a list of integers
                # We can assume the building is in its own radius. 
                # Faster approach: just len() - 1 if self is present, 
                # but explicit check is safer for data integrity.
                valid_indices = [idx for idx in indices if idx != global_i]
                n_count = len(valid_indices)
                
                res_count[global_i] = n_count
                
                if n_count > 0:
                    # Get coords of neighbors
                    # Advanced indexing in numpy
                    neighbor_coords = coords[valid_indices]
                    curr_coord = coords[global_i]
                    
                    # Distances
                    dists = np.linalg.norm(neighbor_coords - curr_coord, axis=1)
                    
                    # Stats
                    d_mean = np.mean(dists)
                    d_std = np.std(dists)
                    
                    res_mean[global_i] = d_mean
                    res_std[global_i]  = d_std
                    res_min[global_i]  = np.min(dists)
                    res_max[global_i]  = np.max(dists)
                    res_cv[global_i]   = (d_std / d_mean) if d_mean > 0 else 0
            
            # Explicit cleanup after every chunk
            del batch_neighbors
            # gc.collect() # Optional inside loop, might slow it down too much if called every chunk

        # 4. Assign columns to GDF
        gdf[f"n_{r}m_count"] = res_count
        gdf[f"n_{r}m_dist_mean"] = res_mean
        gdf[f"n_{r}m_dist_std"] = res_std
        gdf[f"n_{r}m_dist_min"] = res_min
        gdf[f"n_{r}m_dist_max"] = res_max
        gdf[f"n_{r}m_dist_cv"] = res_cv
        
        # Cleanup arrays for this radius
        del res_count, res_mean, res_std, res_min, res_max, res_cv
        gc.collect()

    return gdf

def process_files():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gpkg_files = list(BASE_DIR.glob("*.gpkg"))
    
    if not gpkg_files:
        logger.error(f"No GeoPackages found in {BASE_DIR}")
        return

    logger.info(f"Found {len(gpkg_files)} cities to process.")
    
    for fp in gpkg_files:
        output_path = OUTPUT_DIR / fp.name
        
        logger.info(f"\n==========================================")
        logger.info(f"Processing: {fp.stem}")
        logger.info(f"Output: {output_path}")
        
        try:
            # 1. Load Data
            # 'ignore_geometry=False' is default, explicit for clarity
            try:
                gdf = gpd.read_file(fp, engine="pyogrio")
            except Exception:
                gdf = gpd.read_file(fp)
            
            if gdf.empty:
                logger.warning("Empty file. Skipping.")
                continue
                
            logger.info(f"Loaded {len(gdf)} footprints.")

            # 2. Run Calculations
            gdf = calculate_geometry_metrics(gdf)
            gdf = calculate_neighborhood_metrics_chunked(gdf, NEIGHBOR_RADII)
            
            # 3. Save
            logger.info(f"   [3/3] Saving result...")
            if output_path.exists(): output_path.unlink()
            gdf.to_file(output_path, driver="GPKG", layer=fp.stem)
            
            # 4. Final Cleanup
            del gdf
            gc.collect()
            logger.info("Success.")
            
        except Exception as e:
            logger.error(f"Failed to process {fp.name}: {e}")
            gc.collect()

    logger.info("\nAll files processed.")

if __name__ == "__main__":
    process_files()