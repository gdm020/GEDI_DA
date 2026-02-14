import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from pathlib import Path
import logging
import sys
import zipfile

# --- Configuration ---
BUILDING_ROOT = Path("/home/gmiddle/rs_paper/data_prep/")
GEDI_ROOT = Path("/home/gmiddle/rs_paper/gedi/")
LOG_FILE = "conflation_log.txt"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

def index_gedi_files(gedi_path):
    """
    Scans GEDI Zip files and indexes internal GeoJSONs.
    """
    logging.info("Indexing GEDI zip archives...")
    gedi_index = []
    
    zip_files = list(gedi_path.glob("*.zip"))
    if not zip_files:
        logging.warning(f"No .zip files found in {gedi_path}")
        return []

    for z_path in zip_files:
        try:
            with zipfile.ZipFile(z_path, 'r') as z:
                internal_files = z.namelist()
                geojsons = [f for f in internal_files if f.lower().endswith('.geojson')]
                
                if not geojsons:
                    continue

                for gj in geojsons:
                    try:
                        full_uri = f"zip://{z_path.absolute()}!{gj}"
                        # Quick bounds check
                        gdf = gpd.read_file(full_uri)
                        
                        if gdf.empty:
                            continue

                        # Standardize to 3857 for index bounds
                        if gdf.crs is None or gdf.crs.to_epsg() != 3857:
                             if gdf.crs is not None:
                                 gdf = gdf.to_crs(epsg=3857)
                             else:
                                 gdf.set_crs(epsg=3857, inplace=True)
                        
                        bounds = gdf.total_bounds
                        gedi_index.append({
                            "uri": full_uri, 
                            "bbox": box(*bounds)
                        })
                        
                    except Exception:
                        logging.error(f"Failed to index {gj} in {z_path.name}")

        except Exception as e:
            logging.error(f"Failed to open ZIP {z_path.name}: {e}")
            
    return gedi_index

def process_city(city_path, gedi_index):
    try:
        logging.info(f"--- Processing {city_path.name} ---")
        
        # 1. Load Buildings
        try:
            gdf_city = gpd.read_file(city_path)
        except Exception as e:
            logging.error(f"Failed to read building file {city_path.name}: {e}")
            return

        if gdf_city.empty:
            logging.warning(f"Skipping {city_path.name}: File is empty.")
            return

        # STRICT: Get Anchor CRS
        anchor_crs = gdf_city.crs
        if not anchor_crs:
            logging.error(f"SKIPPING {city_path.name}: No CRS defined.")
            return

        logging.info(f"  Anchor CRS identified: {anchor_crs.name if anchor_crs.name else anchor_crs.to_string()}")

        # 2. Filter GEDI Files
        try:
            city_bbox_3857_geom = box(*gdf_city.to_crs(epsg=3857).total_bounds)
        except Exception:
            logging.error(f"  Failed to reproject bounds for indexing check.")
            return

        # 3. Load & Validate GEDI Points
        relevant_gedi_dfs = []
        for gentry in gedi_index:
            if city_bbox_3857_geom.intersects(gentry['bbox']):
                try:
                    g_df = gpd.read_file(gentry['uri'])
                    
                    if g_df.empty:
                        continue

                    if 'rh95' not in g_df.columns:
                        logging.error(f"MISSING rh95 in {gentry['uri']}")
                        continue

                    if g_df.crs is None:
                        g_df.set_crs(epsg=3857, inplace=True)

                    g_df = g_df.to_crs(anchor_crs)
                    
                    # --- ROBUSTNESS FIX 1: Rename Source Column ---
                    # We rename 'rh95' to '_temp_gedi_val' immediately.
                    # This guarantees NO collision with the building file columns.
                    g_df = g_df.rename(columns={'rh95': '_temp_gedi_val'})
                    
                    relevant_gedi_dfs.append(g_df)
                except Exception as e:
                    logging.error(f"  Error loading {gentry['uri']}: {e}")

        # --- VALIDATION ---
        has_valid_gedi = False
        gdf_gedi = None
        
        if relevant_gedi_dfs:
            gdf_gedi = pd.concat(relevant_gedi_dfs, ignore_index=True)
            if not gdf_gedi.empty:
                has_valid_gedi = True
                logging.info(f"  Validated {len(relevant_gedi_dfs)} GEDI sources. Total points: {len(gdf_gedi)}")
            else:
                logging.warning(f"  GEDI files found but total points = 0.")
        else:
             logging.warning(f"  No GEDI points intersect this city.")

        count_rule_a = 0
        count_rule_b = 0
        
        # Initialize the FINAL destination column
        gdf_city['rh95'] = np.nan
        gdf_city['rh95'] = gdf_city['rh95'].astype('float')
        
        if 'gedi_n' not in gdf_city.columns:
            gdf_city['gedi_n'] = 0

        if has_valid_gedi:
            # --- Rule A: Direct Intersection ---
            # Buildings (Left) -> GEDI (Right, with column '_temp_gedi_val')
            joined = gpd.sjoin(gdf_city, gdf_gedi[['_temp_gedi_val', 'geometry']], how='left', predicate='intersects')
            
            # Find hits
            matches = joined[joined['_temp_gedi_val'].notna()]
            
            if not matches.empty:
                # Aggregate
                stats = matches.groupby(matches.index).agg(
                    max_val=('_temp_gedi_val', 'max'),
                    count_val=('_temp_gedi_val', 'count')
                )
                # Assign to the explicit 'rh95' column
                gdf_city.loc[stats.index, 'rh95'] = stats['max_val']
                gdf_city.loc[stats.index, 'gedi_n'] = stats['count_val']
                count_rule_a = len(stats)
            
            logging.info(f"  Rule A complete. Buildings hit: {count_rule_a}")

            # --- Rule B: Propagation (25 units) ---
            centroids = gdf_city.geometry.centroid
            gdf_centroids = gpd.GeoDataFrame(
                {'rh95': gdf_city['rh95']}, 
                geometry=centroids, 
                crs=anchor_crs,
                index=gdf_city.index
            )

            # Anchors: Buildings that now have a value in 'rh95'
            anchors = gdf_centroids[gdf_centroids['rh95'].notna()]
            # Targets: Buildings that are still NaN
            targets = gdf_centroids[gdf_centroids['rh95'].isna()]

            if not anchors.empty and not targets.empty:
                # --- ROBUSTNESS FIX 2: Rename Anchor Column ---
                # We take the anchors and rename 'rh95' to '_temp_anchor_val'
                # so the nearest join produces exactly that column name.
                anchors_clean = anchors[['rh95', 'geometry']].rename(columns={'rh95': '_temp_anchor_val'})
                
                joined_b = gpd.sjoin_nearest(
                    targets,
                    anchors_clean,
                    how='inner',
                    max_distance=25,
                    distance_col="dist"
                )
                
                if not joined_b.empty:
                    # We grab '_temp_anchor_val'
                    propagated = joined_b.groupby(joined_b.index)['_temp_anchor_val'].max()
                    
                    # Assign back to 'rh95'
                    gdf_city.loc[propagated.index, 'rh95'] = propagated
                    count_rule_b = len(propagated)

            logging.info(f"  Rule B complete. Buildings filled: {count_rule_b}")
        
        null_count = gdf_city['rh95'].isna().sum()

        # 5. Save Output
        output_name = city_path.stem + "_gedi_conflated.gpkg"
        output_path = city_path.parent / output_name
        
        # Write to file
        gdf_city.to_file(output_path, driver="GPKG")
        
        log_entry = (
            f"SUMMARY for {city_path.name}:\n"
            f"  Anchor CRS: {anchor_crs.name}\n"
            f"  Total Buildings: {len(gdf_city)}\n"
            f"  Rule A Hits: {count_rule_a}\n"
            f"  Rule B Fills: {count_rule_b}\n"
            f"  Final Nulls: {null_count}\n"
        )
        logging.info(log_entry)

    except Exception as e:
        logging.error(f"CRITICAL FAILURE processing {city_path.name}: {e}", exc_info=True)

def main():
    gedi_index = index_gedi_files(GEDI_ROOT)
    
    if not gedi_index:
        logging.error("No valid GEDI files found in zips. Aborting workflow.")
        return

    city_files = list(BUILDING_ROOT.rglob("*.gpkg"))
    city_files = [f for f in city_files if "_gedi_conflated" not in f.name]
    
    logging.info(f"Found {len(city_files)} building files to process.")
    
    for city_file in city_files:
        process_city(city_file, gedi_index)

if __name__ == "__main__":
    main()