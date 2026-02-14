import os
import yaml
import glob
import logging
import sys
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
import pyogrio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial.distance import jensenshannon
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# --- CONFIGURATION & SETUP ---
warnings.filterwarnings("ignore")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --- CUSTOM METRICS ---
def huber_loss(y_true, y_pred, delta=1.35):
    """Calculates Mean Huber Loss."""
    residual = np.abs(y_true - y_pred)
    loss = np.where(residual <= delta, 
                    0.5 * residual**2, 
                    delta * (residual - 0.5 * delta))
    return np.mean(loss)

def tukey_biweight_loss(y_true, y_pred, c=4.685):
    """Calculates Tukey's Biweight Loss."""
    residual = y_true - y_pred
    # Estimate scale (MAD)
    mad = np.median(np.abs(residual - np.median(residual)))
    sigma = 1.4826 * mad + 1e-6 # Avoid zero div
    
    u = residual / (c * sigma)
    rho = np.where(np.abs(u) <= 1, 
                   (c**2 / 6) * (1 - (1 - u**2)**3), 
                   c**2 / 6)
    return np.mean(rho)

def calculate_metrics(y_true, y_pred, bins=50):
    """Calculates RMSE, MAE, R2, JSD, Huber, and Tukey."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Use np.sqrt for RMSE to avoid version conflicts
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Custom Robust Metrics
    huber = huber_loss(y_true, y_pred)
    tukey = tukey_biweight_loss(y_true, y_pred)
    
    # JSD
    p, _ = np.histogram(y_true, bins=bins, density=True)
    q, _ = np.histogram(y_pred, bins=bins, density=True)
    p = p + 1e-12
    q = q + 1e-12
    jsd = jensenshannon(p, q)
    
    return rmse, mae, r2, jsd, huber, tukey

def create_violin_plot(df_results, city_name, output_dir):
    """Generates the Error Distribution by Height Bin violin plot."""
    palette = {"0-10m": "#1f77b4", "10-20m": "#ff7f0e", "20-30m": "#2ca02c", "30m+": "#d62728"}
    
    df_results['Residual'] = df_results['Observed Height'] - df_results['Predicted Height']
    df_results['HeightBin'] = pd.cut(
        df_results['Observed Height'], 
        bins=[0, 10, 20, 30, np.inf],
        labels=["0-10m", "10-20m", "20-30m", "30m+"],
        right=False
    )
    
    plot_df = df_results.dropna(subset=['HeightBin'])
    if plot_df.empty: return

    # Filter extreme outliers for plotting clarity
    thr = plot_df['Residual'].abs().quantile(0.975)
    plot_df = plot_df[plot_df['Residual'].abs() <= thr]
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x='HeightBin', y='Residual', data=plot_df,
        palette=palette, inner='quartile', cut=0
    )
    
    plt.ylim(-20, 20)
    plt.title(f"{city_name} Error Distribution by Height Bin")
    plt.xlabel("Height Bin (m)")
    plt.ylabel("Error (m)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    out_path = os.path.join(output_dir, f"{city_name}_Error_Violin.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

# --- WORKER FUNCTION FOR PARALLEL PROCESSING ---
def process_single_city(file_info):
    """
    Worker function to process a single city.
    file_info: tuple (city_name, file_path, config_dict)
    """
    city_name, file_path, cfg = file_info
    
    # Re-setup logger for worker process
    worker_logger = logging.getLogger(f"worker_{city_name}")
    if not worker_logger.handlers:
        logging.basicConfig(level=logging.INFO)

    try:
        worker_logger.info(f"Starting processing: {city_name}")
        
        # 1. Load Data
        df = pyogrio.read_dataframe(file_path)
        target_col = cfg['target_column']
        ignore_cols = set(cfg.get('ignore_columns', []) + ["geometry", "geom", target_col])
        features = [c for c in df.columns if c not in ignore_cols]
        
        # --- ROBUST DATA CLEANING ---
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        valid_mask = df[target_col].notnull() & np.isfinite(df[target_col]) & (df[target_col] > 0)
        df_train_full = df[valid_mask].copy()
        
        if df_train_full.empty:
            worker_logger.warning(f"Skipping {city_name}: No valid target data.")
            return None

        # Clean features (Inf -> Nan)
        df_train_full[features] = df_train_full[features].replace([np.inf, -np.inf], np.nan)
        df_train_full = df_train_full.dropna(subset=features, how='all')
        
        if len(df_train_full) < 50:
            worker_logger.warning(f"Skipping {city_name}: Not enough data.")
            return None

        # Log Transform
        df_train_full['log_height'] = np.log1p(df_train_full[target_col])
        
        if not np.isfinite(df_train_full['log_height']).all():
             worker_logger.warning(f"Skipping {city_name}: Infinite Y values.")
             return None
        
        # Stratification
        n_bins = min(20, len(df_train_full) // 5)
        n_bins = max(2, n_bins)
        try:
            df_train_full['height_bin'] = pd.qcut(df_train_full[target_col], q=n_bins, duplicates='drop')
        except ValueError:
            df_train_full['height_bin'] = pd.cut(df_train_full[target_col], bins=n_bins)

        X = df_train_full[features]
        y = df_train_full['log_height']
        
        # Split
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, stratify=df_train_full['height_bin'], test_size=0.3, random_state=42
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        sample_weights = 1.0 / (df_train_full.loc[y_train.index, target_col] + 1.0)

        # Pipeline
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', RobustScaler()),
            ('model', XGBRegressor(
                objective='reg:absoluteerror',
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=5  # 5 Cores per city
            ))
        ])

        # Search Space
        search_space = {
            'model__n_estimators': Integer(600, 2200),
            'model__max_depth': Integer(2, 12),
            'model__learning_rate': Real(0.01, 0.12, prior='log-uniform'),
            'model__min_child_weight': Integer(1, 10),
            'model__gamma': Real(0.0, 1.0),
            'model__reg_alpha': Real(1e-8, 10.0, prior='log-uniform'),
            'model__reg_lambda': Real(1e-8, 20.0, prior='log-uniform')
        }

        # Optimization
        opt = BayesSearchCV(
            pipeline,
            search_spaces=search_space,
            n_iter=50, 
            scoring='neg_mean_absolute_error',
            cv=10,
            verbose=0,
            n_jobs=1, # Sequential search, XGB uses the cores
            random_state=42
        )

        opt.fit(X_train, y_train, model__sample_weight=sample_weights)
        model = opt.best_estimator_
        
        # Predictions
        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)
        y_val_orig = df_train_full.loc[y_val.index, target_col]
        
        rmse, mae, r2, jsd, huber, tukey = calculate_metrics(y_val_orig, y_pred)
        
        metrics_dict = {
            "City": city_name, "RMSE": rmse, "MAE": mae, "R2": r2, 
            "JSD": jsd, "Huber_Loss": huber, "Tukey_Loss": tukey
        }
        
        # Prep GeoJSON data
        results_df = X_val.copy()
        results_df['Observed Height'] = y_val_orig
        results_df['Predicted Height'] = y_pred
        results_df['City'] = city_name
        
        gdf_results = None
        if 'geometry' in df.columns:
            # Reattach geometry
            geoms = df.loc[y_val.index, 'geometry']
            gdf_results = gpd.GeoDataFrame(results_df, geometry=geoms, crs=df.crs)

        return (city_name, metrics_dict, gdf_results, results_df)

    except Exception as e:
        worker_logger.error(f"Error in {city_name}: {e}")
        return None

# --- MAIN ORCHESTRATOR ---
def run_pipeline():
    cfg = load_config()
    input_dir = Path(cfg['input_dir'])
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(input_dir.glob("*.gpkg")))
    if not files:
        logger.error(f"No .gpkg files found in {input_dir}")
        return

    # Deduplication Logic
    city_map = {}
    for f in files:
        # Assumes format "CityName_v1.gpkg" -> "CityName"
        base_name = f.stem.split('_')[0]
        if base_name not in city_map:
            city_map[base_name] = f
        else:
            logger.info(f"Skipping duplicate version for '{base_name}': {f.name} (keeping {city_map[base_name].name})")

    unique_files = list(city_map.items())
    logger.info(f"Found {len(files)} files. Processing {len(unique_files)} unique cities.")

    tasks = []
    for city_name, file_path in unique_files:
        if (output_dir / f"{city_name}_metrics.csv").exists():
            logger.info(f"Skipping {city_name}: Output exists.")
            continue
        tasks.append((city_name, file_path, cfg))

    # Parallel Execution: 5 Cities at once
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_city, task): task[0] for task in tasks}
        
        for future in as_completed(futures):
            city_name = futures[future]
            try:
                result = future.result()
            except Exception as e:
                logger.error(f"Crash in worker for {city_name}: {e}")
                continue
            
            if result:
                c_name, metrics, gdf, df_res = result
                
                # 1. Save Metrics
                pd.DataFrame([metrics]).to_csv(output_dir / f"{c_name}_metrics.csv", index=False)
                
                # 2. Save Individual GeoJSON
                if gdf is not None:
                    geojson_path = output_dir / f"{c_name}_val_predictions.geojson"
                    try:
                        gdf.to_file(geojson_path, driver="GeoJSON")
                    except Exception as e:
                        logger.error(f"Failed to save GeoJSON for {c_name}: {e}")
                
                # 3. Save Violin Plot
                create_violin_plot(df_res, c_name, str(output_dir))
                
                logger.info(f"Finished {c_name}: RMSE={metrics['RMSE']:.2f}")
            else:
                logger.warning(f"City {city_name} returned no results.")

if __name__ == "__main__":
    run_pipeline()
