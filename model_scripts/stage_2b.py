import pandas as pd
import xgboost as xgb
import yaml
import logging
import sys
import gc
import random
import numpy as np
import pyogrio
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import jensenshannon, cosine
from scipy.stats import wasserstein_distance, iqr, median_abs_deviation, linregress
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor
from sklearn.isotonic import IsotonicRegression

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.ERROR)

# --- CONFIGURATION ---
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --- METRICS ---
def calculate_metrics(y_true, y_pred, bins=50):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]; y_pred = y_pred[mask]
    
    if len(y_true) < 2: return {}

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    if min_val == max_val: 
        jsd = 0.0 if np.array_equal(y_true, y_pred) else 1.0
    else:
        p, _ = np.histogram(y_true, bins=bins, range=(min_val, max_val), density=True)
        q, _ = np.histogram(y_pred, bins=bins, range=(min_val, max_val), density=True)
        p += 1e-12; q += 1e-12
        p /= np.sum(p); q /= np.sum(q)
        jsd = jensenshannon(p, q)

    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "EMD": wasserstein_distance(y_true, y_pred),
        "JSD": jsd
    }

# --- SHAP & FEATURE SELECTION ---
def compute_shap_permutation_importance(model, X_eval, feature_cols, group_cols=['city_id', 'strat_bin']):
    logger.info("Computing Balanced SHAP Permutation Importance...")
    explainer = shap.TreeExplainer(model)
    X_matrix = X_eval[feature_cols]
    
    if len(X_matrix) > 5000:
        strat_key = X_eval['city_id'].astype(str) + "_" + X_eval['strat_bin'].astype(str)
        try:
            X_sample, _ = train_test_split(X_eval, train_size=5000, stratify=strat_key, random_state=42)
        except ValueError:
            X_sample, _ = train_test_split(X_eval, train_size=5000, random_state=42)
        X_eval = X_sample.copy()
        X_matrix = X_eval[feature_cols]

    base_shap_values = explainer.shap_values(X_matrix)
    if isinstance(base_shap_values, list): base_shap_values = base_shap_values[0]
    base_mean_abs = np.mean(np.abs(base_shap_values), axis=0)
    
    group_ids = X_eval.groupby(group_cols).grouper.group_info[0]
    counts = np.bincount(group_ids)
    singleton_mask = np.isin(group_ids, np.where(counts < 2)[0])
    
    importances = {}
    for i, col in enumerate(feature_cols):
        X_permuted = X_matrix.copy()
        permuted_vals = X_eval[col].values.copy()
        if not np.all(singleton_mask):
            valid_rows = ~singleton_mask
            permuted_vals[valid_rows] = X_eval[valid_rows].groupby(group_cols)[col].transform(np.random.permutation).values
        if np.any(singleton_mask):
            singleton_rows = singleton_mask
            permuted_vals[singleton_rows] = X_eval[singleton_rows].groupby('city_id')[col].transform(np.random.permutation).values
        
        X_permuted[col] = permuted_vals
        perm_shap_values = explainer.shap_values(X_permuted)
        if isinstance(perm_shap_values, list): perm_shap_values = perm_shap_values[0]
        perm_mean_abs = np.mean(np.abs(perm_shap_values), axis=0)
        importances[col] = base_mean_abs[i] - perm_mean_abs[i]

    return pd.DataFrame({'feature': list(importances.keys()), 'importance': list(importances.values())}).sort_values('importance', ascending=False)

def get_global_top_50_features(files, cfg, allowed_features=None, label="Global"):
    logger.info(f"[{label}] Selecting Top 50 Features...")
    X_train_list, y_train_list, X_eval_list = [], [], []
    height_bins = [0, 10, 20, 40, 70, 999]

    for idx, f in enumerate(files):
        try:
            df = pyogrio.read_dataframe(f, read_geometry=False)
            df = df[(df[cfg['target_column']] >= 2) & (df[cfg['target_column']] <= 200)]
            df['strat_bin'] = pd.cut(df[cfg['target_column']], bins=height_bins, labels=False)
            df['city_id'] = idx 
            
            # Pandas 2.2 Fix
            if len(df) > 2000:
                train_sample = df.groupby('strat_bin', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), 1000), random_state=42),
                    include_groups=False if pd.__version__ >= '2.2' else True
                )
            else:
                train_sample = df
            
            eval_sample = df.groupby('strat_bin', group_keys=False).apply(
                lambda x: x.sample(min(len(x), 50), random_state=42),
                include_groups=False if pd.__version__ >= '2.2' else True
            )
            eval_sample['strat_bin'] = df.loc[eval_sample.index, 'strat_bin']
            eval_sample['city_id'] = idx
            
            cols_drop = [c for c in cfg['ignore_columns'] if c in df.columns]
            t_df = train_sample.drop(columns=cols_drop + ['strat_bin', 'city_id'], errors='ignore')
            if allowed_features: t_df = t_df[[c for c in t_df.columns if c == cfg['target_column'] or c in allowed_features]]
            
            X_train_list.append(t_df.drop(columns=[cfg['target_column']]).select_dtypes(include=[np.number]))
            y_train_list.append(t_df[cfg['target_column']])
            
            e_df = eval_sample.drop(columns=cols_drop, errors='ignore')
            if allowed_features: 
                keep = [c for c in e_df.columns if c == cfg['target_column'] or c in allowed_features or c in ['strat_bin', 'city_id']]
                e_df = e_df[keep]
            X_eval_list.append(e_df)
        except: continue
        
    if not X_train_list: return []
    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)
    X_eval_full = pd.concat(X_eval_list, ignore_index=True)
    
    model = xgb.XGBRegressor(n_estimators=100, tree_method="hist", n_jobs=40, random_state=42)
    model.fit(X_train, y_train)
    feat_imp = compute_shap_permutation_importance(model, X_eval_full, X_train.columns.tolist())
    top_50 = feat_imp.head(50)['feature'].tolist()
    logger.info(f"[{label}] Selected {len(top_50)} features.")
    gc.collect()
    return top_50

# --- SIMILARITY ---
def calculate_similarity_matrix(files, target_col, sim_cols):
    logger.info(f"Computing Similarity Vectors ({len(sim_cols)} cols)...")
    raw_vectors = {}
    valid_files = {}

    for f in files:
        try:
            info = pyogrio.read_info(f)
            available = info['fields']
            present_cols = [c for c in sim_cols if c in available]
            if len(present_cols) / len(sim_cols) < 0.7: continue

            df = pyogrio.read_dataframe(f, read_geometry=False, columns=present_cols + [target_col])
            df = df[(df[target_col] >= 2) & (df[target_col] <= 200)]
            if df.empty: continue
            if len(df) > 50000: df = df.sample(50000, random_state=42)
            
            stats_list = []
            for col in sim_cols:
                if col in df.columns:
                    vals = df[col].astype(np.float32).values
                    q10 = np.percentile(vals, 10)
                    q50 = np.percentile(vals, 50)
                    q90 = np.percentile(vals, 90)
                    iqr_val = np.subtract(*np.percentile(vals, [75, 25]))
                    stats_list.extend([q10, q50, q90, iqr_val])
                else:
                    stats_list.extend([np.nan, np.nan, np.nan, np.nan])
            
            city_name = f.stem.replace("_gedi_conflated", "").replace("_data", "")
            raw_vectors[city_name] = np.array(stats_list, dtype=np.float32)
            valid_files[city_name] = f
            del df; gc.collect()
        except: pass
            
    if not raw_vectors: return {}
    city_names = list(raw_vectors.keys())
    matrix = np.stack([raw_vectors[c] for c in city_names])
    global_median = np.nanmedian(matrix, axis=0)
    global_iqr = iqr(matrix, axis=0, nan_policy='omit')
    global_iqr[global_iqr == 0] = 1.0
    scaled_matrix = (matrix - global_median) / global_iqr
    scaled_matrix = np.nan_to_num(scaled_matrix, nan=0.0)

    meta = {}
    for i, city in enumerate(city_names):
        meta[city] = {"file": valid_files[city], "scaled_vector": scaled_matrix[i]}
    return meta

def find_nearest_neighbor(target_city, meta):
    if target_city not in meta: return None
    target_vec = meta[target_city]['scaled_vector']
    best_city = None; best_score = float('inf')
    for city, data in meta.items():
        if city == target_city: continue
        dist = cosine(target_vec, data['scaled_vector'])
        if np.isnan(dist): dist = 2.0
        if dist < best_score:
            best_score = dist
            best_city = city
    return {"city": best_city, "file": meta[best_city]['file'] if best_city else None, "score": best_score}

# --- OPTUNA TUNING ---
def optimize_hyperparameters(X, y, w, cfg, label_prefix="tuning"):
    grid = cfg['param_grid']
    def objective(trial):
        params = {
            'objective': cfg['fixed_params']['objective'],
            'tree_method': cfg['fixed_params']['tree_method'],
            'eval_metric': 'rmse',
            'n_jobs': cfg['fixed_params']['n_jobs'],
            'learning_rate': trial.suggest_float('learning_rate', max(1e-8, min(grid['learning_rate'])), max(grid['learning_rate']), log=True),
            'max_depth': trial.suggest_int('max_depth', min(grid['max_depth']), max(grid['max_depth'])),
            'min_child_weight': trial.suggest_int('min_child_weight', min(grid['min_child_weight']), max(grid['min_child_weight'])),
            'gamma': trial.suggest_float('gamma', min(grid['gamma']), max(grid['gamma'])),
            'subsample': trial.suggest_float('subsample', min(grid['subsample']), max(grid['subsample'])),
            'colsample_bytree': trial.suggest_float('colsample_bytree', min(grid['colsample_bytree']), max(grid['colsample_bytree'])),
            'reg_alpha': trial.suggest_float('reg_alpha', max(1e-8, min(grid['reg_alpha'])), max(grid['reg_alpha']), log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', max(1e-8, min(grid['reg_lambda'])), max(grid['reg_lambda']), log=True),
        }
        n_round = trial.suggest_int('n_estimators', min(grid['n_estimators']), max(grid['n_estimators']), step=100)
        dtrain = xgb.DMatrix(X, label=y, weight=w) if w is not None else xgb.DMatrix(X, label=y)
        try:
            cv = xgb.cv(params, dtrain, num_boost_round=n_round, nfold=3, metrics='rmse', early_stopping_rounds=10, verbose_eval=False)
            return cv['test-rmse-mean'].iloc[-1]
        except: return float('inf')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=cfg['search']['n_iter'])
    return {**cfg['fixed_params'], **study.best_params}

# --- GEDI HELPERS: ROBUST WEIGHTED REFIT ---
BINS = [0, 10, 20, 40, 70, 999]

def get_stratified_calibration_sample(df, target_col='rh95', n_samples=100):
    """
    Selects 100 points maximizing spread across the height distribution.
    """
    try:
        df['calib_bin'] = pd.qcut(df[target_col], 5, labels=False, duplicates='drop')
        calib, _ = train_test_split(df, train_size=n_samples, stratify=df['calib_bin'], random_state=42)
        return calib
    except ValueError:
        return df.sample(min(len(df), n_samples), random_state=42)

def fit_isotonic_bias_correction_weighted(calib_df, target_col='height', gedi_col='rh95'):
    """
    1. Fits Initial Isotonic
    2. Computes residuals & robust scale (MAD)
    3. Calculates weights (Huber-style)
    4. Refits Isotonic with weights
    Returns: Fitted Model, Sigma Map (per bin)
    """
    if len(calib_df) < 10: return None, None
    
    X = calib_df[gedi_col].values
    y = calib_df[target_col].values
    
    # 1. Initial Fit
    iso_init = IsotonicRegression(out_of_bounds='clip')
    try:
        iso_init.fit(X, y)
    except:
        return None, None
        
    # 2. Compute Residuals & Robust Scale
    preds_init = iso_init.predict(X)
    residuals = y - preds_init
    
    mad = median_abs_deviation(residuals)
    s = 1.4826 * mad
    if s < 1e-6: s = 0.5 # Floor to prevent zero division
    
    # 3. Assign Robust Weights
    c = 2.5
    abs_e = np.abs(residuals)
    weights = np.ones_like(abs_e)
    
    mask_outlier = abs_e > c * s
    # Downweight outliers: w = c*s / |e|
    weights[mask_outlier] = (c * s) / abs_e[mask_outlier]
    weights = np.clip(weights, 0.05, 1.0)
    
    # 4. Refit Bias Model using Weights
    iso_final = IsotonicRegression(out_of_bounds='clip')
    try:
        iso_final.fit(X, y, sample_weight=weights)
    except:
        return None, None
        
    # 5. Sigma Map for Anchor Weighting
    # Use residuals from the final robust model to estimate bin noise
    preds_final = iso_final.predict(X)
    final_residuals = y - preds_final
    
    # Create bin map for sigma
    calib_df = calib_df.copy()
    calib_df['resid'] = final_residuals
    calib_df['bin'] = pd.cut(calib_df[gedi_col], bins=BINS, labels=False)
    
    def robust_sigma(x):
        if len(x) < 2: return 1.0
        return 1.4826 * median_abs_deviation(x)
        
    sigma_map = calib_df.groupby('bin')['resid'].apply(robust_sigma).to_dict()
    
    # Fill missing bins with global sigma
    global_sigma = robust_sigma(final_residuals)
    for b in range(len(BINS)-1):
        if b not in sigma_map: sigma_map[b] = global_sigma
        
    return iso_final, sigma_map

def get_common_features(X_src_cols, X_tgt_cols, feature_subset):
    return [c for c in feature_subset if c in X_src_cols and c in X_tgt_cols]

# --- PLOTTING ---
def generate_global_plots(preds_df, output_dir, batch_name):
    if preds_df.empty: return
    preds_df['Abs_Error'] = np.abs(preds_df['True_Height'] - preds_df['Pred_Height'])
    
    # 1. Binned Violin Plot
    plot_bins = [0, 10, 20, 30, 40, 50, 1000]
    preds_df['Height_Bin'] = pd.cut(preds_df['True_Height'], bins=plot_bins, labels=["0-10m", "10-20m", "20-30m", "30-40m", "40-50m", ">50m"])
    
    plt.figure(figsize=(14, 7))
    sns.violinplot(data=preds_df, x='Height_Bin', y='Abs_Error', hue='Test_Type', split=False, cut=0)
    plt.ylim(0, 30)
    plt.title(f"{batch_name} Absolute Error by Height Bin")
    plt.tight_layout()
    plt.savefig(output_dir / f"gedi_{batch_name}_error_violin_binned.png")
    plt.close()

    # 2. KDE Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(preds_df['True_Height'], label='True Height', fill=True, color='grey', alpha=0.3)
    test_types = preds_df['Test_Type'].unique()
    styles = ['--', '-.', ':']
    for i, ttype in enumerate(test_types):
        style = styles[i % len(styles)]
        subset = preds_df[preds_df['Test_Type'] == ttype]
        sns.kdeplot(subset['Pred_Height'], label=f"{ttype} Pred", linestyle=style)
    plt.xlim(0, 60)
    plt.title(f"{batch_name} Height Prediction KDE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"gedi_{batch_name}_height_kde.png")
    plt.close()

# --- BATCH EXECUTION ---
def run_experiment_batch(batch_name, feature_subset_raw, files, sim_cols, cfg, output_dir):
    logger.info(f"\n{'='*40}\nBATCH: {batch_name}\n{'='*40}")
    
    top_50 = get_global_top_50_features(files, cfg, allowed_features=set(feature_subset_raw), label=batch_name)
    if not top_50: return

    meta = calculate_similarity_matrix(files, cfg['target_column'], sim_cols)
    
    combined_metrics = []
    all_preds_list = [] 
    
    ADAPT_ROUNDS = cfg['search'].get('adapt_rounds', 100)
    MODEL_DIR = output_dir / "gedi_models" / batch_name
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for target_city, target_meta in meta.items():
        logger.info(f"[{batch_name}] Processing {target_city}...")
        target_file = target_meta['file']
        
        try:
            df = pyogrio.read_dataframe(target_file, read_geometry=True) 
            df = df[(df[cfg['target_column']] >= 2) & (df[cfg['target_column']] <= 200)]
            
            city_res = {"City": target_city}
            
            # --- 1. BASELINE ---
            df_g = df[df['rh95'].notnull() & (df['rh95'] > 0)]
            m_a = {}
            if len(df_g) > 50:
                m_a = calculate_metrics(df_g[cfg['target_column']], df_g['rh95'])
                for k, v in m_a.items(): city_res[f"Baseline_{k}"] = v
                city_res["Baseline_N"] = len(df_g)
                
                sub = df_g.sample(min(len(df_g), 1000), random_state=42)
                for _, r in sub.iterrows():
                    all_preds_list.append({
                        "City": target_city, "True_Height": r[cfg['target_column']], 
                        "Pred_Height": r['rh95'], "Test_Type": "Baseline"
                    })
            else:
                city_res["Baseline_N"] = 0

            # --- PREP ADAPT ---
            neighbor = find_nearest_neighbor(target_city, meta)
            if not neighbor: 
                combined_metrics.append(city_res); continue
                
            source_city = neighbor['city']
            source_file = neighbor['file']
            city_res["Source"] = source_city
            
            # Keep the 80/20 split internally for anchor selection so training sample size doesn't change
            df_remain, df_eval_orig = train_test_split(df, test_size=0.2, random_state=42)
            
            df_gedi_pool = df_remain[df_remain['rh95'].notnull() & (df_remain['rh95'] > 0)].copy()
            if len(df_gedi_pool) < 200:
                combined_metrics.append(city_res); continue
            
            # Stratified Calibration (using rh95 bins)
            df_calib = get_stratified_calibration_sample(df_gedi_pool, 'rh95', 100)
            df_anchors = df_gedi_pool.drop(df_calib.index)
            if len(df_anchors) < 50:
                combined_metrics.append(city_res); continue
            
            src_cols = pyogrio.read_info(source_file)['fields']
            tgt_cols = df.columns
            valid_feats = get_common_features(src_cols, tgt_cols, top_50)
            if len(valid_feats) < 5: 
                combined_metrics.append(city_res); continue
            
            # --- TRAIN BASE ---
            X_src = pyogrio.read_dataframe(source_file, read_geometry=False, columns=valid_feats+[cfg['target_column']])
            X_src = X_src[(X_src[cfg['target_column']] >= 2) & (X_src[cfg['target_column']] <= 200)]
            X_tune = X_src.sample(50000, random_state=42) if len(X_src) > 50000 else X_src
                
            logger.info(f"  Tuning base model on {source_city}...")
            best_params = optimize_hyperparameters(
                X_tune[valid_feats], X_tune[cfg['target_column']], None, cfg, 
                label_prefix=f"{target_city}_base"
            )
            n_est = best_params.pop('n_estimators', 1000)
            
            dtrain_src = xgb.DMatrix(X_src[valid_feats], label=X_src[cfg['target_column']])
            base_model = xgb.train(best_params, dtrain_src, num_boost_round=n_est, verbose_eval=False)
            base_model.save_model(MODEL_DIR / f"gedi_{batch_name}_{target_city}_base.json")
            
            # Evaluate and output against the FULL dataframe, not just the 20% validation chunk
            dtest = xgb.DMatrix(df[valid_feats], label=df[cfg['target_column']])
            
            # --- 2. NAIVE ADAPT ---
            dtrain_naive = xgb.DMatrix(df_anchors[valid_feats], label=df_anchors['rh95'])
            model_naive = xgb.train(best_params, dtrain_naive, num_boost_round=ADAPT_ROUNDS, xgb_model=base_model)
            model_naive.save_model(MODEL_DIR / f"gedi_{batch_name}_{target_city}_adapt_naive.json")
            
            preds_b = model_naive.predict(dtest)
            m_b = calculate_metrics(df[cfg['target_column']], preds_b)
            for k, v in m_b.items(): city_res[f"Adapt_{k}"] = v
            city_res["Adapt_N"] = len(df_anchors)
            
            sub_eval = pd.DataFrame({'True_Height': df[cfg['target_column']], 'Pred_Height': preds_b})
            if len(sub_eval) > 1000: sub_eval = sub_eval.sample(1000, random_state=42)
            sub_eval['City'] = target_city; sub_eval['Test_Type'] = 'Adapt_Naive'
            all_preds_list.extend(sub_eval.to_dict('records'))
            
            # --- 3. BC ADAPT (Weighted Robust Refit) ---
            iso_model, sigma_map = fit_isotonic_bias_correction_weighted(df_calib, cfg['target_column'], 'rh95')
            m_c = {}
            preds_c = np.zeros_like(preds_b)
            
            if iso_model:
                # Correct Anchor Labels
                y_bc = iso_model.predict(df_anchors['rh95'])
                y_bc = np.clip(y_bc, 2, 200)
                
                # Apply Noise Weights (1/sigma^2)
                anchor_bins = pd.cut(df_anchors['rh95'], bins=BINS, labels=False)
                try:
                    # Fill missing bins with 1.0 sigma if any
                    w_bc = 1.0 / (anchor_bins.map(sigma_map).fillna(1.0) ** 2 + 1e-6)
                    w_bc = w_bc.values
                except:
                    w_bc = None
                
                dtrain_bc = xgb.DMatrix(df_anchors[valid_feats], label=y_bc, weight=w_bc)
                model_bc = xgb.train(best_params, dtrain_bc, num_boost_round=ADAPT_ROUNDS, xgb_model=base_model)
                model_bc.save_model(MODEL_DIR / f"gedi_{batch_name}_{target_city}_adapt_bc.json")
                
                preds_c = model_bc.predict(dtest)
                m_c = calculate_metrics(df[cfg['target_column']], preds_c)
                
                for k, v in m_c.items(): city_res[f"AdaptBC_{k}"] = v
                city_res["BC_Type"] = "Weighted_Isotonic"

                sub_bc = pd.DataFrame({'True_Height': df[cfg['target_column']], 'Pred_Height': preds_c})
                if len(sub_bc) > 1000: sub_bc = sub_bc.sample(1000, random_state=42)
                sub_bc['City'] = target_city; sub_bc['Test_Type'] = 'Adapt_BC'
                all_preds_list.extend(sub_bc.to_dict('records'))
            else:
                city_res["BC_Failed"] = True

            # --- OUTPUTS ---
            combined_metrics.append(city_res)
            
            # 1. Per-City Metrics
            with open(output_dir / f"gedi_{batch_name}_stats_{target_city}.txt", "w") as f:
                f.write(f"City: {target_city}\nSource: {source_city}\n")
                f.write(f"Baseline: {m_a}\n")
                f.write(f"Adapt Naive: {m_b}\n")
                f.write(f"Adapt BC: {m_c}\n")
            
            # 2. Per-City Centroids GPKG - NOW OUTPUTS FULL DATAFRAME
            city_out_gdf = df[['geometry', cfg['target_column']]].copy()
            city_out_gdf.rename(columns={cfg['target_column']: 'True_Height'}, inplace=True)
            city_out_gdf['Baseline_Pred'] = df['rh95'] 
            city_out_gdf['Adapt_Naive_Pred'] = preds_b
            if iso_model:
                city_out_gdf['Adapt_BC_Pred'] = preds_c
            
            out_gpkg = output_dir / f"gedi_{batch_name}_centroids_{target_city}.gpkg"
            pyogrio.write_dataframe(city_out_gdf, out_gpkg, driver="GPKG")
            
            gc.collect()

        except Exception as e:
            logger.error(f"Failed {target_city}: {e}")
            continue

    # 3. Combined Batch Metrics
    if combined_metrics:
        pd.DataFrame(combined_metrics).to_csv(output_dir / f"gedi_{batch_name}_combined_metrics.csv", index=False)
    
    # 4. Global Plots
    if all_preds_list:
        full_preds = pd.DataFrame(all_preds_list)
        generate_global_plots(full_preds, output_dir, batch_name)

# --- MAIN ---
def run_gedi_tests():
    cfg = load_config()
    INPUT_DIR = Path(cfg['input_dir'])
    OUTPUT_DIR = Path(cfg['output_dir']) 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Collecting GEDI files...")
    all_files = sorted(list(INPUT_DIR.rglob("*_gedi_conflated.gpkg")))
    valid_files = []
    for f in all_files:
        try:
            info = pyogrio.read_info(f)
            if 'rh95' in info['fields']: valid_files.append(f)
        except: pass
        
    if not valid_files:
        logger.error("No valid GEDI files found.")
        return

    morph_list = cfg.get('morphological_features', [])
    rs_list = cfg.get('remote_sensing_features', [])
    
    SIM_COLS_MORPH = [
        'area', 'perimeter', 'compactness', 'rectangularity', 
        'n_100m_count', 'n_250m_count', 'n_500m_count', 'n_1000m_count'
    ]
    SIM_COLS_RS = rs_list[:8]

    # ALL
    run_experiment_batch("ALL", morph_list + rs_list, valid_files, SIM_COLS_MORPH, cfg, OUTPUT_DIR)
    
    # MORPH
    run_experiment_batch("MORPH", morph_list, valid_files, SIM_COLS_MORPH, cfg, OUTPUT_DIR)
    
    # RS
    run_experiment_batch("RS", rs_list, valid_files, SIM_COLS_RS, cfg, OUTPUT_DIR)

if __name__ == "__main__":
    run_gedi_tests()