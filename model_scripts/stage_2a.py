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
from scipy.stats import wasserstein_distance, iqr
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

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

# --- SHAP PERMUTATION IMPORTANCE (FIXED) ---
def compute_shap_permutation_importance(model, X_eval, feature_cols, group_cols=['city_id', 'strat_bin']):
    logger.info("Computing Balanced SHAP Permutation Importance...")
    explainer = shap.TreeExplainer(model)
    X_matrix = X_eval[feature_cols]
    
    if len(X_matrix) > 5000:
        logger.info(f"Subsampling eval set from {len(X_matrix)} to 5000 for SHAP speed.")
        strat_key = X_eval['city_id'].astype(str) + "_" + X_eval['strat_bin'].astype(str)
        try:
            # FIX: Correct usage of train_size to get the sample
            X_sample, _ = train_test_split(X_eval, train_size=5000, stratify=strat_key, random_state=42)
        except ValueError:
            logger.warning("Stratified subsample failed. Falling back to random.")
            X_sample, _ = train_test_split(X_eval, train_size=5000, random_state=42)
        X_eval = X_sample.copy()
        X_matrix = X_eval[feature_cols]

    base_shap_values = explainer.shap_values(X_matrix)
    if isinstance(base_shap_values, list): base_shap_values = base_shap_values[0]
    base_mean_abs = np.mean(np.abs(base_shap_values), axis=0)
    
    group_ids = X_eval.groupby(group_cols).grouper.group_info[0]
    counts = np.bincount(group_ids)
    singleton_mask = np.isin(group_ids, np.where(counts < 2)[0])
    
    if np.any(singleton_mask):
        logger.info(f"Handling {np.sum(singleton_mask)} rows in singleton groups via city-level permutation.")

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

# --- GLOBAL FEATURE SELECTION ---
def get_global_top_50_features(files, cfg, allowed_features=None, label="Global"):
    logger.info(f"[{label}] Feature Selection: Loading Data...")
    X_train_list, y_train_list, X_eval_list = [], [], []
    height_bins = [0, 10, 20, 40, 70, 999]

    for idx, f in enumerate(files):
        try:
            df = pyogrio.read_dataframe(f, read_geometry=False)
            df = df[(df[cfg['target_column']] >= 2) & (df[cfg['target_column']] <= 200)]
            
            df['strat_bin'] = pd.cut(df[cfg['target_column']], bins=height_bins, labels=False)
            df['city_id'] = idx 
            
            if len(df) > 2000:
                train_sample = df.groupby('strat_bin', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), 1000), random_state=42), 
                    include_groups=False if pd.__version__ >= '2.2' else True
                )
            else:
                train_sample = df
            
            N_EVAL = 50 
            eval_sample = df.groupby('strat_bin', group_keys=False).apply(
                lambda x: x.sample(min(len(x), N_EVAL), random_state=42), 
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
            
        except Exception: continue
        
    if not X_train_list: return [], []

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)
    X_eval_full = pd.concat(X_eval_list, ignore_index=True)
    
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train.loc[:, num_cols] = X_train.loc[:, num_cols].astype(np.float32)
    
    logger.info(f"[{label}] Training Selection Model...")
    model = xgb.XGBRegressor(n_estimators=100, tree_method="hist", n_jobs=40, random_state=42)
    model.fit(X_train, y_train)
    
    feature_cols = X_train.columns.tolist()
    feat_imp = compute_shap_permutation_importance(model, X_eval_full, feature_cols, group_cols=['city_id', 'strat_bin'])
    
    top_50 = feat_imp.head(50)['feature'].tolist()
    dropped = feat_imp.tail(len(feat_imp)-50)['feature'].tolist()
    
    logger.info(f"[{label}] Top 50 Selected via SHAP Permutation. Dropped {len(dropped)} features.")
    gc.collect()
    return top_50, dropped

# --- NEW SIMILARITY (VECTORIZED & ROBUST SCALED WITH IMPUTATION) ---
def calculate_similarity_matrix(files, target_col, sim_cols):
    """
    Computes robust similarity vectors.
    1. Checks for 70% column coverage.
    2. Computes stats (q10, q50, q90, IQR) for available columns; fills missing with NaN.
    3. Robust scales using Global Median/IQR (ignoring NaNs).
    4. Fills remaining NaNs with 0 (median) for cosine distance calculation.
    """
    logger.info(f"Computing Similarity Vectors on {len(sim_cols)} columns...")
    
    raw_vectors = {}
    valid_files = {}

    for f in files:
        try:
            info = pyogrio.read_info(f)
            available = info['fields']
            
            # --- FIX: 70% Intersection Rule ---
            present_cols = [c for c in sim_cols if c in available]
            if len(present_cols) / len(sim_cols) < 0.7:
                logger.warning(f"File {f.stem} missing too many sim cols ({len(present_cols)}/{len(sim_cols)}). Skipping.")
                continue

            # Read only present columns
            df = pyogrio.read_dataframe(f, read_geometry=False, columns=present_cols + [target_col])
            df = df[(df[target_col] >= 2) & (df[target_col] <= 200)]
            if df.empty: continue
            
            if len(df) > 50000: 
                df = df.sample(50000, random_state=42)
            
            # Calculate stats per column. If column missing, use NaNs.
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
            
            city_name = f.stem.replace("_data", "")
            raw_vectors[city_name] = np.array(stats_list, dtype=np.float32)
            valid_files[city_name] = f
            
            del df; gc.collect()
        except Exception as e:
            logger.error(f"Error processing {f.stem}: {e}")
            pass
            
    if not raw_vectors:
        return {}

    # 2. Robust Scaling (NaN-safe)
    city_names = list(raw_vectors.keys())
    matrix = np.stack([raw_vectors[c] for c in city_names])
    
    # Calculate global Median and IQR per dimension (ignoring NaNs)
    global_median = np.nanmedian(matrix, axis=0)
    global_iqr = iqr(matrix, axis=0, nan_policy='omit')
    
    # Avoid division by zero
    global_iqr[global_iqr == 0] = 1.0
    
    # Scale: Z = (V - Median) / IQR
    # This preserves NaNs
    scaled_matrix = (matrix - global_median) / global_iqr
    
    # Fill NaNs with 0.0 (Global Median in Z-score space) 
    # This ensures missing dimensions contribute 0 to the distance magnitude/direction bias
    scaled_matrix = np.nan_to_num(scaled_matrix, nan=0.0)

    # 3. Store Scaled Vectors in Meta
    meta = {}
    for i, city in enumerate(city_names):
        meta[city] = {
            "file": valid_files[city],
            "scaled_vector": scaled_matrix[i]
        }
        
    logger.info(f"Similarity matrix computed for {len(meta)} cities.")
    return meta

def find_similar_cities(target_city, meta, sim_cols_unused=None):
    if target_city not in meta: return []
    
    target_vec = meta[target_city]['scaled_vector']
    scores = []
    
    for city, data in meta.items():
        if city == target_city: continue
        
        dist = cosine(target_vec, data['scaled_vector'])
        if np.isnan(dist): dist = 2.0
            
        scores.append({"city": city, "file": data['file'], "score": dist})
        
    return sorted(scores, key=lambda x: x['score'])

# --- LOADING ---
def load_data_robust(filepath, valid_features, target_col, weight_factor=1.0):
    try:
        df = pyogrio.read_dataframe(filepath, read_geometry=False)
        df = df[(df[target_col] >= 2) & (df[target_col] <= 200)]
        if df.empty: return None, None, None
        
        available = [c for c in valid_features if c in df.columns]
        if not available: return None, None, None
        
        X = df[available].copy()
        numerics = X.select_dtypes(include=[np.number]).columns
        X[numerics] = X[numerics].astype(np.float32)
        
        y = df[target_col].astype(np.float32)
        w = np.full(len(y), weight_factor, dtype=np.float32)
        del df; gc.collect()
        return X, y, w
    except Exception as e:
        return None, None, None

# --- OPTUNA (INDEPENDENT RUNS) ---
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
        
        if w is None:
            dtrain = xgb.DMatrix(X, label=y)
        else:
            dtrain = xgb.DMatrix(X, label=y, weight=w)
            
        try:
            cv = xgb.cv(params, dtrain, num_boost_round=n_round, nfold=3, metrics='rmse', early_stopping_rounds=10, verbose_eval=False)
            return cv['test-rmse-mean'].iloc[-1]
        except: return float('inf')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=cfg['search']['n_iter'])
    return {**cfg['fixed_params'], **study.best_params}

# --- PLOTTING ---
def generate_shap_beeswarms(shap_values, features, output_dir, prefix):
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_order = np.argsort(mean_shap)[::-1]
    
    for i in range(0, min(shap_values.shape[1], 50), 10):
        plt.figure(figsize=(10, 6))
        idx = feature_order[i : i+10]
        shap.summary_plot(shap_values[:, idx], features.iloc[:, idx], show=False, plot_type="dot")
        plt.title(f"{prefix} SHAP Top {i+1}-{i+10}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_shap_top_{i+1}_{i+10}.png")
        plt.close()

def generate_global_plots(all_preds_df, output_dir, prefix):
    n_sample = min(all_preds_df['City'].value_counts().min(), 5000)
    indices = []
    for _, g in all_preds_df.groupby('City'):
        indices.extend(g.sample(n_sample, random_state=42).index.tolist() if len(g) >= n_sample else g.index.tolist())
    balanced = all_preds_df.loc[indices].copy()

    plt.figure(figsize=(12, 6))
    for m in ['Loco', '1to1', 'Adapt']:
        balanced[f"Error_{m}"] = np.abs(balanced['True_Height'] - balanced[f"{m}_Pred"])
    
    bins = [0, 10, 20, 30, 40, 50, 1000]
    balanced['Height_Bin'] = pd.cut(balanced['True_Height'], bins=bins, labels=["0-10m", "10-20m", "20-30m", "30-40m", "40-50m", ">50m"])
    melted = balanced.melt(id_vars=['Height_Bin'], value_vars=['Error_Loco', 'Error_1to1', 'Error_Adapt'], var_name='Model', value_name='Abs_Error')
    
    sns.violinplot(data=melted, x='Height_Bin', y='Abs_Error', hue='Model', split=False)
    plt.title(f"{prefix} Global Error Distribution (Equalized)")
    plt.ylim(0, 30); plt.savefig(output_dir / f"{prefix}_global_error_violin.png"); plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(balanced['True_Height'], label='True', fill=True, alpha=0.3)
    for m, s in zip(['Loco', '1to1', 'Adapt'], ['--', '-.', ':']):
        sns.kdeplot(balanced[f"{m}_Pred"], label=m, linestyle=s)
    plt.xlim(0, 60); plt.title(f"{prefix} Height KDE (Equalized)"); plt.legend(); plt.savefig(output_dir / f"{prefix}_global_kde.png"); plt.close()

# --- EXPERIMENT BATCH ---
def run_experiment_batch(batch_name, feature_subset, files, meta, sim_cols, cfg):
    logger.info(f"\n{'='*40}\nBATCH: {batch_name}\n{'='*40}")
    if not feature_subset: return
    
    OUTPUT_DIR = Path(cfg['output_dir'])
    MODEL_DIR = OUTPUT_DIR / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    res, preds, shaps = [], [], []
    height_bins = [0, 10, 20, 40, 70, 999]
    
    for test_idx, f in enumerate(files):
        city = f.stem.replace("_data", "")
        if city not in meta: continue
        logger.info(f"[{batch_name}] {test_idx+1}: {city}")
        
        df = pyogrio.read_dataframe(f, read_geometry=True)
        cols_drop = [c for c in cfg['ignore_columns'] if c in df.columns]
        if cols_drop: df = df.drop(columns=cols_drop)
        df = df[(df[cfg['target_column']] >= 2) & (df[cfg['target_column']] <= 200)]
        if len(df) <= 200: continue
        
        # --- FIX: Robust Adapt Sampling ---
        try:
            df['strat_bin'] = pd.cut(df[cfg['target_column']], bins=height_bins, labels=False)
            adapt, eval_df = train_test_split(df, train_size=100, stratify=df['strat_bin'], random_state=42)
        except:
            logger.warning(f"Stratified sampling failed for {city}; retrying with coarser bins.")
            coarse_bins = [0, 15, 30, 60, 999]
            df['strat_bin'] = pd.cut(df[cfg['target_column']], bins=coarse_bins, labels=False)
            try:
                adapt, eval_df = train_test_split(df, train_size=100, stratify=df['strat_bin'], random_state=42)
            except:
                adapt, eval_df = train_test_split(df, train_size=100, random_state=42)
            
        eval_meta = eval_df[['geometry', cfg['target_column']]].copy()
        eval_meta.rename(columns={cfg['target_column']: 'True_Height'}, inplace=True)
        
        X_eval = eval_df[feature_subset].astype(np.float32)
        dtest = xgb.DMatrix(X_eval, label=eval_meta['True_Height'])
        
        X_adapt = adapt[feature_subset].astype(np.float32)
        dtrain_adapt = xgb.DMatrix(X_adapt, label=adapt[cfg['target_column']].values)
        
        # 1. LOCO (RETUNE INDEPENDENTLY)
        sims = find_similar_cities(city, meta)
        top5 = sims[:5]
        X_pool, y_pool, w_pool = [], [], []
        
        for s in top5:
            xt, yt, _ = load_data_robust(s['file'], feature_subset, cfg['target_column'])
            if xt is not None:
                X_pool.append(xt); y_pool.append(yt)
                sc = s['score'] # Cosine Distance
                if pd.isna(sc) or sc < 0: sc = 2.0
                w_pool.append(np.full(len(yt), 1.0 / (sc + 0.05)))
        
        if not X_pool:
            logger.warning(f"No similarity matches for {city}. Skipping.")
            continue

        X_tr = pd.concat(X_pool, ignore_index=True)
        y_tr = pd.concat(y_pool, ignore_index=True)
        w_tr = np.concatenate(w_pool)
        w_tr = np.nan_to_num(w_tr, nan=1.0)
        w_tr = np.maximum(w_tr, 1e-6)
        
        if len(X_tr) > 100000:
            s_idx = X_tr.sample(100000, random_state=42).index
            X_opt = X_tr.loc[s_idx]
            y_opt = y_tr.loc[s_idx]
            w_opt = w_tr[s_idx]
        else:
            X_opt, y_opt, w_opt = X_tr, y_tr, w_tr
            
        logger.info(f"  Tuning LOCO for {city}...")
        best_loco = optimize_hyperparameters(X_opt, y_opt, w_opt, cfg, label_prefix="LOCO")
        n_trees_loco = best_loco.pop('n_estimators', 1000)
        
        dm = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        model_loco = xgb.train(best_loco, dm, num_boost_round=n_trees_loco, verbose_eval=False)
        del X_tr, y_tr, w_tr, dm
        
        p_loco = model_loco.predict(dtest)
        model_loco.save_model(MODEL_DIR / f"{batch_name}_{city}_loco.json")
        if len(X_eval) > 500:
            expl = shap.TreeExplainer(model_loco)
            shaps.append(expl(X_eval.sample(500, random_state=42)))
        
        # 2. 1-to-1 (RETUNE INDEPENDENTLY)
        src = sims[0]
        xt_1, yt_1, _ = load_data_robust(src['file'], feature_subset, cfg['target_column'])
        
        logger.info(f"  Tuning 1to1 for {city} (Source: {src['city']})...")
        
        n_tune = min(len(xt_1), 50000)
        idx_tune = xt_1.sample(n_tune, random_state=42).index
        best_1to1 = optimize_hyperparameters(
            xt_1.loc[idx_tune], 
            yt_1.loc[idx_tune], 
            None, 
            cfg, 
            label_prefix="1to1"
        )
        n_trees_1to1 = best_1to1.pop('n_estimators', 1000)
        
        model_1to1 = xgb.train(best_1to1, xgb.DMatrix(xt_1, label=yt_1), num_boost_round=n_trees_1to1, verbose_eval=False)
        p_1to1 = model_1to1.predict(dtest)
        model_1to1.save_model(MODEL_DIR / f"{batch_name}_{city}_1to1.json")
        del xt_1, yt_1
        
        # 3. Adapt (Start from 1to1, Own Budget)
        aparams = best_1to1.copy()
        adapt_rounds = cfg['search'].get('adapt_rounds', 100) 
        
        model_adapt = xgb.train(aparams, dtrain_adapt, num_boost_round=adapt_rounds, xgb_model=model_1to1, verbose_eval=False)
        p_adapt = model_adapt.predict(dtest)
        model_adapt.save_model(MODEL_DIR / f"{batch_name}_{city}_adapt.json")
        
        # Save Results
        eval_meta['Loco_Pred'] = p_loco; eval_meta['1to1_Pred'] = p_1to1; eval_meta['Adapt_Pred'] = p_adapt
        eval_meta['City'] = city
        
        output_gpkg = OUTPUT_DIR / f"{batch_name}_centroids_{city}.gpkg"
        pyogrio.write_dataframe(eval_meta, output_gpkg, driver="GPKG")
        preds.append(eval_meta.drop(columns='geometry'))
        
        m_l, m_1, m_a = calculate_metrics(eval_meta['True_Height'], p_loco), calculate_metrics(eval_meta['True_Height'], p_1to1), calculate_metrics(eval_meta['True_Height'], p_adapt)
        
        with open(OUTPUT_DIR / f"{batch_name}_stats_{city}.txt", "w") as f:
            f.write(f"City: {city}\nSource: {src['city']}\nLoco: {m_l}\n1to1: {m_1}\nAdapt: {m_a}\n")
            
        row = {"City": city, "Source": src['city']}
        for k, v in m_l.items(): row[f"Loco_{k}"] = v
        for k, v in m_1.items(): row[f"1to1_{k}"] = v
        for k, v in m_a.items(): row[f"Adapt_{k}"] = v
        res.append(row)
        gc.collect()

    pd.DataFrame(res).to_csv(OUTPUT_DIR / f"{batch_name}_metrics.csv", index=False)
    if preds:
        full = pd.concat(preds, ignore_index=True)
        g_l, g_1, g_a = calculate_metrics(full['True_Height'], full['Loco_Pred']), calculate_metrics(full['True_Height'], full['1to1_Pred']), calculate_metrics(full['True_Height'], full['Adapt_Pred'])
        with open(OUTPUT_DIR / f"{batch_name}_global_metrics.txt", "w") as f: f.write(f"Loco: {g_l}\n1to1: {g_1}\nAdapt: {g_a}\n")
        generate_global_plots(full, OUTPUT_DIR, batch_name)
    if shaps:
        s_vals = np.concatenate([s.values for s in shaps], axis=0)
        s_data = pd.concat([pd.DataFrame(s.data, columns=shaps[0].feature_names) for s in shaps], axis=0, ignore_index=True)
        generate_shap_beeswarms(s_vals, s_data, OUTPUT_DIR, batch_name)

# --- MAIN ---
def run_comprehensive_experiment():
    cfg = load_config()
    files = sorted(list(Path(cfg['input_dir']).glob("*.gpkg")))
    if not files: return
    
    # --- HARDCODED SIMILARITY COLUMNS (MORPH ONLY) ---
    SIM_COLS_MORPH = [
        'area', 'perimeter', 
        'compactness', 'rectangularity', 
        'n_100m_count', 'n_250m_count', 'n_500m_count', 'n_1000m_count'
    ]
    # --- RS SIMILARITY (Top 8 from Config) ---
    rs_list = cfg.get('remote_sensing_features', [])
    SIM_COLS_RS = rs_list[:8]

    # --- COMPUTE SIMILARITY MATRICES ---
    # 1. MORPH (Used for ALL and MORPH)
    meta_morph = calculate_similarity_matrix(files, cfg['target_column'], SIM_COLS_MORPH)
    
    # 2. RS (Used for RS only)
    meta_rs = calculate_similarity_matrix(files, cfg['target_column'], SIM_COLS_RS)

    # --- EXPERIMENTS ---
    # 1. ALL (Uses MORPH Sim)
    t50_all, _ = get_global_top_50_features(files, cfg, label="ALL")
    run_experiment_batch("ALL", t50_all, files, meta_morph, SIM_COLS_MORPH, cfg)
    
    # 2. MORPH (Uses MORPH Sim)
    morph_features = cfg.get('morphological_features', [])
    t50_morph, _ = get_global_top_50_features(files, cfg, allowed_features=set(morph_features), label="MORPH")
    run_experiment_batch("MORPH", t50_morph, files, meta_morph, SIM_COLS_MORPH, cfg)
    
    # 3. RS (Uses RS Sim)
    t50_rs, _ = get_global_top_50_features(files, cfg, allowed_features=set(rs_list), label="RS")
    run_experiment_batch("RS", t50_rs, files, meta_rs, SIM_COLS_RS, cfg)

if __name__ == "__main__":
    run_comprehensive_experiment()