#!/usr/bin/env python3
"""
Training script for Business Entity Resolution
LightGBM with macro F_0.5 optimization
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import joblib
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_feature_dataset
from blocking import generate_candidates_optimized

# ==================== MACRO F_0.5 METRIC ====================

def macro_f05_score(y_true, y_pred, groups):
    """
    Compute macro F_0.5 score grouped by source1_entity_id
    F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'group': groups
    })
    
    f05_scores = []
    for g_id, g_df in df.groupby('group'):
        tp = ((g_df['y_true'] == 1) & (g_df['y_pred'] == 1)).sum()
        fp = ((g_df['y_true'] == 0) & (g_df['y_pred'] == 1)).sum()
        fn = ((g_df['y_true'] == 1) & (g_df['y_pred'] == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        else:
            f05 = 0.0
        f05_scores.append(f05)
    
    return np.mean(f05_scores)


def f05_lgb_metric(y_pred, dataset):
    """LightGBM custom metric for F_0.5 (approximate, per-group)"""
    # This is a proxy - actual macro F_0.5 needs group info
    y_true = dataset.get_label()
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    tp = ((y_true == 1) & (y_pred_binary == 1)).sum()
    fp = ((y_true == 0) & (y_pred_binary == 1)).sum()
    fn = ((y_true == 1) & (y_pred_binary == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall > 0:
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    else:
        f05 = 0.0
    
    return 'f05', f05, True


# ==================== THRESHOLD OPTIMIZATION ====================

def optimize_threshold_macro_f05(y_true, y_pred_proba, groups):
    """
    Find optimal threshold for macro F_0.5
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_thresh = 0.5
    best_score = 0.0
    
    df = pd.DataFrame({
        'y_true': y_true,
        'y_prob': y_pred_proba,
        'group': groups
    })
    
    for thresh in thresholds:
        y_pred = (df['y_prob'] >= thresh).astype(int)
        score = macro_f05_score(df['y_true'], y_pred, df['group'])
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh, best_score


def optimize_threshold_per_country(y_true, y_pred_proba, groups, countries):
    """
    Find optimal threshold per country for macro F_0.5
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_prob': y_pred_proba,
        'group': groups,
        'country': countries
    })
    
    thresholds = {}
    for country in df['country'].unique():
        c_df = df[df['country'] == country]
        best_thresh = 0.5
        best_score = 0.0
        for thresh in np.arange(0.05, 0.95, 0.01):
            y_pred = (c_df['y_prob'] >= thresh).astype(int)
            score = macro_f05_score(c_df['y_true'], y_pred, c_df['group'])
            if score > best_score:
                best_score = score
                best_thresh = thresh
        thresholds[country] = best_thresh
        print(f"  {country}: threshold={best_thresh:.3f}, macro_F05={best_score:.4f}")
    
    return thresholds


# ==================== TRAINING PIPELINE ====================

def train_model(X_train, y_train, groups_train, countries_train, 
                X_val, y_val, groups_val, countries_val,
                params=None):
    """
    Train LightGBM with macro F_0.5 optimization
    """
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 50,
            'min_child_weight': 1e-3,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'verbosity': -1,
            'random_state': 42,
            'n_jobs': -1,
        }
    
    # Create datasets
    train_data = lgb.Dataset(X_train.drop(['source1_entity_id', 'candidate_entity_id'], axis=1, errors='ignore'), 
                              label=y_train, group=groups_train)
    val_data = lgb.Dataset(X_val.drop(['source1_entity_id', 'candidate_entity_id'], axis=1, errors='ignore'), 
                            label=y_val, group=groups_val, reference=train_data)
    
    print("Training LightGBM...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(50),
        ],
    )
    
    # Predict on validation
    val_proba = model.predict(val_data.data)
    
    # Optimize global threshold
    print("\nOptimizing global threshold...")
    best_thresh, best_f05 = optimize_threshold_macro_f05(y_val, val_proba, groups_val)
    print(f"Best global threshold: {best_thresh:.3f}, Macro F_0.5: {best_f05:.4f}")
    
    # Optimize per-country thresholds
    print("\nOptimizing per-country thresholds...")
    country_thresholds = optimize_threshold_per_country(y_val, val_proba, groups_val, countries_val)
    
    return model, best_thresh, country_thresholds


def prepare_data(data_dir='dataset/train', sample_frac=1.0):
    """
    Load data, generate candidates, build features, create train/val split
    """
    print("Loading data...")
    s1 = pd.read_csv(f"{data_dir}/train_source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(f"{data_dir}/train_source2.tsv", sep="\t", dtype=str)
    s3 = pd.read_csv(f"{data_dir}/train_source3.tsv", sep="\t", dtype=str)
    gt = pd.read_csv(f"{data_dir}/train_ground_truth.tsv", sep="\t", dtype=str)
    
    if sample_frac < 1.0:
        s1 = s1.sample(frac=sample_frac, random_state=42)
    
    # Build labels dict
    labels_dict = {}
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        if pd.notna(row['matched_entity_ids']) and row['matched_entity_ids'] != '':
            labels_dict[s1_id] = set(row['matched_entity_ids'].split(','))
        else:
            labels_dict[s1_id] = set()
    
    # Generate candidates
    print("Generating candidates...")
    candidates_df = generate_candidates_optimized(s1, s2, s3, max_candidates_per_s1=100)
    
    candidates_dict = {}
    for _, row in candidates_df.iterrows():
        cands = row['candidate_entity_ids'].split(',') if row['candidate_entity_ids'] else []
        candidates_dict[row['source1_entity_id']] = cands
    
    cand_df = pd.concat([s2, s3], ignore_index=True)
    
    # Build features
    print("Building features...")
    X, y, pairs = build_feature_dataset(s1, cand_df, candidates_dict, labels_dict, is_train=True)
    
    # Extract group and country info for splitting
    s1_lookup = s1.set_index('entity_id')
    groups = np.array([p[0] for p in pairs])
    countries = np.array([s1_lookup.loc[p[0], 'country'] if p[0] in s1_lookup.index else 'Unknown' for p in pairs])
    
    return X, y, groups, countries, candidates_dict, cand_df, s1


def train_val_split(X, y, groups, countries, val_frac=0.2):
    """
    GroupKFold split by source1_entity_id, stratified by country
    """
    gkf = GroupKFold(n_splits=int(1/val_frac))
    train_idx, val_idx = list(gkf.split(X, y, groups))[0]
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    groups_train, groups_val = groups[train_idx], groups[val_idx]
    countries_train, countries_val = countries[train_idx], countries[val_idx]
    
    print(f"Train: {len(X_train)} pairs, {len(np.unique(groups_train))} S1 entities")
    print(f"Val: {len(X_val)} pairs, {len(np.unique(groups_val))} S1 entities")
    print(f"Train pos rate: {y_train.mean():.4f}, Val pos rate: {y_val.mean():.4f}")
    print(f"Train countries: {pd.Series(countries_train).value_counts().to_dict()}")
    print(f"Val countries: {pd.Series(countries_val).value_counts().to_dict()}")
    
    return X_train, X_val, y_train, y_val, groups_train, groups_val, countries_train, countries_val


def save_model_artifacts(model, global_thresh, country_thresholds, feature_cols, output_dir='models'):
    """Save model and thresholds"""
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_model(f'{output_dir}/lgb_model.txt')
    joblib.dump(global_thresh, f'{output_dir}/global_threshold.pkl')
    joblib.dump(country_thresholds, f'{output_dir}/country_thresholds.pkl')
    joblib.dump(feature_cols, f'{output_dir}/feature_cols.pkl')
    
    with open(f'{output_dir}/thresholds.json', 'w') as f:
        json.dump({
            'global': float(global_thresh),
            'per_country': {k: float(v) for k, v in country_thresholds.items()}
        }, f, indent=2)
    
    print(f"Model artifacts saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=float, default=1.0, help='Fraction of S1 to use (for quick testing)')
    parser.add_argument('--data-dir', default='dataset/train')
    parser.add_argument('--output-dir', default='models')
    args = parser.parse_args()
    
    # Prepare data
    X, y, groups, countries, candidates_dict, cand_df, s1 = prepare_data(args.data_dir, args.sample)
    
    # Split
    X_train, X_val, y_train, y_val, groups_train, groups_val, countries_train, countries_val = \
        train_val_split(X, y, groups, countries, val_frac=0.2)
    
    # Train
    model, global_thresh, country_thresholds = train_model(
        X_train, y_train, groups_train, countries_train,
        X_val, y_val, groups_val, countries_val
    )
    
    # Save
    feature_cols = [c for c in X.columns if c not in ['source1_entity_id', 'candidate_entity_id']]
    save_model_artifacts(model, global_thresh, country_thresholds, feature_cols, args.output_dir)
    
    # Final validation score
    val_proba = model.predict(X_val.drop(['source1_entity_id', 'candidate_entity_id'], axis=1, errors='ignore'))
    
    # Apply per-country thresholds
    val_pred = np.zeros_like(val_proba)
    for country, thresh in country_thresholds.items():
        mask = countries_val == country
        val_pred[mask] = (val_proba[mask] >= thresh).astype(int)
    
    # Fallback to global for any missing
    mask = np.isin(countries_val, list(country_thresholds.keys()), invert=True)
    if mask.any():
        val_pred[mask] = (val_proba[mask] >= global_thresh).astype(int)
    
    final_f05 = macro_f05_score(y_val, val_pred, groups_val)
    print(f"\n=== FINAL VALIDATION MACRO F_0.5: {final_f05:.4f} ===")