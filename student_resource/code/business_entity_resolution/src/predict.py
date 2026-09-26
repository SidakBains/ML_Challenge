#!/usr/bin/env python3
"""
Test Inference for Business Entity Resolution
Generates matching_results.tsv and candidate_pairs.tsv for submission
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import generate_candidates_optimized
from features import build_feature_dataset

def load_model_artifacts(model_dir='models'):
    """Load trained model and thresholds"""
    model = lgb.Booster(model_file=f'{model_dir}/lgb_model.txt')
    global_thresh = joblib.load(f'{model_dir}/global_threshold.pkl')
    country_thresholds = joblib.load(f'{model_dir}/country_thresholds.pkl')
    feature_cols = joblib.load(f'{model_dir}/feature_cols.pkl')
    return model, global_thresh, country_thresholds, feature_cols


def run_inference(test_dir='dataset/test', model_dir='models', output_dir='output', max_candidates=100):
    """
    Full inference pipeline on test set
    """
    print("Loading test data...")
    test_s1 = pd.read_csv(f"{test_dir}/test_source1.tsv", sep="\t", dtype=str)
    test_s2 = pd.read_csv(f"{test_dir}/test_source2.tsv", sep="\t", dtype=str)
    test_s3 = pd.read_csv(f"{test_dir}/test_source3.tsv", sep="\t", dtype=str)
    
    print(f"Test S1: {len(test_s1)}, S2: {len(test_s2)}, S3: {len(test_s3)}")
    print(f"Test countries: {test_s1['country'].value_counts().to_dict()}")
    
    # Load model
    print("Loading model...")
    model, global_thresh, country_thresholds, feature_cols = load_model_artifacts(model_dir)
    print(f"Global threshold: {global_thresh:.3f}")
    print(f"Country thresholds: {country_thresholds}")
    
    # Generate candidates (same as training)
    print("Generating candidates...")
    candidates_df = generate_candidates_optimized(test_s1, test_s2, test_s3, max_candidates_per_s1=max_candidates)
    
    # Save candidate_pairs.tsv (required for submission)
    os.makedirs(output_dir, exist_ok=True)
    candidates_df.to_csv(f"{output_dir}/candidate_pairs.tsv", sep="\t", index=False)
    print(f"Saved candidate_pairs.tsv to {output_dir}/")
    
    # Build feature matrix for test
    print("Building test features...")
    candidates_dict = {}
    for _, row in candidates_df.iterrows():
        cands = row['candidate_entity_ids'].split(',') if row['candidate_entity_ids'] else []
        candidates_dict[row['source1_entity_id']] = cands
    
    test_cand_df = pd.concat([test_s2, test_s3], ignore_index=True)
    X_test, pair_ids = build_feature_dataset(test_s1, test_cand_df, candidates_dict, is_train=False)
    
    # Predict
    print("Predicting...")
    test_features = X_test.drop(['source1_entity_id', 'candidate_entity_id'], axis=1, errors='ignore')
    # Ensure column order matches training
    test_features = test_features[feature_cols]
    
    proba = model.predict(test_features)
    
    # Apply thresholds per country
    print("Applying thresholds...")
    # Get country for each pair
    s1_lookup = test_s1.set_index('entity_id')
    pair_countries = [s1_lookup.loc[p[0], 'country'] if p[0] in s1_lookup.index else 'Unknown' for p in pair_ids]
    
    predictions = np.zeros_like(proba)
    for country, thresh in country_thresholds.items():
        mask = np.array([c == country for c in pair_countries])
        predictions[mask] = (proba[mask] >= thresh).astype(int)
    
    # Fallback to global threshold
    known_countries = set(country_thresholds.keys())
    mask = np.array([c not in known_countries for c in pair_countries])
    if mask.any():
        predictions[mask] = (proba[mask] >= global_thresh).astype(int)
        print(f"  Applied global threshold to {mask.sum()} pairs with unknown country")
    
    # Build matching_results.tsv
    print("Building matching_results.tsv...")
    results = []
    for (s1_id, cand_id), pred in zip(pair_ids, predictions):
        if pred == 1:
            results.append({'source1_entity_id': s1_id, 'candidate_entity_id': cand_id})
    
    # Group by S1 entity
    if results:
        results_df = pd.DataFrame(results)
        matched = results_df.groupby('source1_entity_id')['candidate_entity_id'].apply(
            lambda x: ','.join(sorted(x))
        ).reset_index()
        matched.columns = ['source1_entity_id', 'matched_entity_ids']
    else:
        matched = pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids'])
    
    # Ensure ALL S1 entities have a row (empty if no matches)
    all_s1 = pd.DataFrame({'source1_entity_id': test_s1['entity_id'].values})
    final_matches = all_s1.merge(matched, on='source1_entity_id', how='left')
    final_matches['matched_entity_ids'] = final_matches['matched_entity_ids'].fillna('')
    
    # Save matching_results.tsv
    final_matches.to_csv(f"{output_dir}/matching_results.tsv", sep="\t", index=False)
    print(f"Saved matching_results.tsv to {output_dir}/")
    print(f"Total S1 entities: {len(final_matches)}")
    print(f"Entities with matches: {(final_matches['matched_entity_ids'] != '').sum()}")
    print(f"Entities without matches (singletons): {(final_matches['matched_entity_ids'] == '').sum()}")
    
    # Stats
    match_counts = final_matches['matched_entity_ids'].apply(
        lambda x: len(x.split(',')) if x else 0
    )
    print(f"Avg matches per S1: {match_counts.mean():.2f}")
    print(f"Max matches per S1: {match_counts.max()}")
    
    return final_matches, candidates_df


def validate_outputs(output_dir='output', test_dir='dataset/test'):
    """Run the official validator"""
    import subprocess
    result = subprocess.run([
        sys.executable, 'utils/validate_submission.py',
        '--matching', f'{output_dir}/matching_results.tsv',
        '--candidate', f'{output_dir}/candidate_pairs.tsv',
        '--test-dir', test_dir,
        '--check-ids'
    ], capture_output=True, text=True, cwd='student_resource')
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', default='dataset/test')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--max-candidates', type=int, default=100)
    parser.add_argument('--validate', action='store_true', help='Run validator after generation')
    args = parser.parse_args()
    
    run_inference(args.test_dir, args.model_dir, args.output_dir, args.max_candidates)
    
    if args.validate:
        print("\nRunning validator...")
        validate_outputs(args.output_dir, args.test_dir)