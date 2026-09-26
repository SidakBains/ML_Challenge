#!/usr/bin/env python3
"""
Check blocking recall against ground truth
"""

import pandas as pd
import sys

sys.path.insert(0, '..')
from blocking import generate_candidates_optimized

print("Loading data...")
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str, nrows=2000)
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str, nrows=20000)
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str, nrows=20000)
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str)

# Filter GT to only S1 entities in our sample
s1_ids = set(s1['entity_id'].values)
gt_sample = gt[gt['source1_entity_id'].isin(s1_ids)].copy()
print(f"GT sample: {len(gt_sample)} S1 entities")

# Generate candidates
print("Generating candidates...")
candidates_df = generate_candidates_optimized(s1, s2, s3, max_candidates_per_s1=100)

# Build candidate lookup
cand_lookup = {}
for _, row in candidates_df.iterrows():
    cands = set(row['candidate_entity_ids'].split(',')) if row['candidate_entity_ids'] else set()
    cand_lookup[row['source1_entity_id']] = cands

# Check recall
total_gt_matches = 0
found_matches = 0
missing_by_s1 = []

for _, row in gt_sample.iterrows():
    s1_id = row['source1_entity_id']
    if pd.isna(row['matched_entity_ids']) or row['matched_entity_ids'] == '':
        continue
    gt_matches = set(row['matched_entity_ids'].split(','))
    total_gt_matches += len(gt_matches)
    
    candidates = cand_lookup.get(s1_id, set())
    found = gt_matches & candidates
    found_matches += len(found)
    
    if len(found) < len(gt_matches):
        missing = gt_matches - candidates
        missing_by_s1.append((s1_id, missing))

recall = found_matches / total_gt_matches if total_gt_matches > 0 else 0
print(f"\n=== BLOCKING RECALL ===")
print(f"Total GT matches: {total_gt_matches}")
print(f"Found in candidates: {found_matches}")
print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")

if missing_by_s1:
    print(f"\nS1 entities with missing matches: {len(missing_by_s1)}")
    for s1_id, missing in missing_by_s1[:10]:
        print(f"  {s1_id}: {len(missing)} missing, e.g., {list(missing)[:3]}")

# Check candidate stats
cand_counts = [len(c) for c in cand_lookup.values()]
print(f"\n=== CANDIDATE STATS ===")
print(f"Avg candidates per S1: {sum(cand_counts)/len(cand_counts):.1f}")
print(f"Median: {sorted(cand_counts)[len(cand_counts)//2]}")
print(f"Max: {max(cand_counts)}")
print(f"Min: {min(cand_counts)}")