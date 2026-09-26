#!/usr/bin/env python3
"""
Fast EDA for Amazon ML Challenge 2026 - Business Entity Resolution
Uses sampling for speed
"""

import pandas as pd
import numpy as np
from collections import Counter
import re
import sys

print("Loading training data (sampled)...", file=sys.stderr)

# Load with sampling for speed
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str, nrows=50000)
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str, nrows=50000)
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str, nrows=50000)
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str)

print(f"S1 sample: {len(s1):,} | S2 sample: {len(s2):,} | S3 sample: {len(s3):,} | GT: {len(gt):,}", file=sys.stderr)

# 1. Country distribution (full GT, sample sources)
print("\n=== COUNTRY DISTRIBUTION ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    print(f"{name}: {df['country'].value_counts().to_dict()}")

# 2. Ground truth analysis (full)
print("\n=== GROUND TRUTH ANALYSIS ===")
gt['num_matches'] = gt['matched_entity_ids'].apply(
    lambda x: len(x.split(',')) if pd.notna(x) and x != '' else 0
)
gt['has_s2'] = gt['matched_entity_ids'].apply(
    lambda x: any(m.startswith('S2-') for m in x.split(',')) if pd.notna(x) and x != '' else False
)
gt['has_s3'] = gt['matched_entity_ids'].apply(
    lambda x: any(m.startswith('S3-') for m in x.split(',')) if pd.notna(x) and x != '' else False
)

print(f"S1 entities in GT: {len(gt):,}")
print(f"Match cardinality: {gt['num_matches'].value_counts().sort_index().head(20).to_dict()}")
print(f"Singleton rate: {(gt['num_matches'] == 0).mean():.3f}")
print(f"Has S2 match: {gt['has_s2'].mean():.3f}")
print(f"Has S3 match: {gt['has_s3'].mean():.3f}")
print(f"Has both S2&S3: {(gt['has_s2'] & gt['has_s3']).mean():.3f}")
print(f"Avg matches per S1: {gt['num_matches'].mean():.2f}")
print(f"Max matches per S1: {gt['num_matches'].max()}")

# 3. Field completeness (sample)
print("\n=== FIELD COMPLETENESS (sample) ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    for col in ['business_name', 'business_address']:
        missing = df[col].isna().sum() + (df[col] == '').sum()
        print(f"{name} {col}: {missing:,} missing ({missing/len(df)*100:.1f}%)")

# 4. PIN code extraction pattern
def extract_pincodes(addr):
    if pd.isna(addr) or addr == '':
        return []
    return re.findall(r'\b\d{5,6}\b', str(addr))

print("\n=== PIN CODE EXTRACTION (sample) ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    has_pin = df['business_address'].apply(lambda x: len(extract_pincodes(x)) > 0).mean()
    print(f"{name}: {has_pin:.3f} have PIN codes")

# 5. Token analysis on matching pairs (sample from GT)
print("\n=== TOKEN OVERLAP ON MATCHING PAIRS (sample) ===")
# Build entity lookup from samples
entities = pd.concat([s1, s2, s3]).set_index('entity_id')

# Sample matched pairs
matched_gt = gt[gt['num_matches'] > 0]
sample_gt = matched_gt.sample(min(1000, len(matched_gt)), random_state=42)

def tokenize(text):
    if pd.isna(text) or text == '':
        return set()
    return set(re.findall(r'[a-zA-Z0-9]+', str(text).lower()))

name_jaccards = []
addr_jaccards = []
matched_pairs_count = 0
for _, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    for m_id in row['matched_entity_ids'].split(','):
        if s1_id in entities.index and m_id in entities.index:
            e1 = entities.loc[s1_id]
            e2 = entities.loc[m_id]
            name1, name2 = tokenize(e1['business_name']), tokenize(e2['business_name'])
            addr1, addr2 = tokenize(e1['business_address']), tokenize(e2['business_address'])
            if name1 or name2:
                name_jaccards.append(len(name1 & name2) / len(name1 | name2) if name1 | name2 else 0)
            if addr1 or addr2:
                addr_jaccards.append(len(addr1 & addr2) / len(addr1 | addr2) if addr1 | addr2 else 0)
            matched_pairs_count += 1

print(f"Analyzed {matched_pairs_count} matched pairs")
print(f"Name Jaccard (matched): mean={np.mean(name_jaccards):.3f}, median={np.median(name_jaccards):.3f}")
print(f"Addr Jaccard (matched): mean={np.mean(addr_jaccards):.3f}, median={np.median(addr_jaccards):.3f}")

# 6. Negative pairs token overlap (sample)
print("\n=== TOKEN OVERLAP ON NEGATIVE PAIRS (sample) ===")
neg_name_jaccards = []
neg_addr_jaccards = []
s1_ids = s1['entity_id'].values
s2_ids = s2['entity_id'].values
s3_ids = s3['entity_id'].values

for _ in range(500):
    s1_id = np.random.choice(s1_ids)
    cand_id = np.random.choice(np.concatenate([s2_ids, s3_ids]))
    if s1_id in entities.index and cand_id in entities.index:
        e1 = entities.loc[s1_id]
        e2 = entities.loc[cand_id]
        name1, name2 = tokenize(e1['business_name']), tokenize(e2['business_name'])
        addr1, addr2 = tokenize(e1['business_address']), tokenize(e2['business_address'])
        if name1 or name2:
            neg_name_jaccards.append(len(name1 & name2) / len(name1 | name2) if name1 | name2 else 0)
        if addr1 or addr2:
            neg_addr_jaccards.append(len(addr1 & addr2) / len(addr1 | addr2) if addr1 | addr2 else 0)

print(f"Name Jaccard (random): mean={np.mean(neg_name_jaccards):.3f}, median={np.median(neg_name_jaccards):.3f}")
print(f"Addr Jaccard (random): mean={np.mean(neg_addr_jaccards):.3f}, median={np.median(neg_addr_jaccards):.3f}")

# 7. Test set preview
print("\n=== TEST SET PREVIEW ===")
test_s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", dtype=str, nrows=10000)
print(f"Test S1 sample: {len(test_s1):,} | Countries: {test_s1['country'].value_counts().to_dict()}")

# 8. Name/address length stats
print("\n=== TEXT LENGTH STATS (sample) ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    name_len = df['business_name'].fillna('').str.len()
    addr_len = df['business_address'].fillna('').str.len()
    print(f"{name} name: mean={name_len.mean():.1f}, median={name_len.median():.1f}")
    print(f"{name} addr: mean={addr_len.mean():.1f}, median={addr_len.median():.1f}")

print("\n=== EDA COMPLETE ===", file=sys.stderr)