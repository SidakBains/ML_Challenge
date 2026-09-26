#!/usr/bin/env python3
"""
EDA for Amazon ML Challenge 2026 - Business Entity Resolution
Run from student_resource/ directory
"""

import pandas as pd
import numpy as np
from collections import Counter
import re

# Load data (sample for speed, then full for key stats)
print("Loading training data...")
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str)
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str)
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str)
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str)

print(f"S1: {len(s1):,} | S2: {len(s2):,} | S3: {len(s3):,} | GT: {len(gt):,}")

# 1. Country distribution
print("\n=== COUNTRY DISTRIBUTION ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    print(f"{name}: {df['country'].value_counts().to_dict()}")

# 2. Ground truth analysis
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
print(f"Match cardinality: {gt['num_matches'].value_counts().sort_index().to_dict()}")
print(f"Singleton rate: {(gt['num_matches'] == 0).mean():.3f}")
print(f"Has S2 match: {gt['has_s2'].mean():.3f}")
print(f"Has S3 match: {gt['has_s3'].mean():.3f}")
print(f"Has both S2&S3: {(gt['has_s2'] & gt['has_s3']).mean():.3f}")

# 3. Field completeness
print("\n=== FIELD COMPLETENESS ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    for col in ['business_name', 'business_address']:
        missing = df[col].isna().sum() + (df[col] == '').sum()
        print(f"{name} {col}: {missing:,} missing ({missing/len(df)*100:.1f}%)")

# 4. PIN code extraction pattern
def extract_pincodes(addr):
    if pd.isna(addr) or addr == '':
        return []
    # India: 6 digits, US: 5 digits (+4), France: 5 digits
    return re.findall(r'\b\d{5,6}\b', str(addr))

print("\n=== PIN CODE EXTRACTION ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    sample = df.sample(min(10000, len(df)), random_state=42)
    has_pin = sample['business_address'].apply(lambda x: len(extract_pincodes(x)) > 0).mean()
    print(f"{name}: {has_pin:.3f} have PIN codes")

# 5. Token analysis on matching pairs (sample)
print("\n=== TOKEN OVERLAP ON MATCHING PAIRS (sample) ===")
# Build entity lookup
entities = pd.concat([s1, s2, s3]).set_index('entity_id')

# Sample matched pairs
matched_pairs = []
for _, row in gt[gt['num_matches'] > 0].sample(min(1000, len(gt[gt['num_matches'] > 0])), random_state=42).iterrows():
    s1_id = row['source1_entity_id']
    for m_id in row['matched_entity_ids'].split(','):
        matched_pairs.append((s1_id, m_id))

def tokenize(text):
    if pd.isna(text) or text == '':
        return set()
    # Simple tokenization: alphanumeric tokens, lowercase
    return set(re.findall(r'[a-zA-Z0-9]+', str(text).lower()))

name_jaccards = []
addr_jaccards = []
for s1_id, m_id in matched_pairs[:500]:
    if s1_id in entities.index and m_id in entities.index:
        e1 = entities.loc[s1_id]
        e2 = entities.loc[m_id]
        name1, name2 = tokenize(e1['business_name']), tokenize(e2['business_name'])
        addr1, addr2 = tokenize(e1['business_address']), tokenize(e2['business_address'])
        if name1 or name2:
            name_jaccards.append(len(name1 & name2) / len(name1 | name2) if name1 | name2 else 0)
        if addr1 or addr2:
            addr_jaccards.append(len(addr1 & addr2) / len(addr1 | addr2) if addr1 | addr2 else 0)

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
    # Random S2 or S3 not in matches
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
test_s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", dtype=str)
print(f"Test S1: {len(test_s1):,} | Countries: {test_s1['country'].value_counts().to_dict()}")

print("\n=== EDA COMPLETE ===")