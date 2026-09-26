#!/usr/bin/env python3
"""
EDA on matched pairs using full entity lookup
"""

import pandas as pd
import numpy as np
import re
import sys

print("Loading full ground truth...", file=sys.stderr)
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str)
print(f"GT: {len(gt):,}", file=sys.stderr)

# Sample GT for analysis
sample_gt = gt[gt['matched_entity_ids'].notna() & (gt['matched_entity_ids'] != '')].sample(5000, random_state=42)

# Load ONLY the entities referenced in sampled GT
s1_ids_needed = set(sample_gt['source1_entity_id'].values)
all_matched = []
for ids in sample_gt['matched_entity_ids']:
    all_matched.extend(str(ids).split(','))
s2_ids_needed = {m for m in all_matched if m.startswith('S2-')}
s3_ids_needed = {m for m in all_matched if m.startswith('S3-')}

print(f"Need {len(s1_ids_needed)} S1, {len(s2_ids_needed)} S2, {len(s3_ids_needed)} S3", file=sys.stderr)

# Load full source files but filter
print("Loading source files...", file=sys.stderr)
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str)
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str)
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str)

# Filter to needed IDs
s1_sub = s1[s1['entity_id'].isin(s1_ids_needed)].copy()
s2_sub = s2[s2['entity_id'].isin(s2_ids_needed)].copy()
s3_sub = s3[s3['entity_id'].isin(s3_ids_needed)].copy()

entities = pd.concat([s1_sub.set_index('entity_id'), s2_sub.set_index('entity_id'), s3_sub.set_index('entity_id')])
print(f"Loaded {len(entities)} entities", file=sys.stderr)

def tokenize(text):
    if pd.isna(text) or text == '':
        return set()
    return set(re.findall(r'[a-zA-Z0-9]+', str(text).lower()))

def jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

# Analyze matched pairs
name_jaccards = []
addr_jaccards = []
name_lev = []
addr_lev = []
country_match = []

from rapidfuzz.distance import Levenshtein

matched_pairs = 0
for _, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    for m_id in row['matched_entity_ids'].split(','):
        if s1_id in entities.index and m_id in entities.index:
            e1 = entities.loc[s1_id]
            e2 = entities.loc[m_id]
            
            name1, name2 = e1['business_name'], e2['business_name']
            addr1, addr2 = e1['business_address'], e2['business_address']
            
            name_jaccards.append(jaccard(tokenize(name1), tokenize(name2)))
            addr_jaccards.append(jaccard(tokenize(addr1), tokenize(addr2)))
            
            # Normalized Levenshtein
            name_lev.append(1 - Levenshtein.normalized_distance(str(name1), str(name2)))
            addr_lev.append(1 - Levenshtein.normalized_distance(str(addr1), str(addr2)))
            
            country_match.append(e1['country'] == e2['country'])
            matched_pairs += 1

print(f"\n=== MATCHED PAIRS ANALYSIS ({matched_pairs} pairs) ===")
print(f"Name Jaccard: mean={np.mean(name_jaccards):.3f}, median={np.median(name_jaccards):.3f}, std={np.std(name_jaccards):.3f}")
print(f"Addr Jaccard: mean={np.mean(addr_jaccards):.3f}, median={np.median(addr_jaccards):.3f}, std={np.std(addr_jaccards):.3f}")
print(f"Name Levenshtein: mean={np.mean(name_lev):.3f}, median={np.median(name_lev):.3f}")
print(f"Addr Levenshtein: mean={np.mean(addr_lev):.3f}, median={np.median(addr_lev):.3f}")
print(f"Same country: {np.mean(country_match):.3f}")

# Percentiles
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  Name Jaccard P{p}: {np.percentile(name_jaccards, p):.3f}")

# Negative sampling from same country
print("\n=== NEGATIVE PAIRS (same country, sampled) ===")
neg_name_jac = []
neg_addr_jac = []
neg_name_lev = []
neg_addr_lev = []

s1_by_country = {c: g['entity_id'].values for c, g in s1[s1['entity_id'].isin(s1_ids_needed)].groupby('country')}
s2_by_country = {c: g['entity_id'].values for c, g in s2[s2['entity_id'].isin(s2_ids_needed)].groupby('country')}
s3_by_country = {c: g['entity_id'].values for c, g in s3[s3['entity_id'].isin(s3_ids_needed)].groupby('country')}

for _ in range(2000):
    c = np.random.choice(['US', 'India'])
    if len(s1_by_country[c]) == 0: continue
    s1_id = np.random.choice(s1_by_country[c])
    
    # Pick S2 or S3
    if np.random.rand() < 0.5 and len(s2_by_country[c]) > 0:
        cand_id = np.random.choice(s2_by_country[c])
    elif len(s3_by_country[c]) > 0:
        cand_id = np.random.choice(s3_by_country[c])
    else:
        continue
    
    if s1_id in entities.index and cand_id in entities.index:
        e1 = entities.loc[s1_id]
        e2 = entities.loc[cand_id]
        
        name1, name2 = e1['business_name'], e2['business_name']
        addr1, addr2 = e1['business_address'], e2['business_address']
        
        neg_name_jac.append(jaccard(tokenize(name1), tokenize(name2)))
        neg_addr_jac.append(jaccard(tokenize(addr1), tokenize(addr2)))
        neg_name_lev.append(1 - Levenshtein.normalized_distance(str(name1), str(name2)))
        neg_addr_lev.append(1 - Levenshtein.normalized_distance(str(addr1), str(addr2)))

print(f"Name Jaccard: mean={np.mean(neg_name_jac):.3f}, median={np.median(neg_name_jac):.3f}")
print(f"Addr Jaccard: mean={np.mean(neg_addr_jac):.3f}, median={np.median(neg_addr_jac):.3f}")
print(f"Name Levenshtein: mean={np.mean(neg_name_lev):.3f}, median={np.median(neg_name_lev):.3f}")
print(f"Addr Levenshtein: mean={np.mean(neg_addr_lev):.3f}, median={np.median(neg_addr_lev):.3f}")

# Check abbreviation patterns
print("\n=== COMMON ABBREVIATIONS IN NAMES ===")
all_names = pd.concat([s1['business_name'], s2['business_name'], s3['business_name']]).dropna()
abbrevs = ['corp', 'inc', 'ltd', 'llc', 'pvt', 'pvt.', 'ltd.', 'inc.', 'corp.', 'co', 'co.', 'company', 'corporation', 'limited', 'private']
for ab in abbrevs:
    count = all_names.str.lower().str.contains(rf'\b{re.escape(ab)}\b').sum()
    if count > 0:
        print(f"  {ab}: {count:,}")

print("\n=== EDA COMPLETE ===", file=sys.stderr)