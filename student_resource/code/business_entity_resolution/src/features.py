#!/usr/bin/env python3
"""
Feature Engineering for Business Entity Resolution
Rich pairwise features for name, address, cross-field matching
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from rapidfuzz import fuzz, distance
import jellyfish

# Reuse normalization from blocking
def remove_accents(text):
    if pd.isna(text) or text == '':
        return ''
    text = str(text)
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

INDIC_TRANSLITERATION_MAP = {
    'aa': 'a', 'ae': 'e', 'ai': 'e', 'au': 'o', 'ou': 'u',
    'ee': 'i', 'ei': 'i', 'ii': 'i', 'oo': 'u', 'uu': 'u',
    'kh': 'k', 'gh': 'g', 'ch': 'c', 'jh': 'j', 'th': 't', 'dh': 'd',
    'ph': 'p', 'bh': 'b', 'sh': 's', 'zh': 'z', 'ksh': 'k', 'tr': 't',
    'iya': 'ia', 'iyaan': 'ian', 'iyaar': 'iar', 'iwal': 'wal',
    'eshwar': 'eshwar', 'ishwar': 'ishwar', 'swamy': 'swami',
    'nagar': 'nagar', 'naggar': 'nagar', 'pura': 'pur', 'puram': 'puram',
    'vihar': 'vihar', 'vihaar': 'vihar', 'dham': 'dham', 'dhaam': 'dham',
    'ganj': 'ganj', 'gunj': 'ganj', 'bazar': 'bazar', 'bazaar': 'bazar',
    'mandir': 'mandir', 'mandira': 'mandir', 'masjid': 'masjid', 'maszid': 'masjid',
}

def normalize_indic_transliteration(text):
    if not text:
        return text
    text = text.lower()
    for variant, canonical in INDIC_TRANSLITERATION_MAP.items():
        text = re.sub(rf'\b{variant}\b', canonical, text)
    return text

def normalize_text(text, remove_diacritics=True, normalize_indic=True):
    if pd.isna(text) or text == '':
        return ''
    text = str(text).lower()
    if remove_diacritics:
        text = remove_accents(text)
    if normalize_indic:
        text = normalize_indic_transliteration(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    if pd.isna(text) or text == '':
        return set()
    return set(re.findall(r'[a-zA-Z0-9]+', str(text).lower()))

def jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0

def extract_pincode(addr, country=None):
    if pd.isna(addr) or addr == '':
        return None
    text = str(addr)
    if country == 'India':
        matches = re.findall(r'\b(\d{6})\b', text)
    elif country == 'France':
        matches = re.findall(r'\b(\d{5})\b', text)
    elif country == 'US':
        matches = re.findall(r'\b(\d{5})(?:-\d{4})?\b', text)
    else:
        matches = re.findall(r'\b(\d{5,6})\b', text)
    return matches[0] if matches else None

def get_city_state(addr, country=None):
    """Extract city/state tokens from address"""
    if pd.isna(addr) or addr == '':
        return set()
    norm = normalize_text(addr)
    # Common city/state indicators
    tokens = set(norm.split())
    # Filter to likely geographic tokens (longer, not common words)
    geo = {t for t in tokens if len(t) > 3 and t not in {
        'street', 'road', 'avenue', 'drive', 'lane', 'boulevard',
        'near', 'opposite', 'behind', 'beside', 'north', 'south',
        'east', 'west', 'unit', 'apt', 'floor', 'suite', 'block',
        'sector', 'phase', 'extension', 'colony', 'nagar', 'area'
    }}
    return geo

def compute_features(s1_row, cand_row):
    """
    Compute all pairwise features between S1 entity and candidate
    Returns dict of features
    """
    features = {}
    
    # Basic info
    s1_name = s1_row['business_name']
    s1_addr = s1_row['business_address']
    s1_country = s1_row['country']
    
    cand_name = cand_row['business_name']
    cand_addr = cand_row['business_address']
    cand_country = cand_row['country']
    cand_source = cand_row['entity_id'][:3]  # S2- or S3-
    
    # Normalized versions
    s1_name_n = normalize_text(s1_name)
    s1_addr_n = normalize_text(s1_addr)
    cand_name_n = normalize_text(cand_name)
    cand_addr_n = normalize_text(cand_addr)
    
    # Token sets
    s1_name_toks = tokenize(s1_name)
    s1_addr_toks = tokenize(s1_addr)
    cand_name_toks = tokenize(cand_name)
    cand_addr_toks = tokenize(cand_addr)
    
    # ==================== NAME FEATURES ====================
    # Jaccard similarities
    features['name_jaccard'] = jaccard(s1_name_toks, cand_name_toks)
    features['name_jaccard_norm'] = jaccard(tokenize(s1_name_n), tokenize(cand_name_n))
    
    # RapidFuzz string similarities
    features['name_ratio'] = fuzz.ratio(s1_name_n, cand_name_n) / 100.0
    features['name_partial_ratio'] = fuzz.partial_ratio(s1_name_n, cand_name_n) / 100.0
    features['name_token_sort_ratio'] = fuzz.token_sort_ratio(s1_name_n, cand_name_n) / 100.0
    features['name_token_set_ratio'] = fuzz.token_set_ratio(s1_name_n, cand_name_n) / 100.0
    features['name_wratio'] = fuzz.WRatio(s1_name_n, cand_name_n) / 100.0
    
    # Levenshtein distances
    features['name_levenshtein'] = 1.0 - distance.Levenshtein.normalized_distance(s1_name_n, cand_name_n)
    features['name_damerau'] = 1.0 - distance.DamerauLevenshtein.normalized_distance(s1_name_n, cand_name_n)
    
    # Jaro-Winkler
    features['name_jaro'] = distance.JaroWinkler.similarity(s1_name_n, cand_name_n)
    
    # Phonetic similarities
    try:
        s1_metaphone = jellyfish.metaphone(s1_name_n)[0] if s1_name_n else ''
        cand_metaphone = jellyfish.metaphone(cand_name_n)[0] if cand_name_n else ''
        features['name_metaphone_match'] = 1.0 if s1_metaphone and s1_metaphone == cand_metaphone else 0.0
    except:
        features['name_metaphone_match'] = 0.0
    
    try:
        features['name_soundex_match'] = 1.0 if jellyfish.soundex(s1_name_n) == jellyfish.soundex(cand_name_n) else 0.0
    except:
        features['name_soundex_match'] = 0.0
    
    # Token-level features
    s1_name_toks_list = list(s1_name_toks)
    cand_name_toks_list = list(cand_name_toks)
    features['name_first_token_match'] = 1.0 if (s1_name_toks_list and cand_name_toks_list and 
                                                   s1_name_toks_list[0] == cand_name_toks_list[0]) else 0.0
    features['name_token_overlap'] = len(s1_name_toks & cand_name_toks)
    features['name_token_overlap_ratio'] = features['name_token_overlap'] / max(1, len(s1_name_toks | cand_name_toks))
    
    # Legal suffix agreement
    LEGAL_SUFFIXES = {'inc', 'llc', 'ltd', 'corp', 'corporation', 'company', 'co', 'llp', 'lp', 
                      'pllc', 'plc', 'pvt', 'private', 'limited', 'incorporated', 'sarl', 'sas', 
                      'sasu', 'eurl', 'sa', 'snc', 'gmbh', 'ag', 'kg', 'ohg', 'kg'}
    s1_suffixes = {t for t in s1_name_toks if t in LEGAL_SUFFIXES}
    cand_suffixes = {t for t in cand_name_toks if t in LEGAL_SUFFIXES}
    features['legal_suffix_match'] = 1.0 if s1_suffixes == cand_suffixes and s1_suffixes else 0.0
    features['legal_suffix_overlap'] = len(s1_suffixes & cand_suffixes)
    
    # ==================== ADDRESS FEATURES ====================
    # Jaccard similarities
    features['addr_jaccard'] = jaccard(s1_addr_toks, cand_addr_toks)
    features['addr_jaccard_norm'] = jaccard(tokenize(s1_addr_n), tokenize(cand_addr_n))
    
    # RapidFuzz
    features['addr_ratio'] = fuzz.ratio(s1_addr_n, cand_addr_n) / 100.0
    features['addr_partial_ratio'] = fuzz.partial_ratio(s1_addr_n, cand_addr_n) / 100.0
    features['addr_token_sort_ratio'] = fuzz.token_sort_ratio(s1_addr_n, cand_addr_n) / 100.0
    features['addr_token_set_ratio'] = fuzz.token_set_ratio(s1_addr_n, cand_addr_n) / 100.0
    
    # Levenshtein
    features['addr_levenshtein'] = 1.0 - distance.Levenshtein.normalized_distance(s1_addr_n, cand_addr_n)
    
    # PIN code match
    s1_pin = extract_pincode(s1_addr, s1_country)
    cand_pin = extract_pincode(cand_addr, cand_country)
    features['pincode_match'] = 1.0 if (s1_pin and cand_pin and s1_pin == cand_pin) else 0.0
    features['pincode_present'] = 1.0 if (s1_pin or cand_pin) else 0.0
    
    # City/State match
    s1_city = get_city_state(s1_addr, s1_country)
    cand_city = get_city_state(cand_addr, cand_country)
    features['city_overlap'] = len(s1_city & cand_city)
    features['city_jaccard'] = jaccard(s1_city, cand_city)
    
    # Numeric token match (street numbers, unit numbers)
    s1_nums = {t for t in s1_addr_toks if t.isdigit()}
    cand_nums = {t for t in cand_addr_toks if t.isdigit()}
    features['numeric_token_match'] = 1.0 if (s1_nums & cand_nums) else 0.0
    features['numeric_overlap_count'] = len(s1_nums & cand_nums)
    
    # ==================== CROSS-FIELD FEATURES ====================
    # Name tokens in address and vice versa
    features['name_tokens_in_cand_addr'] = len(s1_name_toks & cand_addr_toks)
    features['cand_name_tokens_in_s1_addr'] = len(cand_name_toks & s1_addr_toks)
    
    # ==================== COUNTRY FEATURES ====================
    features['same_country'] = 1.0 if s1_country == cand_country else 0.0
    features['cand_is_s2'] = 1.0 if cand_source == 'S2-' else 0.0
    features['cand_is_s3'] = 1.0 if cand_source == 'S3-' else 0.0
    
    # Country-specific features
    features['both_india'] = 1.0 if (s1_country == 'India' and cand_country == 'India') else 0.0
    features['both_us'] = 1.0 if (s1_country == 'US' and cand_country == 'US') else 0.0
    features['both_france'] = 1.0 if (s1_country == 'France' and cand_country == 'France') else 0.0
    
    # ==================== META FEATURES ====================
    # Length features
    features['s1_name_len'] = len(s1_name_n)
    features['cand_name_len'] = len(cand_name_n)
    features['name_len_diff'] = abs(len(s1_name_n) - len(cand_name_n))
    features['name_len_ratio'] = min(len(s1_name_n), len(cand_name_n)) / max(1, max(len(s1_name_n), len(cand_name_n)))
    
    features['s1_addr_len'] = len(s1_addr_n)
    features['cand_addr_len'] = len(cand_addr_n)
    features['addr_len_diff'] = abs(len(s1_addr_n) - len(cand_addr_n))
    features['addr_len_ratio'] = min(len(s1_addr_n), len(cand_addr_n)) / max(1, max(len(s1_addr_n), len(cand_addr_n)))
    
    # Token counts
    features['s1_name_token_count'] = len(s1_name_toks)
    features['cand_name_token_count'] = len(cand_name_toks)
    features['s1_addr_token_count'] = len(s1_addr_toks)
    features['cand_addr_token_count'] = len(cand_addr_toks)
    
    return features


def build_feature_dataset(s1_df, cand_df, candidates_dict, labels_dict=None, is_train=True):
    """
    Build feature matrix from candidate pairs
    
    Args:
        s1_df: Source 1 dataframe
        cand_df: Combined S2+S3 dataframe
        candidates_dict: {s1_id: [cand_ids]} from blocking
        labels_dict: {s1_id: set(matched_ids)} from ground truth (for training)
        is_train: Whether to include labels
    
    Returns:
        X: Feature DataFrame
        y: Labels array (if is_train)
        pair_ids: List of (s1_id, cand_id) tuples
    """
    # Build candidate lookup for fast access
    cand_lookup = cand_df.set_index('entity_id')
    s1_lookup = s1_df.set_index('entity_id')
    
    rows = []
    labels = []
    pair_ids = []
    
    print(f"Building features for {len(candidates_dict)} S1 entities...")
    
    for s1_id, cand_ids in candidates_dict.items():
        if s1_id not in s1_lookup.index:
            continue
        s1_row = s1_lookup.loc[s1_id]
        
        for cand_id in cand_ids:
            if cand_id not in cand_lookup.index:
                continue
            cand_row = cand_lookup.loc[cand_id]
            
            feats = compute_features(s1_row, cand_row)
            feats['source1_entity_id'] = s1_id
            feats['candidate_entity_id'] = cand_id
            rows.append(feats)
            pair_ids.append((s1_id, cand_id))
            
            if is_train and labels_dict:
                label = 1 if cand_id in labels_dict.get(s1_id, set()) else 0
                labels.append(label)
    
    X = pd.DataFrame(rows)
    print(f"Built feature matrix: {X.shape}")
    
    if is_train:
        y = np.array(labels)
        print(f"Positive rate: {y.mean():.4f}")
        return X, y, pair_ids
    else:
        return X, pair_ids


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, '.')
    from blocking import generate_candidates_optimized
    
    print("Testing feature engineering...")
    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str, nrows=100)
    s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str, nrows=500)
    s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str, nrows=500)
    gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str)
    
    # Filter GT
    s1_ids = set(s1['entity_id'].values)
    gt_sample = gt[gt['source1_entity_id'].isin(s1_ids)]
    
    labels_dict = {}
    for _, row in gt_sample.iterrows():
        if pd.notna(row['matched_entity_ids']) and row['matched_entity_ids']:
            labels_dict[row['source1_entity_id']] = set(row['matched_entity_ids'].split(','))
        else:
            labels_dict[row['source1_entity_id']] = set()
    
    candidates_df = generate_candidates_optimized(s1, s2, s3, max_candidates_per_s1=50)
    
    # Convert to dict
    candidates_dict = {}
    for _, row in candidates_df.iterrows():
        cands = row['candidate_entity_ids'].split(',') if row['candidate_entity_ids'] else []
        candidates_dict[row['source1_entity_id']] = cands
    
    cand_df = pd.concat([s2, s3], ignore_index=True)
    X, y, pairs = build_feature_dataset(s1, cand_df, candidates_dict, labels_dict, is_train=True)
    print(f"Feature columns: {list(X.columns)}")
    print(X.head())