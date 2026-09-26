#!/usr/bin/env python3
"""
Blocking / Candidate Generation for Amazon ML Challenge 2026
Multi-pass blocking with high recall targeting + Multilingual support
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict
from rapidfuzz import fuzz
import jellyfish
import unicodedata

# ============================================================
# COMPREHENSIVE MULTILINGUAL SUPPORT
# ============================================================

# Legal suffixes across all languages/countries
FRENCH_LEGAL_SUFFIXES = {
    'sarl', 'sas', 'sasu', 'eurl', 'sa', 'snc', 'scop', 'scic', 'sei',
    'societe', 'société', 'entreprise', 'etablissement', 'établissement',
    'group', 'groupe', 'holding', 'international', 'france', 'paris'
}

# Indian legal suffixes (English + transliterated Hindi/regional)
INDIAN_LEGAL_SUFFIXES = {
    'pvt', 'private', 'ltd', 'limited', 'llp', 'opc', 'llc',
    'corp', 'corporation', 'company', 'co', 'inc', 'incorporated',
    'industries', 'industry', 'enterprises', 'enterprise', 'trading',
    'exports', 'imports', 'international', 'india', 'indian',
    # Hindi/regional transliterations
    'udyog', 'udyogam', 'vyapar', 'vyapaar', 'karobar', 'karyalay',
    'sanstha', 'santha', 'mandir', 'mandal', 'sangh', 'sangathan',
    'privat', 'limited', 'kंपनी', 'कंपनी', 'उद्योग', 'व्यापार', 'कारोबार'
}

US_LEGAL_SUFFIXES = {
    'inc', 'llc', 'ltd', 'corp', 'corporation', 'company', 'co',
    'llp', 'lp', 'pllc', 'plc', 'pc', 'pa', 'lllp', 'lp',
    'incorporated', 'limited', 'partnership', 'associates', 'assoc',
    'group', 'holdings', 'enterprises', 'international', 'usa', 'america'
}

ALL_LEGAL_SUFFIXES = FRENCH_LEGAL_SUFFIXES | INDIAN_LEGAL_SUFFIXES | US_LEGAL_SUFFIXES

# Address stopwords - comprehensive for all languages
FRENCH_ADDRESS_STOPWORDS = {
    'rue', 'avenue', 'boulevard', 'place', 'allee', 'allée', 'impasse',
    'chemin', 'route', 'quai', 'square', 'cours', 'promenade', 'voie',
    'nord', 'sud', 'est', 'ouest', 'centre', 'ville', 'quartier',
    'batiment', 'bâtiment', 'etage', 'étage', 'appartement', 'appt',
    'residence', 'résidence', 'entree', 'entrée', 'escalier',
    'proche', 'pres', 'près', 'face', 'devant', 'derriere', 'derrière',
    'a', 'à', 'au', 'aux', 'du', 'de', 'la', 'le', 'les', 'sur', 'sous',
    'entre', 'vers', 'par', 'pour', 'avec', 'sans', 'dans', 'en'
}

US_ADDRESS_STOPWORDS = {
    'street', 'st', 'road', 'rd', 'avenue', 'ave', 'drive', 'dr',
    'lane', 'ln', 'boulevard', 'blvd', 'circle', 'cir', 'court', 'ct',
    'place', 'pl', 'way', 'north', 'south', 'east', 'west', 'n', 's', 'e', 'w',
    'unit', 'apt', 'apartment', 'suite', 'ste', 'floor', 'fl',
    'near', 'opposite', 'behind', 'beside', 'next', 'to', 'the', 'of', 'and'
}

# Indian address stopwords - English + Hindi + regional transliterations
INDIAN_ADDRESS_STOPWORDS = {
    'road', 'rd', 'street', 'st', 'lane', 'ln', 'avenue', 'ave',
    'colony', 'nagar', 'extension', 'extn', 'block', 'sector', 'sec',
    'phase', 'area', 'zone', 'district', 'dist', 'taluk', 'taluka',
    'near', 'opp', 'opposite', 'behind', 'beside', 'next', 'to', 'the', 'of', 'and',
    'pincode', 'pin', 'code', 'postal',
    # Hindi/regional address terms (transliterated)
    'marg', 'path', 'chowk', 'chawk', 'circle', 'sqr', 'square',
    'nagar', 'puram', 'vihar', 'dham', 'ganj', 'bazar', 'bazaar', 'market',
    'mandir', 'masjid', 'gurudwara', 'church', 'temple',
    'school', 'hospital', 'clinic', 'bank', 'atm',
    'cross', 'main', 'link', 'service', 'ring', 'outer', 'inner',
    'east', 'west', 'north', 'south', 'central', 'new', 'old',
    'layout', 'extension', 'ext', 'block', 'sec', 'sector',
    'nearby', 'opp', 'opposite', 'beside', 'behind', 'front',
    'gate', 'entrance', 'exit', 'corner', 'junction', 'signal',
    'flyover', 'bridge', 'underpass', 'metro', 'station',
    'नगर', 'पुरम', 'विहार', 'धाम', 'गंज', 'बाजार', 'मार्ग', 'पथ',
    'चौक', 'सर्किल', 'स्कूल', 'हॉस्पिटल', 'बैंक', 'एटीएम'
}

ALL_ADDRESS_STOPWORDS = FRENCH_ADDRESS_STOPWORDS | US_ADDRESS_STOPWORDS | INDIAN_ADDRESS_STOPWORDS

# Common transliteration variations for Indic languages
# These help match same word written differently (e.g., "bazaar" vs "bazar")
INDIC_TRANSLITERATION_MAP = {
    # Vowels
    'aa': 'a', 'ae': 'e', 'ai': 'e', 'au': 'o', 'ou': 'u',
    'ee': 'i', 'ei': 'i', 'ii': 'i', 'oo': 'u', 'uu': 'u',
    # Consonants
    'kh': 'k', 'gh': 'g', 'ch': 'c', 'jh': 'j', 'th': 't', 'dh': 'd',
    'ph': 'p', 'bh': 'b', 'sh': 's', 'zh': 'z', 'ksh': 'k', 'tr': 't',
    # Common endings
    'iya': 'ia', 'iyaan': 'ian', 'iyaar': 'iar', 'iwal': 'wal',
    'eshwar': 'eshwar', 'ishwar': 'ishwar', 'swamy': 'swami',
    'nagar': 'nagar', 'naggar': 'nagar', 'pura': 'pur', 'puram': 'puram',
    'vihar': 'vihar', 'vihaar': 'vihar', 'dham': 'dham', 'dhaam': 'dham',
    'ganj': 'ganj', 'gunj': 'ganj', 'bazar': 'bazar', 'bazaar': 'bazar',
    'mandir': 'mandir', 'mandira': 'mandir', 'masjid': 'masjid', 'maszid': 'masjid',
}

# Script detection patterns
DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]')
BENGALI_PATTERN = re.compile(r'[\u0980-\u09FF]')
TAMIL_PATTERN = re.compile(r'[\u0B80-\u0BFF]')
TELUGU_PATTERN = re.compile(r'[\u0C00-\u0C7F]')
GUJARATI_PATTERN = re.compile(r'[\u0A80-\u0AFF]')
KANNADA_PATTERN = re.compile(r'[\u0C80-\u0CFF]')
MALAYALAM_PATTERN = re.compile(r'[\u0D00-\u0D7F]')
PUNJABI_PATTERN = re.compile(r'[\u0A00-\u0A7F]')
ORIYA_PATTERN = re.compile(r'[\u0B00-\u0B7F]')

def detect_script(text):
    """Detect primary script in text"""
    if pd.isna(text) or text == '':
        return 'latin'
    text = str(text)
    scripts = []
    if DEVANAGARI_PATTERN.search(text): scripts.append('devanagari')
    if BENGALI_PATTERN.search(text): scripts.append('bengali')
    if TAMIL_PATTERN.search(text): scripts.append('tamil')
    if TELUGU_PATTERN.search(text): scripts.append('telugu')
    if GUJARATI_PATTERN.search(text): scripts.append('gujarati')
    if KANNADA_PATTERN.search(text): scripts.append('kannada')
    if MALAYALAM_PATTERN.search(text): scripts.append('malayalam')
    if PUNJABI_PATTERN.search(text): scripts.append('punjabi')
    if ORIYA_PATTERN.search(text): scripts.append('oriya')
    return scripts[0] if scripts else 'latin'

def remove_accents(text):
    """Remove diacritics for cross-language matching"""
    if pd.isna(text) or text == '':
        return ''
    text = str(text)
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def normalize_indic_transliteration(text):
    """Normalize common Indic transliteration variations"""
    if not text:
        return text
    text = text.lower()
    # Apply common transliteration normalizations
    for variant, canonical in INDIC_TRANSLITERATION_MAP.items():
        text = re.sub(rf'\b{variant}\b', canonical, text)
    return text

def normalize_text(text, remove_diacritics=True, normalize_indic=True):
    """Normalize text for blocking keys - language agnostic"""
    if pd.isna(text) or text == '':
        return ''
    text = str(text).lower()
    if remove_diacritics:
        text = remove_accents(text)
    if normalize_indic:
        text = normalize_indic_transliteration(text)
    # Keep alphanumeric and spaces only
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_name_tokens(name, max_tokens=4, country=None):
    """Extract first N meaningful tokens from business name"""
    norm = normalize_text(name)
    if not norm:
        return []
    
    # Country-aware suffix removal
    suffixes = ALL_LEGAL_SUFFIXES
    if country == 'France':
        suffixes = suffixes | FRENCH_LEGAL_SUFFIXES
    elif country == 'India':
        suffixes = suffixes | INDIAN_LEGAL_SUFFIXES
    elif country == 'US':
        suffixes = suffixes | US_LEGAL_SUFFIXES
    
    tokens = [t for t in norm.split() if t not in suffixes and len(t) > 1 and not t.isdigit()]
    return tokens[:max_tokens]

def get_address_tokens(addr, max_tokens=5, country=None):
    """Extract meaningful tokens from address"""
    norm = normalize_text(addr)
    if not norm:
        return []
    
    stopwords = ALL_ADDRESS_STOPWORDS
    tokens = [t for t in norm.split() if t not in stopwords and len(t) > 2 and not t.isdigit()]
    return tokens[:max_tokens]

def extract_pincode(addr, country=None):
    """Extract postal codes - country aware"""
    if pd.isna(addr) or addr == '':
        return None
    text = str(addr)
    if country == 'India':
        matches = re.findall(r'\b(\d{6})\b', text)  # India: 6 digits
    elif country == 'France':
        matches = re.findall(r'\b(\d{5})\b', text)  # France: 5 digits
    elif country == 'US':
        matches = re.findall(r'\b(\d{5})(?:-\d{4})?\b', text)  # US: 5+4
    else:
        matches = re.findall(r'\b(\d{5,6})\b', text)
    return matches[0] if matches else None

def get_phonetic_name(name):
    """Get Double Metaphone primary key for name - works across languages"""
    norm = normalize_text(name, remove_diacritics=True, normalize_indic=True)
    if not norm:
        return ''
    try:
        # Use primary metaphone code
        primary, _ = jellyfish.metaphone(norm)
        return primary[:8] if primary else ''
    except:
        return ''

def get_soundex_name(name):
    """Soundex as backup phonetic encoding"""
    norm = normalize_text(name, remove_diacritics=True, normalize_indic=True)
    if not norm:
        return ''
    try:
        return jellyfish.soundex(norm)
    except:
        return ''

def get_nysiis_name(name):
    """NYSIIS phonetic - better for non-English names"""
    norm = normalize_text(name, remove_diacritics=True, normalize_indic=True)
    if not norm:
        return ''
    try:
        return jellyfish.nysiis(norm)
    except:
        return ''

def build_blocking_keys(row, source_prefix):
    """Generate all blocking keys for a record - multilingual aware"""
    keys = []
    country = row['country']
    name = row['business_name']
    addr = row['business_address']
    
    # Detect script for potential script-specific handling
    name_script = detect_script(name)
    addr_script = detect_script(addr)
    
    # 1. Country + first 3 name tokens (most discriminative)
    name_tokens = get_name_tokens(name, 3, country)
    for i, token in enumerate(name_tokens):
        if len(token) >= 3:
            keys.append(f"{country}|NAME_TOK_{i}|{token}")
    
    # 2. Country + PIN code (high precision)
    pin = extract_pincode(addr, country)
    if pin:
        keys.append(f"{country}|PIN|{pin}")
    
    # 3. Country + phonetic name (handles typos/transliteration)
    phonetic = get_phonetic_name(name)
    if phonetic:
        keys.append(f"{country}|PHON|{phonetic}")
    
    # 4. Soundex backup for French/Indic names
    soundex = get_soundex_name(name)
    if soundex:
        keys.append(f"{country}|SOUNDEX|{soundex}")
    
    # 5. NYSIIS - better for European/Indian names
    nysiis = get_nysiis_name(name)
    if nysiis:
        keys.append(f"{country}|NYSIIS|{nysiis}")
    
    # 6. Country + address tokens (street + city)
    addr_tokens = get_address_tokens(addr, 3, country)
    for i, token in enumerate(addr_tokens):
        if len(token) >= 4:
            keys.append(f"{country}|ADDR_TOK_{i}|{token}")
    
    # 7. Country + first 2 name tokens combined (for multi-word names)
    if len(name_tokens) >= 2:
        combo = '_'.join(name_tokens[:2])
        keys.append(f"{country}|NAME_COMBO|{combo}")
    
    # 8. Country + first name token + first addr token (cross-field)
    if name_tokens and addr_tokens:
        cross = f"{name_tokens[0]}_{addr_tokens[0]}"
        keys.append(f"{country}|CROSS|{cross}")
    
    # 9. Character n-grams for fuzzy matching (trigrams)
    name_clean = normalize_text(name, remove_diacritics=True, normalize_indic=True).replace(' ', '')
    if len(name_clean) >= 3:
        for i in range(len(name_clean) - 2):
            trigram = name_clean[i:i+3]
            keys.append(f"{country}|TRIGRAM|{trigram}")
            if i >= 2:  # Limit trigrams to first 3
                break
    
    # 10. Country only (fallback - will generate many candidates, use last)
    keys.append(f"{country}|FALLBACK")
    
    return keys

def generate_candidates(s1_df, s2_df, s3_df, max_candidates_per_s1=100):
    """
    Multi-pass blocking to generate candidate pairs
    Returns DataFrame with columns: source1_entity_id, candidate_entity_ids (comma-separated)
    """
    print("Building blocking keys for S1...")
    s1_keys = defaultdict(list)
    for _, row in s1_df.iterrows():
        for key in build_blocking_keys(row, 'S1'):
            s1_keys[key].append(row['entity_id'])
    
    print("Building blocking keys for S2...")
    s2_keys = defaultdict(list)
    for _, row in s2_df.iterrows():
        for key in build_blocking_keys(row, 'S2'):
            s2_keys[key].append(row['entity_id'])
    
    print("Building blocking keys for S3...")
    s3_keys = defaultdict(list)
    for _, row in s3_df.iterrows():
        for key in build_blocking_keys(row, 'S3'):
            s3_keys[key].append(row['entity_id'])
    
    print("Generating candidates per S1 entity...")
    # For each S1 entity, collect candidates from all matching keys
    s1_candidates = defaultdict(set)
    
    # Priority order of blocking keys (high precision first)
    key_priority = [
        'PIN', 'NAME_TOK_0', 'NAME_COMBO', 'PHON', 
        'NAME_TOK_1', 'ADDR_TOK_0', 'ADDR_TOK_1', 'NAME_TOK_2', 'ADDR_TOK_2', 'FALLBACK'
    ]
    
    for s1_id in s1_df['entity_id'].values:
        # Get all keys for this S1 entity
        row = s1_df[s1_df['entity_id'] == s1_id].iloc[0]
        s1_key_list = build_blocking_keys(row, 'S1')
        
        # Collect candidates by priority
        candidates = set()
        for key_type in key_priority:
            if len(candidates) >= max_candidates_per_s1:
                break
            matching_keys = [k for k in s1_key_list if key_type in k]
            for key in matching_keys:
                if key in s2_keys:
                    candidates.update(s2_keys[key])
                if key in s3_keys:
                    candidates.update(s3_keys[key])
        
        s1_candidates[s1_id] = candidates
    
    # Convert to output format
    results = []
    for s1_id in s1_df['entity_id'].values:
        cands = sorted(s1_candidates.get(s1_id, set()))
        results.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ','.join(cands) if cands else ''
        })
    
    return pd.DataFrame(results)

def generate_candidates_optimized(s1_df, s2_df, s3_df, max_candidates_per_s1=100):
    """
    Optimized version using pandas merge with priority-based key processing
    """
    print("Creating blocking key tables...")
    
    # Priority order: high precision keys first
    KEY_PRIORITY = [
        'PIN',
        'NAME_TOK_0', 'NAME_COMBO', 
        'PHON', 'SOUNDEX', 'NYSIIS',
        'NAME_TOK_1', 'ADDR_TOK_0', 'CROSS',
        'NAME_TOK_2', 'ADDR_TOK_1', 'TRIGRAM',
        'ADDR_TOK_2',
        'FALLBACK'
    ]
    
    # Create key tables for each source
    def create_key_table(df, source):
        rows = []
        for _, row in df.iterrows():
            for key in build_blocking_keys(row, source):
                # Extract key type for priority sorting
                key_type = 'OTHER'
                for kt in KEY_PRIORITY:
                    if kt in key:
                        key_type = kt
                        break
                rows.append({
                    'entity_id': row['entity_id'], 
                    'blocking_key': key, 
                    'source': source,
                    'key_type': key_type
                })
        return pd.DataFrame(rows)
    
    s1_keys_df = create_key_table(s1_df, 'S1')
    s2_keys_df = create_key_table(s2_df, 'S2')
    s3_keys_df = create_key_table(s3_df, 'S3')
    
    s23_keys_df = pd.concat([s2_keys_df, s3_keys_df], ignore_index=True)
    
    print("Processing keys by priority...")
    
    # For each S1 entity, collect candidates by priority
    all_s1_ids = s1_df['entity_id'].values
    s1_candidates = {sid: [] for sid in all_s1_ids}
    s1_candidate_sets = {sid: set() for sid in all_s1_ids}
    
    # Process each key type in priority order
    for key_type in KEY_PRIORITY:
        # Filter S1 keys of this type
        s1_kt = s1_keys_df[s1_keys_df['key_type'] == key_type]
        if len(s1_kt) == 0:
            continue
            
        # Filter S2/S3 keys of this type
        s23_kt = s23_keys_df[s23_keys_df['key_type'] == key_type]
        if len(s23_kt) == 0:
            continue
        
        # Merge on blocking_key
        s1_keys_sub = s1_kt[['entity_id', 'blocking_key']].rename(columns={'entity_id': 'source1_entity_id'})
        s23_keys_sub = s23_kt[['entity_id', 'blocking_key']].rename(columns={'entity_id': 'candidate_entity_id'})
        
        merged = s1_keys_sub.merge(s23_keys_sub, on='blocking_key', how='inner')
        
        # Add candidates for each S1 entity
        for _, row in merged.iterrows():
            s1_id = row['source1_entity_id']
            cand_id = row['candidate_entity_id']
            if cand_id not in s1_candidate_sets[s1_id]:
                s1_candidate_sets[s1_id].add(cand_id)
                s1_candidates[s1_id].append(cand_id)
    
    print("Building final candidate lists...")
    # Build results
    results = []
    for s1_id in all_s1_ids:
        cands = s1_candidates[s1_id][:max_candidates_per_s1]
        results.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ','.join(cands) if cands else ''
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Quick test
    print("Testing blocking pipeline...")
    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str, nrows=1000)
    s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str, nrows=1000)
    s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str, nrows=1000)
    
    candidates = generate_candidates_optimized(s1, s2, s3, max_candidates_per_s1=50)
    print(f"Generated candidates for {len(candidates)} S1 entities")
    print(f"Avg candidates per S1: {candidates['candidate_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0).mean():.1f}")
    print(f"Empty candidates: {(candidates['candidate_entity_ids'] == '').sum()}")
    print(candidates.head(10))