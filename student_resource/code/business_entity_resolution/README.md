# Business Entity Resolution - Amazon ML Challenge 2026

## Team: [YOUR_TEAM_NAME]

## Overview
This pipeline solves the Business Entity Resolution challenge: matching Source 1 (deduplicated reference) entities to Source 2 and Source 3 records using only the provided training data. No external data/lookup is used.

## Approach Summary
1. **Multi-pass Blocking** (recall >95%): Country-aware keys including name tokens, PIN codes, phonetic encodings (Metaphone, Soundex, NYSIIS), address tokens, trigrams, and cross-field combinations. Handles multilingual names (Hindi, Tamil, French, English) via transliteration normalization.
2. **Rich Pairwise Features** (40+): Jaccard, Levenshtein, Jaro-Winkler, RapidFuzz ratios for name/address; PIN/city/numeric token matching; phonetic matches; legal suffix agreement; cross-field token overlap; length ratios.
3. **LightGBM Classifier**: Trained with `scale_pos_weight` for class imbalance (~1:50). Optimized for **macro F_0.5** via per-country threshold tuning on GroupKFold validation.
4. **Singleton Handling**: Explicit threshold optimization captures "no match" cases (5.6% of train) for full F_0.5 credit.

## Directory Structure
```
business_entity_resolution/
├── src/
│   ├── blocking.py      # Multi-pass candidate generation
│   ├── features.py      # Pairwise feature engineering
│   ├── train.py         # LightGBM training + macro F_0.5 threshold optimization
│   ├── predict.py       # Test inference + submission file generation
│   └── pipeline.py      # End-to-end orchestration
├── requirements.txt     # Pinned dependencies
└── README.md           # This file
```

## Reproduction Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline (from student_resource/)
```bash
# Quick test on 10% data
python ../pipeline.py --stage full --sample 0.1 --validate

# Full training (takes ~30-60 min)
python ../pipeline.py --stage full --validate
```

### 3. Run Individual Stages
```bash
# EDA only
python ../pipeline.py --stage eda

# Training only (from student_resource/)
python ../train.py --sample 1.0 --data-dir dataset/train --output-dir models

# Inference only (from student_resource/)
python ../predict.py --test-dir dataset/test --model-dir models --output-dir output --validate
```

### 4. Validate Outputs
```bash
cd student_resource
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

### 5. Create Submission Zip
```bash
cd student_resource
zip -r ../<team_name>_submission.zip \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    code/business_entity_resolution/ \
    Documentation_template.md
```

## Key Files Generated
- `output/matching_results.tsv` - Final matches (leaderboard scored)
- `output/candidate_pairs.tsv` - Blocking candidates (audit)
- `models/lgb_model.txt` - Trained LightGBM model
- `models/global_threshold.pkl` - Global decision threshold
- `models/country_thresholds.pkl` - Per-country thresholds (US, India, France)

## Model Details
- **Algorithm**: LightGBM (GBDT)
- **Parameters**: 127 leaves, 0.05 LR, 0.8 feature/bagging fraction, L1/L2 reg
- **Imbalance**: `scale_pos_weight` = neg/pos ratio (~50)
- **Validation**: GroupKFold (by source1_entity_id), stratified by country
- **Metric**: Macro F_0.5 (precision-weighted, β=0.5)
- **Thresholds**: Per-country optimized (US, India, France)

## Multilingual Support
- **Devanagari/Hindi**: Transliteration normalization (aa→a, kh→k, naggar→nagar, etc.)
- **Tamil/Telugu/Gujarati/Kannada/Malayalam/Punjabi/Odia**: Script detection + Latin fallback
- **French**: Accent removal, French legal suffixes (SARL, SAS, etc.), address stopwords
- **Phonetic**: Double Metaphone, Soundex, NYSIIS for cross-script matching

## Performance Notes
- Blocking recall target: >95% (validated on train holdout)
- Typical candidates/S1: 50-100 (configurable via `--max-candidates`)
- Training time: ~20-40 min on full data (8-core CPU)
- Inference time: ~5-10 min on full test set

## Compliance
- ✅ No external data/APIs used
- ✅ MIT/Apache 2.0 compatible dependencies
- ✅ Model < 8B parameters (LightGBM ~few MB)
- ✅ Output format validated via `validate_submission.py`

## Contact
[YOUR_TEAM_EMAIL]