# Business Entity Resolution — reproduction guide

Pipeline: **normalize → block (TF-IDF top-k per country) → pair features → XGBoost → one-S1-per-record assignment with a tuned threshold**.
It uses only the provided training/test data. There are no external APIs or lookups.

## Environment

- Python 3.13 (tested on 3.13.3), Windows 11. Linux works too.
- `pip install -r requirements.txt`
- Hardware used: 16-core i7, 16 GB RAM, RTX 4060 8 GB. XGBoost uses CUDA if available and falls back to CPU otherwise.

## Data layout

By default the code expects this layout (override with `ER_DATA_DIR`, `ER_WORK_DIR`, `ER_OUT_DIR`):

```
student_resource/
  dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
  dataset/test/test_source{1,2,3}.tsv
  code/business_entity_resolution/src/   <- run from here
  work/     (intermediate caches, created automatically)
  output/   (matching_results.tsv, candidate_pairs.tsv)
```

## Run end-to-end

```bash
cd src
python run_all.py
```

Or run the steps individually:

| Step | Command | What it does |
|---|---|---|
| 1 | `python translit_dict.py` | Learns a non-Latin→Latin token dictionary (Devanagari, Kannada, …) from **training** matches |
| 2 | `python prep.py --split train` / `--split test` | Normalizes names and addresses (multiprocess) → parquet |
| 3 | `python blocking.py --split train --k 20` / `--split test` | Per-country TF-IDF (name words, name char 3-grams, address words, numbers) → top-k Source-1 candidates per S2/S3 record; prints recall@k on train |
| 4 | `python train.py` | Pair features + XGBoost; tunes the threshold on held-out S1 entities with the exact macro F0.5 |
| 5 | `python predict.py` | Scores test candidates, assigns matches, writes `output/*.tsv` |

Then validate:

```bash
cd ../../..   # student_resource/
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Source files

| File | Purpose |
|---|---|
| `config.py` | Paths and settings |
| `normalize.py` | Rule-based name/address canonicalization (legal forms, street types, US/India states, French street types, transliteration) |
| `translit_dict.py` | Learned script dictionary (training data only) |
| `prep.py` | Parallel normalization and parquet cache |
| `blocking.py` | Candidate generation and recall report |
| `features.py` | Pairwise similarity, IDF-overlap and context features |
| `evaluate.py` | Exact challenge metric (per-entity F0.5, macro) |
| `train.py` | Validation split, model training, threshold tuning |
| `predict.py` | Test inference and submission writing |
| `run_all.py` | Runs everything in order |
