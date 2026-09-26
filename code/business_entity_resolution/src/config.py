"""Paths and global settings. Override any path with environment variables."""
import os
from pathlib import Path

# src/ -> business_entity_resolution/ -> code/ -> student_resource/
_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = Path(os.environ.get("ER_DATA_DIR", _ROOT / "dataset"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", _ROOT / "work"))
OUT_DIR = Path(os.environ.get("ER_OUT_DIR", _ROOT / "output"))

N_JOBS = int(os.environ.get("ER_N_JOBS", os.cpu_count() or 8))
SEED = 42

for _d in (WORK_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def raw_path(split, src):
    """Raw TSV for split in {train,test}, src in {1,2,3}."""
    return DATA_DIR / split / f"{split}_source{src}.tsv"


def prep_path(split, src):
    """Normalized parquet cache produced by prep.py."""
    return WORK_DIR / f"{split}_s{src}.parquet"
