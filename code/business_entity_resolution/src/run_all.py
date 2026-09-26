"""End-to-end: data -> normalization -> blocking -> training -> test matching -> outputs.

Usage (from this src/ folder):  python run_all.py
Paths default to ../../../dataset, ../../../work, ../../../output; override with
ER_DATA_DIR / ER_WORK_DIR / ER_OUT_DIR.
"""
import subprocess
import sys

STEPS = [
    ["translit_dict.py"],
    ["prep.py", "--split", "train"],
    ["prep.py", "--split", "test"],
    ["blocking.py", "--split", "train", "--k", "20"],
    ["blocking.py", "--split", "test", "--k", "20"],
    ["train.py"],
    ["predict.py"],
]

if __name__ == "__main__":
    for step in STEPS:
        print(">>>", " ".join(step), flush=True)
        subprocess.run([sys.executable, *step], check=True)
