"""Stage 5: score TEST candidates, assign matches, write both submission files.

  output/candidate_pairs.tsv  - exactly the pairs the model scores (blocking rank < k)
  output/matching_results.tsv - each S2/S3 query assigned to its best S1 candidate
                                if probability >= tuned threshold

Usage: python predict.py
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from blocking import load_cands, load_tables
from features import Context, build_idf, compute_features, make_pool
from train import assign, device

PAIR_CHUNK = 6_000_000


def write_lists(path, header, s1_ids, s1_rows, q_ids, q_rows):
    """One line per S1 entity: id <TAB> comma-joined S2/S3 ids (may be empty)."""
    order = np.argsort(s1_rows, kind="stable")
    sr, qr = s1_rows[order], q_rows[order]
    starts = np.searchsorted(sr, np.arange(len(s1_ids)), side="left")
    ends = np.searchsorted(sr, np.arange(len(s1_ids)), side="right")
    q_ids = np.asarray(q_ids, dtype=object)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for i, sid in enumerate(s1_ids):
            a, b = starts[i], ends[i]
            ids = ",".join(dict.fromkeys(q_ids[qr[a:b]])) if b > a else ""
            f.write(f"{sid}\t{ids}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="v1", help="model/blocking version to use")
    args = ap.parse_args()
    t0 = time.time()
    mdir = config.WORK_DIR / f"model_{args.mode}"
    meta = json.load(open(mdir / "meta.json"))
    bst = xgb.Booster()
    bst.load_model(mdir / "xgb.json")
    bst.set_param({"device": device()})

    s1, q = load_tables("test")
    cands = load_cands("test", meta["mode"], meta["k"])
    print(f"test cands (rank<{meta['k']}): {len(cands):,} [{time.time() - t0:.0f}s]", flush=True)
    context = Context(cands)
    idf = build_idf(s1)

    prob = np.empty(len(cands), np.float32)
    with make_pool(idf) as pool:
        for start in range(0, len(cands), PAIR_CHUNK):
            part = cands.iloc[start:start + PAIR_CHUNK]
            X = compute_features(part, s1, q, pool, context)[meta["features"]]
            prob[start:start + len(part)] = bst.predict(xgb.DMatrix(X), iteration_range=(0, meta["best_iteration"] + 1))
            print(f"  scored {start + len(part):,}/{len(cands):,} [{time.time() - t0:.0f}s]", flush=True)
    np.save(config.WORK_DIR / f"test_prob_{args.mode}.npy", prob)

    qa, sa, pa = assign(cands, prob)
    m = pa >= meta["threshold"]
    print(f"assigned {m.sum():,} of {len(qa):,} queries (t={meta['threshold']})", flush=True)

    s1_ids = s1.entity_id.values
    q_ids = q.entity_id.values
    write_lists(config.OUT_DIR / "matching_results.tsv", ["source1_entity_id", "matched_entity_ids"],
                s1_ids, sa[m], q_ids, qa[m])
    write_lists(config.OUT_DIR / "candidate_pairs.tsv", ["source1_entity_id", "candidate_entity_ids"],
                s1_ids, cands.s1_row.values, q_ids, cands.q_row.values)
    print(f"wrote outputs to {config.OUT_DIR} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
