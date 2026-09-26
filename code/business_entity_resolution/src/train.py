"""Stage 4: train the pair classifier on TRAIN candidates and tune the decision rule.

Validation protocol (mirrors the test-time decision exactly):
  * EVAL_S1: a random set of Source-1 entities held out for scoring.
  * eval queries: every S2/S3 query whose true S1 is in EVAL_S1, plus every query
    that has an EVAL_S1 entity among its top-3 blocking candidates (so false
    merges onto held-out entities are counted).
  * training queries: a random sample of the remaining queries.
Decision rule: each query is assigned to its highest-probability S1 candidate if
that probability >= threshold (each S2/S3 record matches at most one S1). The
threshold maximizes the exact macro F0.5 on EVAL_S1.

Model: XGBoost (CUDA if available), binary logistic on pair features.

Usage: python train.py [--n-train-q 800000] [--n-eval-s1 40000] [--k 10]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from blocking import load_cands, load_tables
from evaluate import macro_f05
from features import Context, build_idf, compute_features, make_pool

XGB_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                  max_depth=9, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, reg_lambda=2.0, max_bin=256)


def device():
    try:
        xgb.train({"device": "cuda", "tree_method": "hist"}, xgb.DMatrix(np.zeros((4, 1)), label=[0, 1, 0, 1]), 1)
        return "cuda"
    except Exception:
        return "cpu"


def assign(pairs, prob):
    """Per query: best candidate and its probability. Returns (q_rows, s1_rows, best_prob)."""
    d = pd.DataFrame({"q": pairs.q_row.values, "s": pairs.s1_row.values, "p": prob})
    best = d.sort_values(["q", "p"], ascending=[True, False]).drop_duplicates("q")
    return best.q.values, best.s.values, best.p.values


def sweep(eval_s1, q, s, p, truth, grid=None):
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.025), 3)
    res = []
    for t in grid:
        m = p >= t
        f, info = macro_f05(eval_s1, q[m], s[m], truth)
        res.append((t, f, info))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-q", type=int, default=800_000)
    ap.add_argument("--n-eval-s1", type=int, default=40_000)
    ap.add_argument("--k", type=int, default=10, help="use candidates with rank < k")
    ap.add_argument("--mode", default="v1", help="blocking version (candidate files)")
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--full", action="store_true", help="final fit: also train on eval queries")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(config.SEED)

    s1, q = load_tables("train")
    truth = np.load(config.WORK_DIR / "train_truth.npy")
    cands = load_cands("train", args.mode, args.k)
    print(f"cands (rank<{args.k}): {len(cands):,}  [{time.time() - t0:.0f}s]", flush=True)
    context = Context(cands)
    print(f"context done [{time.time() - t0:.0f}s]", flush=True)

    # ---- split
    eval_s1 = rng.choice(len(s1), args.n_eval_s1, replace=False)
    in_eval = np.zeros(len(s1), bool)
    in_eval[eval_s1] = True
    eval_q = np.zeros(len(q), bool)
    eval_q[(truth >= 0) & in_eval[np.maximum(truth, 0)]] = True
    near = cands[(cands["rank"] < 3) & in_eval[cands.s1_row.values]].q_row.values
    eval_q[near] = True
    pool_q = np.flatnonzero(~eval_q)
    train_q = np.zeros(len(q), bool)
    train_q[rng.choice(pool_q, min(args.n_train_q, len(pool_q)), replace=False)] = True
    if args.full:
        train_q |= eval_q
    tr = cands[train_q[cands.q_row.values]]
    ev = cands[eval_q[cands.q_row.values]]
    print(f"train pairs {len(tr):,} ({train_q.sum():,} q) | eval pairs {len(ev):,} ({eval_q.sum():,} q)", flush=True)

    idf = build_idf(s1)
    with make_pool(idf) as pool:
        Xtr = compute_features(tr, s1, q, pool, context)
        print(f"train features {Xtr.shape} [{time.time() - t0:.0f}s]", flush=True)
        Xev = compute_features(ev, s1, q, pool, context)
        print(f"eval features {Xev.shape} [{time.time() - t0:.0f}s]", flush=True)
    ytr = (truth[tr.q_row.values] == tr.s1_row.values).astype(np.float32)
    yev = (truth[ev.q_row.values] == ev.s1_row.values).astype(np.float32)
    print(f"positive rate train {ytr.mean():.3f} eval {yev.mean():.3f}", flush=True)

    dev = device()
    params = dict(XGB_PARAMS, device=dev, seed=config.SEED)
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dev_m = xgb.DMatrix(Xev, label=yev)
    bst = xgb.train(params, dtr, args.rounds, evals=[(dtr, "train"), (dev_m, "eval")],
                    early_stopping_rounds=50, verbose_eval=100)
    print(f"xgb ({dev}) best_iter={bst.best_iteration} [{time.time() - t0:.0f}s]", flush=True)
    pev = bst.predict(dev_m, iteration_range=(0, bst.best_iteration + 1))

    qa, sa, pa = assign(ev, pev)
    res = sweep(eval_s1, qa, sa, pa, truth)
    best_t, best_f, best_info = max(res, key=lambda r: r[1])
    for t, f, info in res[::4]:
        print(f"  t={t:.3f} F0.5={f:.4f} P={info['precision']:.4f} R={info['recall']:.4f} single={info['singleton_acc']:.4f}")
    print(f"BEST t={best_t:.3f} F0.5={best_f:.4f} {best_info}", flush=True)
    # upper bound given blocking (perfect classifier on these candidates)
    hit = yev.astype(bool)
    ub, _ = macro_f05(eval_s1, ev.q_row.values[hit], ev.s1_row.values[hit], truth)
    print(f"blocking ceiling F0.5 (rank<{args.k}) on eval: {ub:.4f}")

    imp = bst.get_score(importance_type="gain")
    print("top features:", sorted(imp.items(), key=lambda x: -x[1])[:15])
    mdir = config.WORK_DIR / f"model_{args.mode}"
    mdir.mkdir(exist_ok=True)
    bst.save_model(mdir / "xgb.json")
    json.dump({"threshold": float(best_t), "k": args.k, "mode": args.mode, "best_iteration": int(bst.best_iteration),
               "features": list(Xtr.columns), "val_f05": best_f},
              open(mdir / "meta.json", "w"), indent=2)
    print(f"saved model [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
