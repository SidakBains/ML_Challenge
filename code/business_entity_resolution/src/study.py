"""Study harness: two-stage cascade experiments on a fixed, reproducible sample.

Protocol
  eval   : the same 30,000 held-out S1 entities (and their queries) as train.py,
           split 50/50 into TUNE (every threshold / band limit is chosen here) and
           REPORT (final numbers + paired bootstrap CI vs the stage-1 baseline).
  A, B   : disjoint query samples from the remaining queries, split by the query's
           true S1 entity so no entity feeds both. Stage 1 trains on A; its
           out-of-sample scores on B define the band and train stage 2.

Cascade (per query; p_max = best stage-1 probability among its candidates)
  L <= p_max < U  -> stage 2 re-scores the query's top-5 stage-1 candidates and
                     assigns its best one if p2 >= tau2
  otherwise       -> stage-1 rule (assign the top candidate if p_max >= tau1)
Stage 2 is trained once on B's wide band [0.01, 0.995); (L, U, tau2) only decide
where it is applied, so the sweep needs no re-fitting.

Stage-2 methods (extra features on top of the stage-1 features and p1):
  control : none -- isolates the effect of a band-specialized re-fit
  dl      : Damerau-Levenshtein similarity (names, squashed names, addresses)
  me      : Monge-Elkan with Jaro-Winkler inner similarity, both directions
  stfidf  : Soft TF-IDF (Cohen et al. 2003), JW >= 0.9, both directions
  ce      : fine-tuned cross-encoder score, 2-fold cross-fitted on B (needs CUDA torch)

Usage (each step caches to WORK_DIR/study; rerun a step to overwrite it):
  python study.py split | features | stage1 | band | report
  python study.py stage2 --method {control,dl,me,stfidf,ce}
  python study.py all            # everything above, all methods except ce
"""
import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xgboost as xgb
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler

import config
from blocking import load_cands, load_tables
from evaluate import macro_f05
from features import Context, _cp, build_idf, compute_features, make_pool
from train import XGB_PARAMS, device

SD = config.WORK_DIR / "study"
N_EVAL_S1 = 30_000       # same as the v2 run (train.py --n-eval-s1 30000) -> identical eval set
N_A = N_B = 1_000_000    # ~20% of the ~9.9M non-eval queries, 10M pairs each
K = 10                   # candidate rank cut, as in v2
TOP2 = 5                 # stage 2 re-scores this many stage-1 candidates per query
WIDE = (0.01, 0.995)     # band stage 2 is trained on; every swept [L, U) lies inside it
FEAT_CHUNK = 3_000_000
GRID_L = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
GRID_U = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
GRID_T2 = np.round(np.arange(0.30, 0.91, 0.05), 2)
GRID_T1 = np.round(np.arange(0.30, 0.96, 0.01), 2)
S2_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", max_depth=6,
                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                 reg_lambda=2.0, max_bin=256)
CE_BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_MAX_TRAIN = 300_000   # per fold, bounds fine-tuning time


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------ metric
class Scorer:
    """Exact challenge metric (same formula as evaluate.macro_f05) with the truth
    counts precomputed, returning per-entity F0.5 so results can be bootstrapped."""

    def __init__(self, truth, n_s1):
        self.truth, self.n = truth, n_s1
        self.ntrue = np.bincount(truth[truth >= 0], minlength=n_s1)

    def per_entity(self, ents, qa, sa, beta=0.5):
        npred = np.bincount(sa, minlength=self.n)
        tp = np.bincount(sa[self.truth[qa] == sa], minlength=self.n)
        nt, npd, t = self.ntrue[ents], npred[ents], tp[ents].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(npd > 0, t / npd, 0.0)
            r = np.where(nt > 0, t / nt, 0.0)
            f = np.where((p + r) > 0, (1 + beta ** 2) * p * r / (beta ** 2 * p + r), 0.0)
        f = np.where(nt == 0, (npd == 0).astype(float), f)
        return f, p, r, npd, nt

    def score(self, ents, qa, sa):
        f, p, r, npd, nt = self.per_entity(ents, qa, sa)
        return dict(f05=float(f.mean()), precision=float(p[npd > 0].mean()), recall=float(r[nt > 0].mean()))


def bootstrap_delta(f_new, f_base, n_boot=2000, seed=config.SEED):
    """Paired bootstrap over entities: 95% CI of mean(F_new - F_base)."""
    d = f_new - f_base
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------ steps
def split():
    """Fixed, reproducible eval / A / B split, saved to split.npz."""
    s1, q = load_tables("train")
    truth = np.load(config.WORK_DIR / "train_truth.npy")
    near = load_cands("train", "v2", 3)
    rng = np.random.default_rng(config.SEED)  # same draw as train.py -> same eval entities
    eval_s1 = rng.choice(len(s1), N_EVAL_S1, replace=False)
    in_eval = np.zeros(len(s1), bool)
    in_eval[eval_s1] = True
    eval_q = np.zeros(len(q), bool)
    eval_q[(truth >= 0) & in_eval[np.maximum(truth, 0)]] = True
    eval_q[near.q_row.values[in_eval[near.s1_row.values]]] = True

    rng2 = np.random.default_rng(config.SEED + 1)
    grp_s1 = rng2.random(len(s1)) < 0.5  # True -> A
    grp_q = np.where(truth >= 0, grp_s1[np.maximum(truth, 0)], rng2.random(len(q)) < 0.5)
    qa = np.sort(rng2.choice(np.flatnonzero(~eval_q & grp_q), N_A, replace=False))
    qb = np.sort(rng2.choice(np.flatnonzero(~eval_q & ~grp_q), N_B, replace=False))
    perm = rng2.permutation(eval_s1)
    np.savez(SD / "split.npz", eval_s1=np.sort(eval_s1), tune_s1=np.sort(perm[:N_EVAL_S1 // 2]),
             report_s1=np.sort(perm[N_EVAL_S1 // 2:]), eval_q=np.flatnonzero(eval_q), qa=qa, qb=qb)
    log(f"split: eval_s1={len(eval_s1):,} (tune/report {N_EVAL_S1 // 2:,} each) eval_q={eval_q.sum():,} "
        f"A={len(qa):,} B={len(qb):,} queries")


def features():
    """Stage-1 features (same 45 as v2) for every candidate pair of A, B and eval."""
    spl = np.load(SD / "split.npz")
    s1, q = load_tables("train")
    truth = np.load(config.WORK_DIR / "train_truth.npy")
    cands = load_cands("train", "v2", K)
    context = Context(cands)
    log(f"cands {len(cands):,}, context done")
    for f in SD.glob("feat_*.parquet"):
        f.unlink()
    with make_pool(build_idf(s1)) as pool:
        for name, qs in (("A", spl["qa"]), ("B", spl["qb"]), ("E", spl["eval_q"])):
            sel = np.zeros(len(q), bool)
            sel[qs] = True
            pairs = cands[sel[cands.q_row.values]]
            for j in range(0, len(pairs), FEAT_CHUNK):
                p = pairs.iloc[j:j + FEAT_CHUNK]
                X = compute_features(p, s1, q, pool, context).reset_index(drop=True)
                X.insert(0, "q_row", p.q_row.values)
                X.insert(1, "s1_row", p.s1_row.values)
                X["y"] = (truth[p.q_row.values] == p.s1_row.values).astype(np.int8)
                X.to_parquet(SD / f"feat_{name}_{j // FEAT_CHUNK:02d}.parquet", index=False)
            log(f"features {name}: {len(pairs):,} pairs, {len(qs):,} queries")


def load_feat(name, cols=None):
    return pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(SD.glob(f"feat_{name}_*.parquet"))],
                     ignore_index=True)


def xcols(df):
    return [c for c in df.columns if c not in ("q_row", "s1_row", "y")]


def by_query(P, pcol="p1"):
    """Sort pairs by (q, -p); add rank/gap within the query; per-query top-1 table."""
    P = P.sort_values(["q_row", pcol], ascending=[True, False], kind="stable").reset_index(drop=True)
    q = P.q_row.values
    first = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1]
    sizes = np.diff(np.r_[first, len(P)])
    P[f"{pcol}_rank"] = (np.arange(len(P)) - np.repeat(first, sizes)).astype(np.int16)
    P[f"{pcol}_gap"] = np.repeat(P[pcol].values[first], sizes) - P[pcol].values
    Q = pd.DataFrame({"q_row": q[first], "p_max": P[pcol].values[first], "s_top": P.s1_row.values[first]})
    return P, Q


def baseline(scorer, spl, Q):
    """Stage-1 only: tau1 tuned on TUNE, scored on TUNE and REPORT."""
    def run(ents, t):
        m = Q.p_max.values >= t
        return scorer.per_entity(ents, Q.q_row.values[m], Q.s_top.values[m])[0]
    tau1 = max(GRID_T1, key=lambda t: run(spl["tune_s1"], t).mean())
    m = Q.p_max.values >= tau1
    qa, sa = Q.q_row.values[m], Q.s_top.values[m]
    return tau1, scorer.score(spl["tune_s1"], qa, sa), scorer.score(spl["report_s1"], qa, sa), run(spl["report_s1"], tau1)


def stage1():
    A = load_feat("A")
    cols = xcols(A)
    rng = np.random.default_rng(config.SEED + 2)
    uq = np.unique(A.q_row.values)
    is_dev = np.isin(A.q_row.values, rng.choice(uq, len(uq) // 20, replace=False))
    params = dict(XGB_PARAMS, device=device(), seed=config.SEED)
    dtr = xgb.QuantileDMatrix(A.loc[~is_dev, cols], label=A.y.values[~is_dev])
    ddv = xgb.QuantileDMatrix(A.loc[is_dev, cols], label=A.y.values[is_dev], ref=dtr)
    log(f"stage1 train pairs {int((~is_dev).sum()):,} dev {int(is_dev.sum()):,} pos rate {A.y.mean():.3f}")
    del A
    bst = xgb.train(params, dtr, 3000, evals=[(ddv, "dev")], early_stopping_rounds=50, verbose_eval=250)
    del dtr, ddv
    bst.save_model(SD / "stage1.json")
    json.dump({"best_iteration": int(bst.best_iteration), "features": cols}, open(SD / "stage1_meta.json", "w"))
    log(f"stage1 best_iter={bst.best_iteration}")
    for name in ("B", "E"):
        D = load_feat(name)
        p = bst.predict(xgb.DMatrix(D[cols]), iteration_range=(0, bst.best_iteration + 1))
        pd.DataFrame({"q_row": D.q_row.values, "s1_row": D.s1_row.values, "y": D.y.values,
                      "p1": p.astype(np.float32)}).to_parquet(SD / f"p1_{name}.parquet", index=False)
        del D
    spl, scorer, _, Q = _eval_setup()
    tau1, tune, rep, _ = baseline(scorer, spl, Q)
    # consistency check against the reference metric implementation
    m = Q.p_max.values >= tau1
    ref, _ = macro_f05(spl["tune_s1"], Q.q_row.values[m], Q.s_top.values[m], scorer.truth)
    assert abs(ref - tune["f05"]) < 1e-9, (ref, tune)
    json.dump({"tau1": float(tau1), "tune": tune, "report": rep}, open(SD / "baseline.json", "w"), indent=2)
    log(f"stage-1 baseline tau1={tau1:.2f}  TUNE {_fmt(tune)}  REPORT {_fmt(rep)}")


def _fmt(d):
    return f"F0.5={d['f05']:.4f} P={d['precision']:.4f} R={d['recall']:.4f}"


def _eval_setup():
    spl = np.load(SD / "split.npz")
    truth = np.load(config.WORK_DIR / "train_truth.npy")
    PE, Q = by_query(pd.read_parquet(SD / "p1_E.parquet"))
    n_s1 = int(max(truth.max(), PE.s1_row.max(), spl["eval_s1"].max())) + 1
    return spl, Scorer(truth, n_s1), PE, Q


def decide(Q, L, U, tau1, s2, p2, tau2):
    """Cascade assignment. s2/p2: per-query stage-2 choice (s2 = -1 where none)."""
    top = Q.p_max.values
    band = (top >= L) & (top < U)
    a1 = ~band & (top >= tau1)
    a2 = band & (s2 >= 0) & (p2 >= tau2)
    qr = Q.q_row.values
    return np.r_[qr[a1], qr[a2]], np.r_[Q.s_top.values[a1], s2[a2]], float(band.mean())


def band():
    """Oracle analysis: F0.5 if stage 2 were perfect inside [L, U) (assigns the true
    S1 when it is among the query's top-5 stage-1 candidates, else nothing)."""
    spl, scorer, PE, Q = _eval_setup()
    b = json.load(open(SD / "baseline.json"))
    hit = PE[(PE.p1_rank < TOP2) & (PE.y == 1)]
    s_orc = np.full(len(Q), -1, np.int64)
    s_orc[np.searchsorted(Q.q_row.values, hit.q_row.values)] = hit.s1_row.values
    rows = []
    for L in GRID_L:
        for U in GRID_U:
            qa, sa, share = decide(Q, L, U, b["tau1"], s_orc, np.ones(len(Q)), 0.5)
            rows.append(dict(L=L, U=U, band_share=share, **scorer.score(spl["tune_s1"], qa, sa)))
    R = pd.DataFrame(rows)
    R["gain"] = R.f05 - b["tune"]["f05"]
    R.to_csv(SD / "band_oracle.csv", index=False)
    full = R.gain.max()
    log(f"baseline TUNE F0.5 {b['tune']['f05']:.4f}; oracle max gain {full:+.4f} "
        f"(stage-2 ceiling given top-{TOP2} candidates)")
    print(R.pivot(index="L", columns="U", values="gain").round(4).to_string())
    print("band share of eval queries:")
    print(R.pivot(index="L", columns="U", values="band_share").round(3).to_string())


# ------------------------------------------------------------------ stage-2 method features
_W = {}


def _winit(idf_n, idf_a, idf_max):
    _W.update(n=idf_n, a=idf_a, max=idf_max)


def _jw_matrix(a, b):
    return process.cdist(a, b, scorer=JaroWinkler.normalized_similarity, dtype=np.float32)


def _monge_elkan(a, b):
    """Mean over tokens of a of the best JW match in b."""
    if not a or not b:
        return np.nan
    return float(_jw_matrix(a, b).max(1).mean())


def _soft_tfidf(a, b, idf, dflt, theta=0.9):
    """Cohen, Ravikumar & Fienberg (2003): sum over w in a with a close token v in b
    (JW >= theta) of V(w,a) * V(v,b) * JW(w,v), with V = L2-normalized IDF weights."""
    if not a or not b:
        return np.nan
    a, b = list(dict.fromkeys(a)), list(dict.fromkeys(b))
    wa = np.array([idf.get(t, dflt) for t in a])
    wb = np.array([idf.get(t, dflt) for t in b])
    na, nb = np.linalg.norm(wa), np.linalg.norm(wb)
    if na == 0 or nb == 0:
        return 0.0
    m = _jw_matrix(a, b)
    j = m.argmax(1)
    d = m[np.arange(len(a)), j]
    ok = d >= theta
    return float((wa[ok] / na * wb[j[ok]] / nb * d[ok]).sum())


def _method_chunk(args):
    method, n1, n2, a1, a2 = args
    out = np.empty((len(n1), 4), np.float32)
    for i in range(len(n1)):
        tn1, tn2, ta1, ta2 = n1[i].split(), n2[i].split(), a1[i].split(), a2[i].split()
        if method == "me":
            out[i] = (_monge_elkan(tn1, tn2), _monge_elkan(tn2, tn1), _monge_elkan(ta1, ta2), _monge_elkan(ta2, ta1))
        else:
            n, a, m = _W["n"], _W["a"], _W["max"]
            out[i] = (_soft_tfidf(tn1, tn2, n, m), _soft_tfidf(tn2, tn1, n, m),
                      _soft_tfidf(ta1, ta2, a, m), _soft_tfidf(ta2, ta1, a, m))
    return out


def method_features(method, keys, s1, q, pool):
    """Extra stage-2 features for pairs `keys` (q_row, s1_row). Row order preserved."""
    ia, ib = keys.s1_row.values, keys.q_row.values
    g = lambda df, c, idx: df[c].values[idx].tolist()
    n1, n2 = g(s1, "name_core", ia), g(q, "name_core", ib)
    a1, a2 = g(s1, "addr_norm", ia), g(q, "addr_norm", ib)
    a_empty = np.array([not s for s in a2])
    if method == "dl":
        f = {"dl_name": _cp(n1, n2, DamerauLevenshtein.normalized_similarity),
             "dl_sq": _cp(g(s1, "name_sq", ia), g(q, "name_sq", ib), DamerauLevenshtein.normalized_similarity),
             "dl_addr": _cp(a1, a2, DamerauLevenshtein.normalized_similarity)}
        f["dl_addr"][a_empty] = np.nan
        return pd.DataFrame(f)
    step = 20_000
    jobs = [(method, n1[j:j + step], n2[j:j + step], a1[j:j + step], a2[j:j + step]) for j in range(0, len(n1), step)]
    M = np.vstack(pool.map(_method_chunk, jobs))
    names = [f"{method}_n_s1q", f"{method}_n_qs1", f"{method}_a_s1q", f"{method}_a_qs1"]
    return pd.DataFrame(M, columns=names)


def _ce_text(df, idx):
    return [f"{n} | {a}" for n, a in zip(df.name_full.values[idx], df.addr_norm.values[idx])]


def _ce_fit_predict(tr_a, tr_b, tr_y, pred_sets, bs=64, lr=3e-5, max_len=96):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    if not torch.cuda.is_available():
        raise SystemExit("cross-encoder needs a CUDA build of torch (installed: %s)" % torch.__version__)
    torch.manual_seed(config.SEED)
    tok = AutoTokenizer.from_pretrained(CE_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(CE_BASE, num_labels=1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = math.ceil(len(tr_y) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    order = np.random.default_rng(config.SEED).permutation(len(tr_y))
    enc = lambda a, b: tok(a, b, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")
    model.train()
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        x = enc([tr_a[k] for k in idx], [tr_b[k] for k in idx])
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(**x).logits.squeeze(-1)
        loss = lossf(logit.float(), torch.tensor(tr_y[idx], dtype=torch.float32, device="cuda"))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if (i // bs) % 1000 == 0:
            log(f"  ce step {i // bs}/{steps} loss {loss.item():.4f}")
    model.eval()
    outs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for a, b in pred_sets:
            o = np.empty(len(a), np.float32)
            for i in range(0, len(a), 512):
                o[i:i + 512] = model(**enc(a[i:i + 512], b[i:i + 512])).logits.squeeze(-1).float().cpu().numpy()
            outs.append(o)
    return outs


def ce_features(PB, PE, s1, q, truth):
    """Cross-encoder logit, 2-fold cross-fitted on B (folds split by true S1 entity)."""
    fold_s1 = np.random.default_rng(config.SEED + 3).integers(0, 2, len(s1))
    tq = truth[PB.q_row.values]
    fold = np.where(tq >= 0, fold_s1[np.maximum(tq, 0)], PB.q_row.values % 2)
    B_a, B_b = _ce_text(s1, PB.s1_row.values), _ce_text(q, PB.q_row.values)
    E_a, E_b = _ce_text(s1, PE.s1_row.values), _ce_text(q, PE.q_row.values)
    ce_B = np.empty(len(PB), np.float32)
    ce_E = np.zeros(len(PE), np.float32)
    rng = np.random.default_rng(config.SEED + 4)
    for k in (0, 1):
        tr = np.flatnonzero(fold == k)
        if len(tr) > CE_MAX_TRAIN:
            tr = np.sort(rng.choice(tr, CE_MAX_TRAIN, replace=False))
        te = np.flatnonzero(fold != k)
        log(f"ce fold {k}: train {len(tr):,} pairs, predict {len(te):,} + eval {len(PE):,}")
        pb, pe = _ce_fit_predict([B_a[i] for i in tr], [B_b[i] for i in tr], PB.y.values[tr].astype(np.float32),
                                 [([B_a[i] for i in te], [B_b[i] for i in te]), (E_a, E_b)])
        ce_B[te] = pb
        ce_E += pe / 2
    return pd.DataFrame({"ce": ce_B}), pd.DataFrame({"ce": ce_E})


def band_pairs(name):
    """Top-5 stage-1 candidates of every query whose p_max lies in the wide band."""
    P, Q = by_query(pd.read_parquet(SD / f"p1_{name}.parquet"))
    inq = Q.q_row.values[(Q.p_max.values >= WIDE[0]) & (Q.p_max.values < WIDE[1])]
    return P[np.isin(P.q_row.values, inq) & (P.p1_rank < TOP2)].reset_index(drop=True)


def stage2(method):
    PB, PE = band_pairs("B"), band_pairs("E")
    log(f"stage2[{method}] band pairs: B {len(PB):,} ({PB.q_row.nunique():,} q), "
        f"E {len(PE):,} ({PE.q_row.nunique():,} q)")
    extra = ["p1", "p1_rank", "p1_gap"]
    FB = PB.merge(load_feat("B").drop(columns="y"), on=["q_row", "s1_row"], how="left")
    FE = PE.merge(load_feat("E").drop(columns="y"), on=["q_row", "s1_row"], how="left")
    cols = json.load(open(SD / "stage1_meta.json"))["features"] + extra
    if method != "control":
        cache = [SD / f"m_{method}_{n}.parquet" for n in ("B", "E")]
        if all(c.exists() for c in cache):
            MB, ME = (pd.read_parquet(c) for c in cache)
        else:
            s1, q = load_tables("train")
            if method == "ce":
                MB, ME = ce_features(PB, PE, s1, q, np.load(config.WORK_DIR / "train_truth.npy"))
            else:
                for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                    os.environ[v] = "1"  # one BLAS thread per worker, as in features.make_pool
                with Pool(config.N_JOBS, initializer=_winit, initargs=build_idf(s1)) as pool:
                    MB = method_features(method, PB, s1, q, pool)
                    ME = method_features(method, PE, s1, q, pool)
            MB.to_parquet(cache[0], index=False)
            ME.to_parquet(cache[1], index=False)
        FB = pd.concat([FB, MB], axis=1)
        FE = pd.concat([FE, ME], axis=1)
        cols += list(MB.columns)
        log(f"method features: {list(MB.columns)}")

    rng = np.random.default_rng(config.SEED + 5)
    uq = np.unique(FB.q_row.values)
    is_dev = np.isin(FB.q_row.values, rng.choice(uq, len(uq) // 10, replace=False))
    dtr = xgb.DMatrix(FB.loc[~is_dev, cols], label=FB.y.values[~is_dev])
    ddv = xgb.DMatrix(FB.loc[is_dev, cols], label=FB.y.values[is_dev])
    bst = xgb.train(dict(S2_PARAMS, device=device(), seed=config.SEED), dtr, 3000, evals=[(ddv, "dev")],
                    early_stopping_rounds=50, verbose_eval=False)
    bst.save_model(SD / f"stage2_{method}.json")
    FE["p2"] = bst.predict(xgb.DMatrix(FE[cols]), iteration_range=(0, bst.best_iteration + 1))
    log(f"stage2[{method}] best_iter={bst.best_iteration}")
    imp = sorted(bst.get_score(importance_type="gain").items(), key=lambda x: -x[1])[:8]

    spl, scorer, _, Q = _eval_setup()
    b = json.load(open(SD / "baseline.json"))
    best2 = FE.sort_values(["q_row", "p2"], ascending=[True, False]).drop_duplicates("q_row")
    s2 = np.full(len(Q), -1, np.int64)
    p2 = np.zeros(len(Q))
    pos = np.searchsorted(Q.q_row.values, best2.q_row.values)
    s2[pos], p2[pos] = best2.s1_row.values, best2.p2.values

    rows = []
    for L in GRID_L:
        for U in GRID_U:
            for t2 in GRID_T2:
                qa, sa, share = decide(Q, L, U, b["tau1"], s2, p2, t2)
                rows.append(dict(L=L, U=U, tau2=float(t2), band_share=share, **scorer.score(spl["tune_s1"], qa, sa)))
    R = pd.DataFrame(rows)
    R.to_csv(SD / f"sweep_{method}.csv", index=False)

    # two picks on TUNE: best F0.5, and best recall with precision within 0.002 of baseline
    pick_f = R.loc[R.f05.idxmax()]
    ok = R[R.precision >= b["tune"]["precision"] - 0.002]
    pick_r = ok.loc[ok.recall.idxmax()] if len(ok) else pick_f
    m0 = Q.p_max.values >= b["tau1"]
    f_base = scorer.per_entity(spl["report_s1"], Q.q_row.values[m0], Q.s_top.values[m0])[0]
    out = {"method": method, "features": cols, "best_iter": int(bst.best_iteration), "top_gain": imp,
           "baseline": b, "picks": {}}
    for tag, pk in (("best_f05", pick_f), ("recall_at_P", pick_r)):
        qa, sa, share = decide(Q, pk.L, pk.U, b["tau1"], s2, p2, pk.tau2)
        rep = scorer.score(spl["report_s1"], qa, sa)
        f_new = scorer.per_entity(spl["report_s1"], qa, sa)[0]
        d, lo, hi = bootstrap_delta(f_new, f_base)
        out["picks"][tag] = dict(L=float(pk.L), U=float(pk.U), tau2=float(pk.tau2), band_share=share,
                                 tune=dict(f05=pk.f05, precision=pk.precision, recall=pk.recall),
                                 report=rep, delta_f05=d, ci95=[lo, hi])
        log(f"stage2[{method}] {tag}: L={pk.L} U={pk.U} tau2={pk.tau2} band={share:.3f} "
            f"REPORT {_fmt(rep)}  dF0.5={d:+.4f} [{lo:+.4f},{hi:+.4f}]")
    json.dump(out, open(SD / f"results_{method}.json", "w"), indent=2, default=float)


def report():
    b = json.load(open(SD / "baseline.json"))
    rows = [dict(method="stage-1 only", L="", U="", tau2="", band="", **b["report"], dF="", ci="")]
    for f in sorted(SD.glob("results_*.json")):
        r = json.load(open(f))
        for tag, pk in r["picks"].items():
            rows.append(dict(method=f"{r['method']} ({tag})", L=pk["L"], U=pk["U"], tau2=pk["tau2"],
                             band=round(pk["band_share"], 3), **pk["report"], dF=round(pk["delta_f05"], 4),
                             ci=f"[{pk['ci95'][0]:+.4f}, {pk['ci95'][1]:+.4f}]"))
    T = pd.DataFrame(rows).rename(columns={"f05": "F0.5", "precision": "P", "recall": "R"})
    print(f"REPORT half ({N_EVAL_S1 // 2:,} held-out S1 entities); tau1={b['tau1']:.2f}")
    print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["split", "features", "stage1", "band", "stage2", "report", "all"])
    ap.add_argument("--method", default="control", choices=["control", "dl", "me", "stfidf", "ce"])
    a = ap.parse_args()
    SD.mkdir(exist_ok=True)
    if a.step == "all":
        split(); features(); stage1(); band()
        for m in ("control", "dl", "me", "stfidf"):
            stage2(m)
        report()
    elif a.step == "stage2":
        stage2(a.method)
    else:
        globals()[a.step]()
