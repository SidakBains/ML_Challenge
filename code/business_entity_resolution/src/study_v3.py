"""Study v3: recall upgrade of the two-stage cascade, aimed at empty-address queries.

Same fixed split and TUNE / REPORT protocol as study.py (it reuses study/split.npz), so
every number compares directly with results.txt.

What changes (numbers refer to the ideas of the recall analysis):
  1  candidates  full v2 union instead of rank < 10; stage 2 re-scores the stage-1 top 8 (was 5)
  3,7,8 blocking four extra searches for the study queries, OR-ed into `found`:
                    8  dense  fine-tuned MiniLM bi-encoder on 'name | address', GPU top-k
                   16  wide   empty-address queries only: name search with relaxed df-pruning
                   32  fuzzy  confusable-folded name tokens (0/o 1/l 5/s 8/b i/l ...), 1-deletion
                              variants, unordered token pairs, metaphone codes + address char 4-grams
                   64  key    exact order-free name key; the key group is re-ranked by address cosine
                 S1-side context (s1_ncand, s1_n_top1, s1_max, s1_rank) always comes from the base
                 v2 pass over ALL queries, so the extra searches never shift it (same at test time).
     stage 1     + name-ambiguity counts (S1 records sharing the name / key), name-space rank and
                 margin, dense and fuzzy cosines, extra found bits
  6  stage 2     cross-encoder on the top 8, twice the training pairs with empty-address queries
                 oversampled x3, rival features (CE rank / gap / margin / softmax)
  4  collective  sibling features: the other queries whose top blocking candidate is this S1
                 (count, same-source count, best name / address similarity to the query)
  2  decision    thresholds tuned separately for empty- and non-empty-address queries
  5  decision    expected-F0.5 assignment per S1 entity (coordinate ascent on the probabilities)

Usage (each step caches to WORK_DIR/study_v3; STUDY_V3_DEBUG=1 runs a small smoke test):
  python study_v3.py dense | cands | recall | features | stage1 | ablation | stage2 | decide | fn | report
"""
import argparse
import json
import math
import os
import time
from itertools import combinations

import jellyfish
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
import xgboost as xgb
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2norm
from sparse_dot_topn import sp_matmul_topn

import config
from blocking import CHANNELS, SPACES, ChannelModel, _rowdot, cands_path
from features import build_idf, compute_features, make_pool
from study import (GRID_L, GRID_T1, GRID_T2, GRID_U, S2_PARAMS, SD, Scorer, bootstrap_delta, by_query,
                   decide, log)
from train import XGB_PARAMS, device

DEBUG = os.environ.get("STUDY_V3_DEBUG") == "1"
V3 = config.WORK_DIR / ("study_v3_dbg" if DEBUG else "study_v3")
DENSE_DIR = config.WORK_DIR / "study_v3"   # bi-encoder artifacts are shared with the debug run
TOP2 = 8                    # stage 2 re-scores this many stage-1 candidates per query
WIDE = (0.01, 0.995)        # band stage 2 is trained on (as in study.py)
K_DENSE = (10, 30)          # top-k (address present, address empty)
K_FUZZY = (10, 30)
K_WIDE = 50                 # empty-address queries only
K_KEY = 5                   # key-group members kept after address re-ranking
KEY_MAX_GROUP = 300         # larger name-key groups are not enumerated
SIB_MIN = 0.5               # blocking score for a query's top candidate to make it a sibling
SIB_CAP = 50                # siblings kept per S1 (highest blocking scores)
BITS = {"comb": 1, "name": 2, "addr": 4, "dense": 8, "wide": 16, "fuzzy": 32, "key": 64}
Q_CHUNK = 200_000
FEAT_CHUNK = 3_000_000
BI_BASE = "sentence-transformers/all-MiniLM-L6-v2"
BI_TRAIN = 5_000 if DEBUG else 400_000
BI_MAXLEN = 48
CE_BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_MAX_TRAIN = 3_000 if DEBUG else 600_000   # per fold (study.py used 300k)
EMPTY_W = 3.0               # oversampling weight of empty-address queries (bi-encoder, cross-encoder)
N_DEBUG = 30_000


# ------------------------------------------------------------------ split / io helpers
def study_split(n_q):
    """Reuse study.py's split. split_of[q_row] = 0 (A), 1 (B), 2 (E = eval) or -1."""
    spl = np.load(SD / "split.npz")
    parts = {"A": spl["qa"], "B": spl["qb"], "E": spl["eval_q"]}
    if DEBUG:
        rng = np.random.default_rng(0)
        parts = {k: np.sort(rng.choice(v, min(len(v), N_DEBUG), replace=False)) for k, v in parts.items()}
    split_of = np.full(n_q, -1, np.int8)
    for i, k in enumerate("ABE"):
        split_of[parts[k]] = i
    return spl, parts, split_of


def tables(cols):
    """blocking.load_tables restricted to some columns (same row order; q gets `src`)."""
    s1 = pd.read_parquet(config.prep_path("train", 1), columns=cols)
    parts = []
    for k in (2, 3):
        d = pd.read_parquet(config.prep_path("train", k), columns=cols)
        d["src"] = np.int8(k)
        parts.append(d)
    return s1, pd.concat(parts, ignore_index=True)


def read_split(prefix, name, columns=None):
    files = sorted(V3.glob(f"{prefix}_{name}_*.parquet"))
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


class Writer:
    """Append DataFrames to one parquet file per key (streamed, memory-flat)."""

    def __init__(self, pattern):
        self.pattern, self.w = pattern, {}

    def write(self, key, df):
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if key not in self.w:
            self.w[key] = pq.ParquetWriter(V3 / self.pattern.format(key), tbl.schema)
        self.w[key].write_table(tbl)

    def close(self):
        for w in self.w.values():
            w.close()


def group_first(keys):
    """Start index of each run of equal values in a sorted array, and run sizes."""
    first = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    return first, np.diff(np.r_[first, len(keys)])


def within_rank(grp, val):
    """For rows sorted by grp: rank of val (descending) inside each group, the group's best
    and second-best value, and the group size. Returns arrays in the input row order."""
    order = np.lexsort((-val, grp))
    g, v = grp[order], val[order]
    first, sizes = group_first(g)
    start = np.repeat(first, sizes)
    rank = np.empty(len(g), np.int32)
    rank[order] = np.arange(len(g)) - start
    top = np.empty(len(g), np.float32)
    top[order] = v[start]
    sec = np.empty(len(g), np.float32)
    sec[order] = np.where(np.repeat(sizes, sizes) > 1, v[np.minimum(start + 1, len(g) - 1)], 0)
    size = np.empty(len(g), np.int32)
    size[order] = np.repeat(sizes, sizes)
    return rank, top, sec, size


# ------------------------------------------------------------------ fuzzy name keys
_DIG = str.maketrans("01345789", "oleastbg")


def fold(tok):
    """Undo the OCR-style confusions of S2/S3 names: digits inside words -> letters, i -> l."""
    if not tok.isdigit():
        tok = tok.translate(_DIG)
    return tok.replace("i", "l")


def name_key(core):
    """Order-free, confusion-folded name key ('eye care optimal' == 'eye optimal care')."""
    return " ".join(sorted(set(fold(t) for t in core.split())))


def fuzzy_doc(core):
    """Tokens of the fuzzy name channel: each folded token with its 1-deletion variants
    (two names within ~2 edits share a variant), its metaphone code, and unordered token
    pairs (common-word names stay searchable after df-pruning)."""
    toks = sorted(set(fold(t) for t in core.split()))[:8]
    out = []
    for t in toks:
        out.append("d:" + t)
        if len(t) >= 5:
            out.extend("d:" + t[:i] + t[i + 1:] for i in range(len(t)))
        if len(t) >= 3 and t.isalpha():
            out.append("m:" + jellyfish.metaphone(t))
    out.extend(f"p:{a}|{b}" for a, b in combinations(toks, 2))
    return " ".join(out)


def _fuzzy_docs_chunk(cores):
    return [fuzzy_doc(c) for c in cores]


def fuzzy_docs(cores, pool):
    step = 50_000
    return [d for part in pool.map(_fuzzy_docs_chunk, [cores[i:i + step] for i in range(0, len(cores), step)])
            for d in part]


class HashedChannels:
    """TF-IDF over hashed tokens (no vocabulary dict), df-pruned like blocking.ChannelModel;
    kept columns are compacted so the transposed S1 matrix stays small."""
    CH = {"f": (dict(token_pattern=r"\S+"), 1.0, 2 ** 26),
          "g": (dict(analyzer="char_wb", ngram_range=(4, 4)), 0.5, 2 ** 22)}

    def __init__(self, docs, max_df_frac=0.003, min_max_df=2000):
        n = len(docs["f"])
        max_df = max(min_max_df, int(max_df_frac * n))
        self.vec, self.used, self.idf, self.S = {}, {}, {}, {}
        for ch, (kw, w, nf) in self.CH.items():
            vec = HashingVectorizer(n_features=nf, lowercase=False, alternate_sign=False, norm=None,
                                    binary=True, dtype=np.float32, **kw)
            X = vec.transform(docs[ch]).tocsr()
            df = np.bincount(X.indices, minlength=nf)
            used = np.flatnonzero((df > 0) & (df <= max_df))
            self.vec[ch], self.used[ch] = vec, used
            self.idf[ch] = ((np.log((n + 1) / (df[used] + 1)) + 1) * w).astype(np.float32)
            self.S[ch] = self._compact(X, ch)

    def _compact(self, X, ch):
        X = X.tocsr()
        used, idf = self.used[ch], self.idf[ch]
        j = np.minimum(np.searchsorted(used, X.indices), len(used) - 1)
        ok = used[j] == X.indices
        Y = sp.csr_matrix((X.data * np.where(ok, idf[j], 0), np.where(ok, j, 0), X.indptr),
                          shape=(X.shape[0], len(used)))
        Y.eliminate_zeros()
        return Y

    def transform(self, docs):
        return {ch: self._compact(self.vec[ch].transform(docs[ch]), ch) for ch in self.vec}

    @staticmethod
    def space(mats):
        return l2norm(sp.hstack([mats["f"], mats["g"]]).tocsr()).astype(np.float32)


def _search_rk(Mq, MsT, k):
    """Top-k sparse cosine search; returns (q, s, score, rank-within-query)."""
    R = sp_matmul_topn(Mq, MsT, top_n=k, threshold=1e-4, sort=True, n_threads=config.N_JOBS).tocsr()
    sizes = np.diff(R.indptr)
    qi = np.repeat(np.arange(R.shape[0], dtype=np.int64), sizes)
    rk = np.arange(R.nnz) - np.repeat(R.indptr[:-1], sizes)
    return qi, R.indices.astype(np.int64), R.data.astype(np.float32), rk


# ------------------------------------------------------------------ step: dense (bi-encoder)
def _bi_text(df, idx):
    return [f"{n} | {a}" for n, a in zip(df.name_full.values[idx], df.addr_norm.values[idx])]


def _encode(model, tok, texts, maxlen=BI_MAXLEN):
    import torch
    x = tok(texts, truncation=True, max_length=maxlen, padding=True, return_tensors="pt").to("cuda")
    h = model(**x).last_hidden_state
    m = x["attention_mask"].unsqueeze(-1).to(h.dtype)
    return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-6), dim=-1)


def _embed_all(model, tok, texts, bs=1024):
    import torch
    order = np.argsort(np.fromiter((len(t) for t in texts), np.int32, len(texts)), kind="stable")
    out = np.empty((len(texts), model.config.hidden_size), np.float16)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            out[idx] = _encode(model, tok, [texts[k] for k in idx]).half().cpu().numpy()
    return out


def hard_negatives(qrows, truth, rng):
    """Best-ranked wrong v2 candidate of each query (random S1 row if there is none)."""
    sel = np.zeros(len(truth), bool)
    sel[qrows] = True
    best = np.full(len(truth), -1, np.int64)
    for f in sorted(config.WORK_DIR.glob("train_v2_cands_*.parquet")):
        c = pd.read_parquet(f, columns=["q_row", "s1_row", "rank"])
        c = c[sel[c.q_row.values]]
        c = c[c.s1_row.values != truth[c.q_row.values]]
        c = c.sort_values(["q_row", "rank"], kind="stable").drop_duplicates("q_row")
        best[c.q_row.values] = c.s1_row.values
    neg = best[qrows]
    miss = neg < 0
    neg[miss] = rng.integers(0, truth.max() + 1, miss.sum())
    return neg


def dense():
    """Fine-tune the bi-encoder on queries outside the study (true S1 not used by A, B or eval),
    embed every S1 record and the study queries, and search each query's country on the GPU."""
    import torch
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    s1, q = tables(["country", "name_full", "addr_norm"])
    truth = np.load(config.WORK_DIR / "train_truth.npy")
    spl, parts, split_of = study_split(len(q))
    busy = np.zeros(len(s1), bool)
    busy[spl["eval_s1"]] = True
    for k in ("A", "B"):
        t = truth[parts[k]]
        busy[t[t >= 0]] = True
    empty = q.addr_norm.values == ""
    pool = np.flatnonzero((truth >= 0) & (split_of < 0) & ~busy[np.maximum(truth, 0)])
    rng = np.random.default_rng(config.SEED + 10)
    w = np.where(empty[pool], EMPTY_W, 1.0)
    # weighted sampling without replacement (Gumbel top-k / Efraimidis-Spirakis keys)
    tr = np.sort(pool[np.argsort(-(np.log(w) + rng.gumbel(size=len(pool))))[:min(BI_TRAIN, len(pool))]])
    neg = hard_negatives(tr, truth, rng)
    log(f"bi-encoder: pool {len(pool):,} queries, training on {len(tr):,} (empty-address {empty[tr].mean():.3f})")
    anc, pos, ngt = _bi_text(q, tr), _bi_text(s1, truth[tr]), _bi_text(s1, neg)

    torch.manual_seed(config.SEED)
    tok = AutoTokenizer.from_pretrained(BI_BASE)
    model = AutoModel.from_pretrained(BI_BASE).cuda()
    bs, lr = 128, 5e-5
    steps = len(tr) // bs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler()
    order = rng.permutation(len(tr))
    model.train()
    for st in range(steps):
        idx = order[st * bs:(st + 1) * bs]
        with torch.autocast("cuda", dtype=torch.float16):
            a = _encode(model, tok, [anc[k] for k in idx])
            b = _encode(model, tok, [pos[k] for k in idx] + [ngt[k] for k in idx])
        sim = a.float() @ b.float().T * 20.0
        loss = torch.nn.functional.cross_entropy(sim, torch.arange(len(idx), device="cuda"))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if st % 500 == 0:
            log(f"  bi step {st}/{steps} loss {loss.item():.4f}")
    del anc, pos, ngt

    qrows = np.flatnonzero(split_of >= 0)
    t = time.time()
    emb_s1 = _embed_all(model, tok, _bi_text(s1, np.arange(len(s1))))
    np.save(DENSE_DIR / "emb_s1.npy", emb_s1)
    emb_q = _embed_all(model, tok, _bi_text(q, qrows))
    np.save(DENSE_DIR / "emb_q.npy", emb_q)
    np.save(DENSE_DIR / "emb_qrows.npy", qrows)
    log(f"embedded {len(s1):,} S1 + {len(qrows):,} queries in {time.time() - t:.0f}s")
    del model
    torch.cuda.empty_cache()

    out = []
    kmax = max(K_DENSE)
    for country in sorted(s1.country.unique()):
        s_idx = np.flatnonzero(s1.country.values == country)
        loc = np.flatnonzero(q.country.values[qrows] == country)
        Es = torch.from_numpy(emb_s1[s_idx]).cuda()
        for i in range(0, len(loc), 512):
            li = loc[i:i + 512]
            Eq = torch.from_numpy(emb_q[li]).cuda()
            val, ind = torch.topk(Eq @ Es.T, min(kmax, len(s_idx)), dim=1)
            val, ind = val.float().cpu().numpy(), ind.cpu().numpy()
            k = np.where(empty[qrows[li]], K_DENSE[1], K_DENSE[0])
            keep = np.arange(val.shape[1])[None, :] < k[:, None]
            rk = np.broadcast_to(np.arange(val.shape[1])[None, :], val.shape)
            out.append(pd.DataFrame({"q_row": np.repeat(qrows[li], val.shape[1])[keep.ravel()].astype(np.int32),
                                     "s1_row": s_idx[ind[keep]].astype(np.int32),
                                     "dense_cos": val[keep].astype(np.float32),
                                     "dense_rank": rk[keep].astype(np.int16)}))
        del Es
        torch.cuda.empty_cache()
        log(f"dense search {country}: {len(loc):,} queries")
    D = pd.concat(out, ignore_index=True)
    D.to_parquet(DENSE_DIR / "dense.parquet", index=False)
    hit = D[truth[D.q_row.values] == D.s1_row.values]
    tq = truth[qrows] >= 0
    log(f"dense candidates {len(D):,}; recall of study true links {hit.q_row.nunique() / tq.sum():.4f}")


# ------------------------------------------------------------------ step: cands (v3 union)
CAND_COLS = ["country", "name_core", "name_sq", "addr_norm", "addr_nums"]
SKEY_BINS = 512  # s1_rank resolution: scores quantized to 1/511 so the (s1, score) key fits int32


def _skey(s1_row, score):
    return (s1_row.astype(np.int64) * SKEY_BINS
            + (SKEY_BINS - 1) - np.floor(np.clip(score, 0, 1) * (SKEY_BINS - 1)).astype(np.int64))


def base_pass(country, split_of, n_s1, stats):
    """Stream the base v2 candidates of ALL queries of a country: S1-side context (added to
    `stats`), the sorted (s1, score) key for s1_rank, the study queries' base rows, siblings."""
    skeys, study, sib = [], [], []
    pf = pq.ParquetFile(cands_path("train", "v2", country))
    for batch in pf.iter_batches(batch_size=10_000_000, columns=["q_row", "s1_row", "score", "rank", "found"]):
        qr, s, sc, rk = (batch.column(c).to_numpy() for c in ("q_row", "s1_row", "score", "rank"))
        stats["ncand"] += np.bincount(s, minlength=n_s1).astype(np.int32)
        stats["ntop1"] += np.bincount(s[rk == 0], minlength=n_s1).astype(np.int32)
        np.maximum.at(stats["max"], s, np.clip(sc, 0, 1).astype(np.float32))
        skeys.append(_skey(s, sc).astype(np.int32))
        m = split_of[qr] >= 0
        study.append(pd.DataFrame({"q_row": qr[m], "s1_row": s[m], "found": batch.column("found").to_numpy()[m],
                                   "v2_rank": rk[m]}))
        m = (rk == 0) & (sc >= SIB_MIN)
        sib.append(pd.DataFrame({"q_row": qr[m], "s1_row": s[m], "score": sc[m]}))
    skey = np.concatenate(skeys)
    del skeys
    skey.sort()
    B0 = pd.concat(study, ignore_index=True).sort_values(["q_row", "v2_rank"], kind="stable").reset_index(drop=True)
    return skey, B0, pd.concat(sib, ignore_index=True)


def cands():
    """v3 candidate table for the study queries (see module docstring), one parquet per
    (split, country), with every context column the features need precomputed."""
    import torch
    t0 = time.time()
    s1, q = tables(CAND_COLS)
    spl, parts, split_of = study_split(len(q))
    empty_q = q.addr_norm.values == ""
    n_s1 = len(s1)
    D = pd.read_parquet(DENSE_DIR / "dense.parquet")
    D = D[split_of[D.q_row.values] >= 0].sort_values(["q_row", "dense_rank"], kind="stable").reset_index(drop=True)
    emb_s1 = np.load(DENSE_DIR / "emb_s1.npy", mmap_mode="r")
    emb_q = np.load(DENSE_DIR / "emb_q.npy", mmap_mode="r")
    emb_qrows = np.load(DENSE_DIR / "emb_qrows.npy")
    stats = {"ncand": np.zeros(n_s1, np.int32), "ntop1": np.zeros(n_s1, np.int32), "max": np.zeros(n_s1, np.float32)}
    sib, out = [], Writer("cands_{}.parquet")
    for country in sorted(s1.country.unique()):
        tc = time.time()
        s_idx = np.flatnonzero(s1.country.values == country)
        S = s1.iloc[s_idx].reset_index(drop=True)
        pos_s = np.full(n_s1, -1, np.int64)
        pos_s[s_idx] = np.arange(len(s_idx))
        qs = np.flatnonzero((split_of >= 0) & (q.country.values == country))
        Dc = D[q.country.values[D.q_row.values] == country].reset_index(drop=True)  # q rows interleave countries
        skey, B0, sb = base_pass(country, split_of, n_s1, stats)
        sib.append(sb)
        log(f"[{country}] base pass: {len(skey):,} pairs, {len(B0):,} study pairs [{time.time() - tc:.0f}s]")

        # wide name search (relaxed df-pruning) for all empty-address study queries, then free it
        e_q = qs[empty_q[qs]]
        cmw = ChannelModel(S, {c: CHANNELS["v2"][c] for c in "nc"}, max_df_frac=0.05)
        qi, si, _, _ = _search_rk(cmw.space(cmw.transform(q.iloc[e_q]), "nc"), cmw.space(cmw.S, "nc").T.tocsr(),
                                  K_WIDE)
        Wd = pd.DataFrame({"q_row": e_q[qi], "ls": si})
        del cmw
        log(f"[{country}] wide search: {len(e_q):,} empty-address queries -> {len(Wd):,} pairs "
            f"[{time.time() - tc:.0f}s]")

        # per-country index structures
        cm = ChannelModel(S, CHANNELS["v2"])
        Ms = {name: cm.space(cm.S, chans) for name, chans in SPACES.items()}
        cm.S = None
        fz = HashedChannels({"f": [fuzzy_doc(c) for c in S.name_core.values], "g": S.addr_norm.tolist()})
        Msf = fz.space(fz.S)
        fz.S = None
        MsfT = Msf.T.tocsr()
        kcode, kuniq = pd.factorize(np.array([name_key(c) for c in S.name_core.values], dtype=object))
        ksize = np.bincount(kcode)
        korder = np.argsort(kcode, kind="stable")
        kstart = np.searchsorted(kcode[korder], np.arange(len(kuniq)))
        kindex = pd.Index(kuniq)
        ncode, nuniq = pd.factorize(S.name_core.values)
        nsize = np.bincount(ncode)
        nindex = pd.Index(nuniq)
        Es = torch.from_numpy(np.ascontiguousarray(emb_s1[s_idx])).cuda()
        log(f"[{country}] index built: s1={len(s_idx):,} fuzzy cols f={len(fz.used['f']):,} "
            f"g={len(fz.used['g']):,} key groups={len(kuniq):,} [{time.time() - tc:.0f}s]")

        for j in range(0, len(qs), Q_CHUNK):
            tj = time.time()
            qc = qs[j:j + Q_CHUNK]
            Qdf = q.iloc[qc]
            emp = empty_q[qc]
            qm = cm.transform(Qdf)
            Mq = {name: cm.space(qm, chans) for name, chans in SPACES.items()}
            del qm
            Mqf = fz.space(fz.transform({"f": [fuzzy_doc(c) for c in Qdf.name_core.values],
                                         "g": Qdf.addr_norm.tolist()}))
            P = []  # (local query, local s1, bit)
            # fuzzy search
            qi, si, _, rk = _search_rk(Mqf, MsfT, K_FUZZY[1])
            keep = rk < np.where(emp[qi], K_FUZZY[1], K_FUZZY[0])
            P.append((qi[keep], si[keep], BITS["fuzzy"]))
            # wide search results of this chunk
            a, b = np.searchsorted(Wd.q_row.values, [qc[0], qc[-1] + 1])
            P.append((np.searchsorted(qc, Wd.q_row.values[a:b]), Wd.ls.values[a:b], BITS["wide"]))
            # name-key block, re-ranked by address cosine
            qk = kindex.get_indexer(np.array([name_key(c) for c in Qdf.name_core.values], dtype=object))
            size = np.where(qk >= 0, ksize[np.maximum(qk, 0)], 0)
            tq = np.flatnonzero((qk >= 0) & (size <= KEY_MAX_GROUP) & (~emp | (size <= K_KEY)))
            if len(tq):
                n_e = size[tq]
                rq = np.repeat(tq, n_e)
                offs = np.arange(n_e.sum()) - np.repeat(np.cumsum(n_e) - n_e, n_e)
                rs = korder[np.repeat(kstart[qk[tq]], n_e) + offs]
                ac = _rowdot(Mq["addr"], Ms["addr"], rq, rs)
                rank_k = within_rank(rq, ac)[0]
                P.append((rq[rank_k < K_KEY], rs[rank_k < K_KEY], BITS["key"]))
            # dense candidates
            a, b = np.searchsorted(Dc.q_row.values, [qc[0], qc[-1] + 1])
            P.append((np.searchsorted(qc, Dc.q_row.values[a:b]), pos_s[Dc.s1_row.values[a:b]], BITS["dense"]))
            # base v2 rows
            a, b = np.searchsorted(B0.q_row.values, [qc[0], qc[-1] + 1])
            b0 = B0.iloc[a:b]
            P.append((np.searchsorted(qc, b0.q_row.values), pos_s[b0.s1_row.values], b0.found.values.astype(np.int8)))

            # --- union of all searches
            lq = np.concatenate([p[0] for p in P]).astype(np.int64)
            ls = np.concatenate([p[1] for p in P]).astype(np.int64)
            bits = np.concatenate([np.full(len(p[0]), p[2], np.int8) if np.isscalar(p[2]) else p[2]
                                   for p in P]).astype(np.int8)
            assert (ls >= 0).all()
            key = lq * len(s_idx) + ls
            uk, inv = np.unique(key, return_inverse=True)
            found = np.zeros(len(uk), np.int8)
            np.bitwise_or.at(found, inv, bits)
            lq, ls = uk // len(s_idx), uk % len(s_idx)
            gs = s_idx[ls]
            v2r = np.full(len(uk), -1, np.int16)
            bkey = np.searchsorted(qc, b0.q_row.values).astype(np.int64) * len(s_idx) + pos_s[b0.s1_row.values]
            v2r[np.searchsorted(uk, bkey)] = b0.v2_rank.values
            del key, inv, bits, P

            # --- cosines in every space
            cos = {name: _rowdot(Mq[name], Ms[name], lq, ls) for name in SPACES}
            fcos = _rowdot(Mqf, Msf, lq, ls)
            Eq = torch.from_numpy(np.ascontiguousarray(emb_q[np.searchsorted(emb_qrows, qc)])).cuda()
            dcos = np.empty(len(lq), np.float32)
            for i in range(0, len(lq), 2_000_000):
                a_ = torch.from_numpy(lq[i:i + 2_000_000]).cuda()
                b_ = torch.from_numpy(ls[i:i + 2_000_000]).cuda()
                dcos[i:i + 2_000_000] = (Eq[a_].float() * Es[b_].float()).sum(1).cpu().numpy()
            del Eq, Mq, Mqf

            # --- query-side context (comb score, name space, dense space) + name ambiguity
            sc = cos["comb"]
            rank, top1, second, qn = within_rank(lq, sc)
            nrank, ntop, nsec, _ = within_rank(lq, cos["name"])
            drank, dtop, dsec, _ = within_rank(lq, dcos)
            skc = kcode[ls]
            _, kinv, kcnt = np.unique(lq * (len(kuniq) + 1) + skc, return_inverse=True, return_counts=True)
            qn_code = nindex.get_indexer(Qdf.name_core.values)
            C = pd.DataFrame({
                "q_row": qc[lq].astype(np.int32), "s1_row": gs.astype(np.int32), "found": found,
                "v2_rank": v2r, "score": sc, "name_cos": cos["name"], "addr_cos": cos["addr"],
                "fuzzy_cos": fcos, "dense_cos": dcos,
                "rank": rank.astype(np.int16), "q_ncand": qn.astype(np.int16), "top1": top1,
                "gap_top1": top1 - sc, "margin": np.where(rank == 0, sc - second, sc - top1).astype(np.float32),
                "name_rank": nrank.astype(np.int16), "name_top1": ntop,
                "name_margin": np.where(nrank == 0, cos["name"] - nsec, cos["name"] - ntop).astype(np.float32),
                "dense_rank": drank.astype(np.int16),
                "dense_margin": np.where(drank == 0, dcos - dsec, dcos - dtop).astype(np.float32),
                "s1_ncand": stats["ncand"][gs], "s1_n_top1": stats["ntop1"][gs],
                "s1_gap_top": stats["max"][gs] - sc,
                "s1_rank": (np.searchsorted(skey, _skey(gs, sc)) - np.searchsorted(skey, gs * SKEY_BINS)).astype(np.int32),
                "s1_name_dup": nsize[ncode[ls]].astype(np.int32),
                "q_name_dup": np.where(qn_code >= 0, nsize[np.maximum(qn_code, 0)], 0)[lq].astype(np.int32),
                "s1_key_dup": ksize[skc].astype(np.int32), "q_key_dup": size[lq].astype(np.int32),
                "key_eq": (qk[lq] == skc).astype(np.int8), "cand_same_key": kcnt[kinv].astype(np.int16),
            })
            for i, name in enumerate("ABE"):
                m = split_of[C.q_row.values] == i
                if m.any():
                    out.write(f"{name}_{country}", C[m])
            log(f"[{country}] chunk {j:,}: {len(qc):,} q -> {len(C):,} pairs ({len(C) / len(qc):.1f}/q) "
                f"[{time.time() - tj:.0f}s]")
            del C
        del Es, cm, Ms, fz, Msf, MsfT, B0, skey, Wd, Dc
        torch.cuda.empty_cache()
    out.close()
    sb = pd.concat(sib, ignore_index=True).sort_values(["s1_row", "score"], ascending=[True, False], kind="stable")
    sb = sb.groupby("s1_row", sort=False).head(SIB_CAP)
    sb.to_parquet(V3 / "siblings.parquet", index=False)
    log(f"cands done: {len(sb):,} sibling links [{time.time() - t0:.0f}s]")


# ------------------------------------------------------------------ step: recall
def _n_same_true(s1, truth, qrows):
    """Number of S1 records (same country) sharing the true S1's exact name_core."""
    key = s1.country.values.astype(object) + "|" + s1.name_core.values.astype(object)
    code, uniq = pd.factorize(key)
    cnt = np.bincount(code)
    return cnt[code[truth[qrows]]]


def recall():
    """Blocking recall of the eval true links: v2 rank<10 (old stage-1 set), full v2, v3,
    plus what each extra search contributes, split by empty / non-empty address."""
    s1, q = tables(["country", "name_core", "addr_norm"])
    truth = np.load(config.WORK_DIR / "train_truth.npy")
    spl, parts, split_of = study_split(len(q))
    E = read_split("cands", "E", ["q_row", "s1_row", "found", "v2_rank"])
    hit = E[truth[E.q_row.values] == E.s1_row.values]
    in_t = np.zeros(len(s1), bool)
    in_t[spl["tune_s1"]] = True
    tq = parts["E"][(truth[parts["E"]] >= 0) & in_t[np.maximum(truth[parts["E"]], 0)]]
    T = pd.DataFrame({"q": tq, "aempty": q.addr_norm.values[tq] == ""})
    h = hit.set_index("q_row").reindex(tq)
    T["found"] = h.found.fillna(0).astype(int).values
    T["v2_rank"] = h.v2_rank.fillna(-1).astype(int).values
    T["n_same"] = _n_same_true(s1, truth, tq)
    rows = []
    for seg, D in (("all", T), ("empty", T[T.aempty]), ("non-empty", T[~T.aempty])):
        r = {"segment": seg, "n": len(D),
             "v2 rank<10": ((D.v2_rank >= 0) & (D.v2_rank < 10)).mean(),
             "v2 all": (D.v2_rank >= 0).mean(), "v3 all": (D.found > 0).mean()}
        for ch in ("dense", "wide", "fuzzy", "key"):
            b = BITS[ch]
            r[f"only {ch}"] = (((D.found & b) > 0) & ((D.found & ~b) == 0)).mean()
        rows.append(r)
    R = pd.DataFrame(rows)
    print(R.round(4).to_string(index=False))
    b = pd.cut(T.n_same, [0, 1, 2, 4, 9, 49, 1e9], labels=["1", "2", "3-4", "5-9", "10-49", "50+"])
    print("\nempty-address true links by #S1 sharing the true name: retrieved share (v2 all -> v3)")
    G = T[T.aempty].assign(b=b[T.aempty]).groupby("b", observed=True).agg(
        n=("q", "size"), v2=("v2_rank", lambda s: (s >= 0).mean()), v3=("found", lambda s: (s > 0).mean()))
    print(G.round(3).to_string())
    R.to_csv(V3 / "recall.csv", index=False)
    log(f"candidates per eval query: {len(E) / len(parts['E']):.1f}")


# ------------------------------------------------------------------ step: features
CTX_COLS = ["score", "rank", "q_ncand", "top1", "gap_top1", "margin", "s1_ncand", "s1_n_top1", "s1_gap_top",
            "s1_rank", "name_cos", "addr_cos", "found"]
NEW_COLS = ["found_x", "n_found", "fuzzy_cos", "dense_cos", "dense_rank", "dense_margin", "name_rank",
            "name_top1", "name_margin", "s1_name_dup", "q_name_dup", "s1_key_dup", "q_key_dup", "key_eq",
            "cand_same_key"]


class PrecomputedContext:
    """features.Context drop-in: the context columns are already in the v3 candidate table."""

    def features(self, pairs):
        f = pairs[CTX_COLS].astype(np.float32)
        f["found"] = (pairs.found.values & 7).astype(np.float32)  # the three v2 searches, as before
        return f


def features():
    s1, q = tables(["name_full", "name_core", "name_sq", "legal", "is_domain", "addr_norm", "addr_nums"])
    truth = np.load(config.WORK_DIR / "train_truth.npy")
    for f in V3.glob("feat_*.parquet"):
        f.unlink()
    ctx = PrecomputedContext()
    popcount = np.array([bin(i).count("1") for i in range(128)], np.float32)
    with make_pool(build_idf(s1)) as pool:
        for name in ("A", "B", "E"):
            n_out, n_pairs = 0, 0
            for f in sorted(V3.glob(f"cands_{name}_*.parquet")):
                for batch in pq.ParquetFile(f).iter_batches(batch_size=FEAT_CHUNK):
                    c = batch.to_pandas()
                    X = compute_features(c, s1, q, pool, ctx).reset_index(drop=True)
                    X["found_x"] = (c.found.values >> 3).astype(np.float32)
                    X["n_found"] = popcount[c.found.values]
                    for col in NEW_COLS[2:]:
                        X[col] = c[col].values.astype(np.float32)
                    X.insert(0, "q_row", c.q_row.values)
                    X.insert(1, "s1_row", c.s1_row.values)
                    X["y"] = (truth[c.q_row.values] == c.s1_row.values).astype(np.int8)
                    X.to_parquet(V3 / f"feat_{name}_{n_out:03d}.parquet", index=False)
                    n_out += 1
                    n_pairs += len(X)
            log(f"features {name}: {n_pairs:,} pairs in {n_out} files")


def xcols(df):
    return [c for c in df.columns if c not in ("q_row", "s1_row", "y")]


# ------------------------------------------------------------------ step: stage 1
class ParquetIter(xgb.DataIter):
    """Streams feature files into a QuantileDMatrix (the float data never sits in RAM at once)."""

    def __init__(self, files, cols, drop_q):
        self.files, self.cols, self.drop_q, self.i = files, cols, drop_q, 0
        super().__init__(release_data=True)

    def next(self, input_data):
        if self.i == len(self.files):
            return False
        df = pd.read_parquet(self.files[self.i])
        df = df[~self.drop_q[df.q_row.values]]
        input_data(data=df[self.cols], label=df.y.values)
        self.i += 1
        return True

    def reset(self):
        self.i = 0


def predict_files(bst, name, cols, n_iter):
    out = []
    for f in sorted(V3.glob(f"feat_{name}_*.parquet")):
        D = pd.read_parquet(f)
        p = bst.inplace_predict(D[cols], iteration_range=(0, n_iter))
        out.append(pd.DataFrame({"q_row": D.q_row.values, "s1_row": D.s1_row.values, "y": D.y.values,
                                 "p1": np.asarray(p, np.float32)}))
    return pd.concat(out, ignore_index=True)


def eval_setup():
    spl = np.load(SD / "split.npz")
    truth = np.load(config.WORK_DIR / "train_truth.npy")
    PE, Q = by_query(pd.read_parquet(V3 / "p1_E.parquet"))
    n_s1 = int(max(truth.max(), PE.s1_row.max(), spl["eval_s1"].max())) + 1
    return spl, truth, Scorer(truth, n_s1), PE, Q


def stage1():
    files = sorted(V3.glob("feat_A_*.parquet"))
    cols = xcols(pd.read_parquet(files[0]).head(1))
    truth = np.load(config.WORK_DIR / "train_truth.npy")
    rng = np.random.default_rng(config.SEED + 2)
    qa = np.unique(pd.concat([pd.read_parquet(f, columns=["q_row"]) for f in files]).q_row.values)
    is_dev = np.zeros(len(truth), bool)
    is_dev[rng.choice(qa, len(qa) // 20, replace=False)] = True
    dev = pd.concat([d[is_dev[d.q_row.values]] for d in (pd.read_parquet(f) for f in files)], ignore_index=True)
    drop = is_dev.copy()
    frac = float(os.environ.get("STUDY_V3_S1_FRAC", "1"))   # optional query subsample if GPU memory is short
    if frac < 1:
        rest = qa[~is_dev[qa]]
        drop[rng.choice(rest, int(len(rest) * (1 - frac)), replace=False)] = True
    dtr = xgb.QuantileDMatrix(ParquetIter(files, cols, drop), max_bin=256)
    ddv = xgb.QuantileDMatrix(dev[cols], label=dev.y.values, ref=dtr)
    log(f"stage1: {len(cols)} features, train rows {dtr.num_row():,} (query fraction {frac}), "
        f"dev rows {len(dev):,}, dev pos rate {dev.y.mean():.3f}")
    del dev
    params = dict(XGB_PARAMS, device=device(), seed=config.SEED)
    bst = xgb.train(params, dtr, 3000, evals=[(ddv, "dev")], early_stopping_rounds=50, verbose_eval=250)
    del dtr, ddv
    bst.save_model(V3 / "stage1.json")
    n_iter = bst.best_iteration + 1
    imp = sorted(bst.get_score(importance_type="gain").items(), key=lambda x: -x[1])[:15]
    json.dump({"best_iteration": int(bst.best_iteration), "features": cols, "top_gain": imp},
              open(V3 / "stage1_meta.json", "w"), indent=2)
    log(f"stage1 best_iter={bst.best_iteration}; top gain: {[(k, round(v)) for k, v in imp[:10]]}")
    for name in ("B", "E"):
        predict_files(bst, name, cols, n_iter).to_parquet(V3 / f"p1_{name}.parquet", index=False)
    spl, truth, scorer, PE, Q = eval_setup()
    tau1, tune, rep, f_rep = baseline_rule(scorer, spl, Q)
    json.dump({"tau1": float(tau1), "tune": tune, "report": rep}, open(V3 / "baseline.json", "w"), indent=2)
    log(f"stage-1 v3 baseline tau1={tau1:.2f}  TUNE {fmt(tune)}  REPORT {fmt(rep)}")
    oracle(scorer, spl, PE, Q, tau1)


def ablation():
    """Stage 1 on the v3 candidates but with only the 45 study.py features: separates the
    effect of the extra candidates from the effect of the new features."""
    files = sorted(V3.glob("feat_A_*.parquet"))
    cols = [c for c in json.load(open(V3 / "stage1_meta.json"))["features"] if c not in NEW_COLS]
    truth = np.load(config.WORK_DIR / "train_truth.npy")
    rng = np.random.default_rng(config.SEED + 2)
    qa = np.unique(pd.concat([pd.read_parquet(f, columns=["q_row"]) for f in files]).q_row.values)
    is_dev = np.zeros(len(truth), bool)
    is_dev[rng.choice(qa, len(qa) // 20, replace=False)] = True
    dev = pd.concat([d[is_dev[d.q_row.values]] for d in (pd.read_parquet(f) for f in files)], ignore_index=True)
    dtr = xgb.QuantileDMatrix(ParquetIter(files, cols, is_dev), max_bin=256)
    ddv = xgb.QuantileDMatrix(dev[cols], label=dev.y.values, ref=dtr)
    del dev
    bst = xgb.train(dict(XGB_PARAMS, device=device(), seed=config.SEED), dtr, 3000, evals=[(ddv, "dev")],
                    early_stopping_rounds=50, verbose_eval=False)
    del dtr, ddv
    PE, Q = by_query(predict_files(bst, "E", cols, bst.best_iteration + 1))
    spl, truth, scorer, _, _ = eval_setup()
    tau1, tune, rep, _ = baseline_rule(scorer, spl, Q)
    json.dump({"features": len(cols), "best_iteration": int(bst.best_iteration), "tau1": float(tau1),
               "tune": tune, "report": rep}, open(V3 / "ablation.json", "w"), indent=2)
    log(f"ablation ({len(cols)} old features on v3 candidates): tau1={tau1:.2f} TUNE {fmt(tune)} REPORT {fmt(rep)}")


def baseline_rule(scorer, spl, Q):
    def run(ents, t):
        m = Q.p_max.values >= t
        return scorer.per_entity(ents, Q.q_row.values[m], Q.s_top.values[m])[0]
    tau1 = max(GRID_T1, key=lambda t: run(spl["tune_s1"], t).mean())
    m = Q.p_max.values >= tau1
    qa, sa = Q.q_row.values[m], Q.s_top.values[m]
    return tau1, scorer.score(spl["tune_s1"], qa, sa), scorer.score(spl["report_s1"], qa, sa), \
        run(spl["report_s1"], tau1)


def fmt(d):
    return f"F0.5={d['f05']:.4f} P={d['precision']:.4f} R={d['recall']:.4f}"


def oracle(scorer, spl, PE, Q, tau1):
    """Stage-2 ceiling: F0.5 on TUNE if stage 2 were perfect on the whole wide band, for top 5 / 8."""
    for k in (5, TOP2):
        hit = PE[(PE.p1_rank < k) & (PE.y == 1)]
        s_orc = np.full(len(Q), -1, np.int64)
        s_orc[np.searchsorted(Q.q_row.values, hit.q_row.values)] = hit.s1_row.values
        qa, sa, share = decide(Q, WIDE[0], WIDE[1], tau1, s_orc, np.ones(len(Q)), 0.5)
        log(f"oracle top-{k} in band {WIDE}: TUNE {fmt(scorer.score(spl['tune_s1'], qa, sa))} band={share:.3f}")


# ------------------------------------------------------------------ step: stage 2
def band_pairs(name):
    P, Q = by_query(pd.read_parquet(V3 / f"p1_{name}.parquet"))
    inq = Q.q_row.values[(Q.p_max.values >= WIDE[0]) & (Q.p_max.values < WIDE[1])]
    return P[np.isin(P.q_row.values, inq) & (P.p1_rank < TOP2)].reset_index(drop=True)


def gather_features(name, P, n_s1):
    """Stage-1 features of the pairs in P (streamed semi-join), aligned with P's rows."""
    want = P.q_row.values.astype(np.int64) * n_s1 + P.s1_row.values
    order = np.argsort(want)
    ws = want[order]
    parts = []
    for f in sorted(V3.glob(f"feat_{name}_*.parquet")):
        D = pd.read_parquet(f)
        k = D.q_row.values.astype(np.int64) * n_s1 + D.s1_row.values
        i = np.minimum(np.searchsorted(ws, k), len(ws) - 1)
        m = ws[i] == k
        parts.append(D[m].assign(_pos=order[i[m]]))
    F = pd.concat(parts, ignore_index=True).sort_values("_pos")
    assert len(F) == len(P) and (F._pos.values == np.arange(len(P))).all()
    return F.drop(columns=["_pos", "q_row", "s1_row", "y"]).reset_index(drop=True)


def _ce_text(df, idx):
    return [f"{n} | {a}" for n, a in zip(df.name_full.values[idx], df.addr_norm.values[idx])]


def _ce_fit_predict(tr_a, tr_b, tr_y, pred_sets, bs=64, lr=3e-5, max_len=96):
    """study._ce_fit_predict with length-sorted prediction batches (same training recipe)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
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
        if (i // bs) % 2000 == 0:
            log(f"  ce step {i // bs}/{steps} loss {loss.item():.4f}")
    model.eval()
    outs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for a, b in pred_sets:
            o = np.empty(len(a), np.float32)
            ln = np.fromiter((len(x) + len(y) for x, y in zip(a, b)), np.int32, len(a))
            srt = np.argsort(ln, kind="stable")
            for i in range(0, len(a), 1024):
                idx = srt[i:i + 1024]
                o[idx] = model(**enc([a[k] for k in idx], [b[k] for k in idx])).logits.squeeze(-1).float().cpu().numpy()
            outs.append(o)
    del model
    torch.cuda.empty_cache()
    return outs


def ce_scores(PB, PE, s1, q, truth, n_train=CE_MAX_TRAIN, tag=""):
    """Cross-encoder logit, 2-fold cross-fitted on B (folds split by true S1 entity);
    n_train training pairs per fold, sampled with empty-address queries up-weighted x EMPTY_W."""
    cache = [V3 / f"ce_B{tag}.npy", V3 / f"ce_E{tag}.npy"]
    if all(c.exists() for c in cache):
        return np.load(cache[0]), np.load(cache[1])
    fold_s1 = np.random.default_rng(config.SEED + 3).integers(0, 2, len(s1))
    tq = truth[PB.q_row.values]
    fold = np.where(tq >= 0, fold_s1[np.maximum(tq, 0)], PB.q_row.values % 2)
    w = np.where(q.addr_norm.values[PB.q_row.values] == "", EMPTY_W, 1.0)
    B_a, B_b = _ce_text(s1, PB.s1_row.values), _ce_text(q, PB.q_row.values)
    E_a, E_b = _ce_text(s1, PE.s1_row.values), _ce_text(q, PE.q_row.values)
    ce_B = np.empty(len(PB), np.float32)
    ce_E = np.zeros(len(PE), np.float32)
    rng = np.random.default_rng(config.SEED + 4)
    for k in (0, 1):
        tr = np.flatnonzero(fold == k)
        if len(tr) > n_train:   # weighted sampling without replacement (Gumbel top-k)
            tr = np.sort(tr[np.argsort(-(np.log(w[tr]) + rng.gumbel(size=len(tr))))[:n_train]])
        te = np.flatnonzero(fold != k)
        log(f"ce fold {k}: train {len(tr):,} pairs (pos {PB.y.values[tr].mean():.3f}), "
            f"predict {len(te):,} + eval {len(PE):,}")
        pb, pe = _ce_fit_predict([B_a[i] for i in tr], [B_b[i] for i in tr], PB.y.values[tr].astype(np.float32),
                                 [([B_a[i] for i in te], [B_b[i] for i in te]), (E_a, E_b)])
        ce_B[te] = pb
        ce_E += pe / 2
    np.save(cache[0], ce_B)
    np.save(cache[1], ce_E)
    return ce_B, ce_E


def rival_features(P, ce):
    """CE score relative to the query's other re-scored candidates."""
    q = P.q_row.values
    rank, top, sec, _ = within_rank(q, ce)
    ex = np.exp(ce - top)
    tot = pd.Series(ex).groupby(q).transform("sum").values
    return pd.DataFrame({"ce": ce, "ce_rank": rank.astype(np.float32), "ce_gap": top - ce,
                         "ce_margin": np.where(rank == 0, ce - sec, ce - top), "ce_soft": ex / tot})


def sibling_features(P, q):
    """Collective evidence for pair (x, E): the other queries whose top v2 blocking
    candidate is E with score >= SIB_MIN (computed over ALL queries, no labels)."""
    sib = pd.read_parquet(V3 / "siblings.parquet").sort_values("s1_row", kind="stable")
    ss, sq_all = sib.s1_row.values, sib.q_row.values
    a = np.searchsorted(ss, P.s1_row.values, "left")
    cnt = np.searchsorted(ss, P.s1_row.values, "right") - a
    rep = np.repeat(np.arange(len(P)), cnt)
    sq = sq_all[np.repeat(a, cnt) + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)]
    x = P.q_row.values[rep]
    keep = sq != x
    rep, sq, x = rep[keep], sq[keep], x[keep]
    same = q.src.values[sq] == q.src.values[x]
    nm, ad = q.name_core.values, q.addr_norm.values
    nsim = process.cpdist(nm[x].tolist(), nm[sq].tolist(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    asim = process.cpdist(ad[x].tolist(), ad[sq].tolist(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    asim[(ad[x] == "") | (ad[sq] == "")] = np.nan

    def gmax(v, m):
        out = np.full(len(P), -1.0, np.float32)
        ok = m & ~np.isnan(v)
        np.maximum.at(out, rep[ok], v[ok])
        return np.where(out < 0, np.nan, out)
    return pd.DataFrame({"sib_n": np.bincount(rep, minlength=len(P)).astype(np.float32),
                         "sib_n_src": np.bincount(rep[same], minlength=len(P)).astype(np.float32),
                         "sib_name_max": gmax(nsim, np.ones(len(rep), bool)),
                         "sib_addr_max": gmax(asim, np.ones(len(rep), bool)),
                         "sib_addr_src_max": gmax(asim, same)})


def stage2():
    s1, q = tables(["name_full", "name_core", "addr_norm"])
    truth = np.load(config.WORK_DIR / "train_truth.npy")
    PB, PE = band_pairs("B"), band_pairs("E")
    log(f"stage2 band pairs (top {TOP2}): B {len(PB):,} ({PB.q_row.nunique():,} q), "
        f"E {len(PE):,} ({PE.q_row.nunique():,} q)")
    s1cols = json.load(open(V3 / "stage1_meta.json"))["features"]
    FB = pd.concat([PB, gather_features("B", PB, len(s1))], axis=1)
    FE = pd.concat([PE, gather_features("E", PE, len(s1))], axis=1)
    ce_B, ce_E = ce_scores(PB, PE, s1, q, truth)
    FB = pd.concat([FB, rival_features(PB, ce_B), sibling_features(PB, q)], axis=1)
    FE = pd.concat([FE, rival_features(PE, ce_E), sibling_features(PE, q)], axis=1)
    cols = s1cols + ["p1", "p1_rank", "p1_gap", "ce", "ce_rank", "ce_gap", "ce_margin", "ce_soft",
                     "sib_n", "sib_n_src", "sib_name_max", "sib_addr_max", "sib_addr_src_max"]
    rng = np.random.default_rng(config.SEED + 5)
    uq = np.unique(FB.q_row.values)
    is_dev = np.isin(FB.q_row.values, rng.choice(uq, len(uq) // 10, replace=False))
    res = {}
    for tag, use in (("full", cols), ("no_sib", [c for c in cols if not c.startswith("sib_")])):
        dtr = xgb.DMatrix(FB.loc[~is_dev, use], label=FB.y.values[~is_dev])
        ddv = xgb.DMatrix(FB.loc[is_dev, use], label=FB.y.values[is_dev])
        bst = xgb.train(dict(S2_PARAMS, device=device(), seed=config.SEED), dtr, 3000, evals=[(ddv, "dev")],
                        early_stopping_rounds=50, verbose_eval=False)
        bst.save_model(V3 / f"stage2_{tag}.json")
        FE[f"p2_{tag}"] = bst.predict(xgb.DMatrix(FE[use]), iteration_range=(0, bst.best_iteration + 1))
        imp = sorted(bst.get_score(importance_type="gain").items(), key=lambda x: -x[1])[:10]
        res[tag] = {"best_iter": int(bst.best_iteration), "dev_logloss": float(bst.best_score),
                    "top_gain": [(k, round(v, 1)) for k, v in imp]}
        log(f"stage2[{tag}] best_iter={bst.best_iteration} dev logloss={bst.best_score:.5f} top: {res[tag]['top_gain'][:6]}")
    FE[["q_row", "s1_row", "y", "p1", "p1_rank", "ce", "p2_full", "p2_no_sib"]].to_parquet(V3 / "p2_E.parquet", index=False)
    json.dump({"features": cols, **res}, open(V3 / "stage2_meta.json", "w"), indent=2)


# ------------------------------------------------------------------ step: decide
def per_query_best(Q, P2, col):
    best = P2.sort_values(["q_row", col], ascending=[True, False]).drop_duplicates("q_row")
    s2 = np.full(len(Q), -1, np.int64)
    p2 = np.zeros(len(Q))
    pos = np.searchsorted(Q.q_row.values, best.q_row.values)
    s2[pos], p2[pos] = best.s1_row.values, best[col].values
    return s2, p2


def cascade_seg(Q, seg, prm, s2, p2):
    """Cascade assignment with per-segment (tau1, L, U, tau2)."""
    qa, sa = [], []
    for g, (t1, L, U, t2) in prm.items():
        m = seg == g
        a, b, _ = decide(Q[m].reset_index(drop=True), L, U, t1, s2[m], p2[m], t2)
        qa.append(a)
        sa.append(b)
    return np.concatenate(qa), np.concatenate(sa)


def sweep_segment(scorer, ents, Q, seg, prm, g, s2, p2, tau1):
    best = None
    for L in GRID_L:
        for U in GRID_U:
            for t2 in GRID_T2:
                trial = dict(prm)
                trial[g] = (float(tau1), float(L), float(U), float(t2))
                f = scorer.score(ents, *cascade_seg(Q, seg, trial, s2, p2))["f05"]
                if best is None or f > best[0]:
                    best = (f, trial[g])
    return best


def expected_f(P, Q, gamma, gamma_e, m0, empty_q, passes=3):
    """Expected-F0.5 assignment. P: all candidate pairs of the eval queries with a final
    probability p (stage 2 for re-scored band pairs, stage 1 otherwise).

    For S1 entity s with assigned set A (n = |A|, t = sum of p over A) and mu_s = expected
    number of true queries (sum of p over all its candidate pairs + m0):
        E[F(A)] ~= 1.25 t / (0.25 mu_s + n)     (n > 0)
        E[F(0)]  = P(no true query) = prod(1 - p) * exp(-m0)
    Each query goes to the candidate with the largest positive gain; coordinate ascent."""
    p = P.p.values.astype(np.float64)
    e = empty_q[P.q_row.values]
    p = np.where(e, p ** gamma_e, p ** gamma)
    tot = pd.Series(p).groupby(P.q_row.values).transform("sum").values
    p = p / np.maximum(tot, 1.0)
    keep = p >= 0.01
    q, s, p = P.q_row.values[keep], P.s1_row.values[keep], p[keep]
    su, sinv = np.unique(s, return_inverse=True)
    mu = (np.bincount(sinv, weights=p, minlength=len(su)) + m0).tolist()
    pi0 = np.exp(np.bincount(sinv, weights=np.log1p(-np.minimum(p, 0.999999)), minlength=len(su)) - m0).tolist()
    order = np.lexsort((-p, q))
    q, sinv, p = q[order], sinv[order], p[order]
    first, sizes = group_first(q)
    visit = np.argsort(-p[first], kind="stable").tolist()   # confident queries first
    first_l, sizes_l, sv, pv = first.tolist(), sizes.tolist(), sinv.tolist(), p.tolist()
    n = [0] * len(su)
    t = [0.0] * len(su)
    cur = [-1] * len(first)   # pair index of each query's current assignment

    for _ in range(passes):
        changed = 0
        for gi in visit:
            c = cur[gi]
            if c >= 0:
                k = sv[c]
                n[k] -= 1
                t[k] -= pv[c]
            best, bj = 0.0, -1
            j0 = first_l[gi]
            for j in range(j0, j0 + sizes_l[gi]):
                k = sv[j]
                nk, tk, d = n[k], t[k], 0.25 * mu[k]
                g_ = 1.25 * (tk + pv[j]) / (d + nk + 1) - (pi0[k] if nk == 0 else 1.25 * tk / (d + nk))
                if g_ > best:
                    best, bj = g_, j
            if bj >= 0:
                k = sv[bj]
                n[k] += 1
                t[k] += pv[bj]
            if bj != c:
                changed += 1
            cur[gi] = bj
        if changed == 0:
            break
    cur = np.array(cur)
    a = cur >= 0
    return q[first[a]], su[sinv[cur[a]]]


def old_systems():
    """Previous study (study.py): stage-1 rule and the best CE cascade, as eval assignments."""
    import study
    spl0, scorer0, PE0, Q0 = study._eval_setup()
    b = json.load(open(SD / "baseline.json"))
    m0 = Q0.p_max.values >= b["tau1"]
    out = {"old stage-1": (Q0.q_row.values[m0], Q0.s_top.values[m0])}
    r = json.load(open(SD / "results_ce.json"))
    pk = r["picks"]["best_f05"]
    PE = study.band_pairs("E")
    FE = PE.merge(study.load_feat("E").drop(columns="y"), on=["q_row", "s1_row"], how="left")
    FE = pd.concat([FE, pd.read_parquet(SD / "m_ce_E.parquet")], axis=1)
    bst = xgb.Booster()
    bst.load_model(SD / "stage2_ce.json")
    FE["p2"] = bst.predict(xgb.DMatrix(FE[r["features"]]), iteration_range=(0, r["best_iter"] + 1))
    s2, p2 = per_query_best(Q0, FE, "p2")
    qa, sa, _ = decide(Q0, pk["L"], pk["U"], b["tau1"], s2, p2, pk["tau2"])
    out["old CE cascade"] = (qa, sa)
    return out


def decide_step():
    s1, q = tables(["addr_norm"])
    spl, truth, scorer, PE, Q = eval_setup()
    empty_q = q.addr_norm.values == ""
    seg = empty_q[Q.q_row.values].astype(int)          # 1 = empty address
    b = json.load(open(V3 / "baseline.json"))
    tau1 = b["tau1"]
    P2 = pd.read_parquet(V3 / "p2_E.parquet")
    tune, rep = spl["tune_s1"], spl["report_s1"]
    systems = {k: v for k, v in old_systems().items()}
    m = Q.p_max.values >= tau1
    systems["v3 stage-1"] = (Q.q_row.values[m], Q.s_top.values[m])
    picks = {}
    for col in ("p2_no_sib", "p2_full"):
        s2, p2 = per_query_best(Q, P2, col)
        # (a) one global (L, U, tau2)
        R = []
        for L in GRID_L:
            for U in GRID_U:
                for t2 in GRID_T2:
                    qa, sa, share = decide(Q, L, U, tau1, s2, p2, t2)
                    R.append(dict(L=L, U=U, tau2=float(t2), band=share, **scorer.score(tune, qa, sa)))
        R = pd.DataFrame(R)
        pk = R.loc[R.f05.idxmax()]
        prm = {g: (float(tau1), float(pk.L), float(pk.U), float(pk.tau2)) for g in (0, 1)}
        systems[f"v3 cascade [{col}] global"] = cascade_seg(Q, seg, prm, s2, p2)
        picks[f"{col} global"] = prm[0]
        # (b) separate (L, U, tau2) for empty / non-empty addresses (coordinate ascent, 2 rounds)
        for _ in range(2):
            for g in (1, 0):
                prm[g] = sweep_segment(scorer, tune, Q, seg, prm, g, s2, p2, tau1)[1]
        systems[f"v3 cascade [{col}] segment"] = cascade_seg(Q, seg, prm, s2, p2)
        picks[f"{col} segment"] = {"empty": prm[1], "non-empty": prm[0]}
        log(f"{col}: global pick {picks[f'{col} global']}, segment picks {picks[f'{col} segment']}")

    # (c) expected-F0.5 on the final probabilities of every candidate pair of the eval queries
    best_col = "p2_full"
    PP = PE[["q_row", "s1_row", "p1"]].merge(P2[["q_row", "s1_row", best_col]], on=["q_row", "s1_row"], how="left")
    PP["p"] = PP[best_col].fillna(PP.p1).values
    band_q = np.isin(PP.q_row.values, P2.q_row.values)
    PP.loc[band_q & PP[best_col].isna(), "p"] = 0.0   # band queries: only the re-scored top 8 count
    PP = PP[PP.p.values >= 0.003].reset_index(drop=True)   # below 0.01 after any gamma in the grid
    best = None
    for gamma in (0.8, 1.0, 1.25, 1.5, 2.0):
        for gamma_e in (0.8, 1.0, 1.25, 1.5, 2.0):
            for m0 in (0.0, 0.1):
                qa, sa = expected_f(PP, Q, gamma, gamma_e, m0, empty_q, passes=2)
                f = scorer.score(tune, qa, sa)["f05"]
                if best is None or f > best[0]:
                    best = (f, gamma, gamma_e, m0)
    log(f"expected-F pick: gamma={best[1]} gamma_empty={best[2]} m0={best[3]} TUNE F0.5={best[0]:.4f}")
    systems["v3 expected-F"] = expected_f(PP, Q, best[1], best[2], best[3], empty_q, passes=3)
    picks["expected-F"] = {"gamma": best[1], "gamma_empty": best[2], "m0": best[3]}

    # score everything: TUNE and REPORT, recall by address segment, paired bootstrap on REPORT
    ref = {k: scorer.per_entity(rep, *systems[k])[0] for k in ("old stage-1", "old CE cascade")}
    rows = []
    for name, (qa, sa) in systems.items():
        tu, re_ = scorer.score(tune, qa, sa), scorer.score(rep, qa, sa)
        f_rep = scorer.per_entity(rep, qa, sa)[0]
        r = dict(system=name, tune_f05=tu["f05"], F05=re_["f05"], P=re_["precision"], R=re_["recall"])
        r.update(link_recall(truth, rep, qa, sa, empty_q))
        for k, fb in ref.items():
            d, lo, hi = bootstrap_delta(f_rep, fb)
            r[f"dF vs {k}"] = f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]"
        rows.append(r)
        np.savez(V3 / f"assign_{name.replace(' ', '_').replace('[', '').replace(']', '')}.npz", qa=qa, sa=sa)
    T = pd.DataFrame(rows)
    T.to_csv(V3 / "decide.csv", index=False)
    json.dump(picks, open(V3 / "picks.json", "w"), indent=2, default=float)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string(index=False))


def link_recall(truth, ents, qa, sa, empty_q):
    """Share of true links of `ents` recovered, overall and by query address segment."""
    in_e = np.zeros(truth.max() + 1, bool)
    in_e[ents] = True
    tq = np.flatnonzero((truth >= 0) & in_e[np.maximum(truth, 0)])
    got = np.zeros(len(truth), bool)
    got[qa[truth[qa] == sa]] = True
    e = empty_q[tq]
    return {"link_R": got[tq].mean(), "link_R_empty": got[tq[e]].mean(), "link_R_nonempty": got[tq[~e]].mean()}


# ------------------------------------------------------------------ step: ce_control (training-size control)
def _sweep_global(scorer, ents, Q, tau1, s2, p2):
    best = None
    for L in GRID_L:
        for U in GRID_U:
            for t2 in GRID_T2:
                qa, sa, _ = decide(Q, L, U, tau1, s2, p2, t2)
                f = scorer.score(ents, qa, sa)["f05"]
                if best is None or f > best[0]:
                    best = (f, float(L), float(U), float(t2))
    return best


def ce_control(mode, n_train):
    """Is the v3 gain just the larger cross-encoder training set (600k pairs per fold vs 300k)?
      mode v3 : the reported v3 cascade, only the CE trained on n_train pairs per fold
      mode old: the study.py cascade, only its CE trained on up to n_train pairs per fold
    Scored exactly like the reported systems: global (L, U, tau2) picked on TUNE, REPORT half,
    paired bootstrap vs the old CE cascade and vs the reported v3 cascade."""
    import study
    s1, q = tables(["name_full", "name_core", "addr_norm"])
    truth = np.load(config.WORK_DIR / "train_truth.npy")
    empty_q = q.addr_norm.values == ""
    if mode == "v3":
        PB, PE = band_pairs("B"), band_pairs("E")
        cols = json.load(open(V3 / "stage2_meta.json"))["features"]
        FB = pd.concat([PB, gather_features("B", PB, len(s1))], axis=1)
        FE = pd.concat([PE, gather_features("E", PE, len(s1))], axis=1)
        ce_B, ce_E = ce_scores(PB, PE, s1, q, truth, n_train=n_train, tag=f"_{n_train}")
        FB = pd.concat([FB, rival_features(PB, ce_B), sibling_features(PB, q)], axis=1)
        FE = pd.concat([FE, rival_features(PE, ce_E), sibling_features(PE, q)], axis=1)
        spl, _, scorer, _, Q = eval_setup()
        tau1 = json.load(open(V3 / "baseline.json"))["tau1"]
    else:
        study.CE_MAX_TRAIN = n_train   # study.ce_features reads it at call time
        PB, PE = study.band_pairs("B"), study.band_pairs("E")
        cols = json.load(open(SD / "stage1_meta.json"))["features"] + ["p1", "p1_rank", "p1_gap", "ce"]
        FB = PB.merge(study.load_feat("B").drop(columns="y"), on=["q_row", "s1_row"], how="left")
        FE = PE.merge(study.load_feat("E").drop(columns="y"), on=["q_row", "s1_row"], how="left")
        cache = [V3 / f"ce_old_B_{n_train}.parquet", V3 / f"ce_old_E_{n_train}.parquet"]
        if all(c.exists() for c in cache):
            MB, ME = (pd.read_parquet(c) for c in cache)
        else:
            MB, ME = study.ce_features(PB, PE, s1, q, truth)
            MB.to_parquet(cache[0], index=False)
            ME.to_parquet(cache[1], index=False)
        FB = pd.concat([FB, MB], axis=1)
        FE = pd.concat([FE, ME], axis=1)
        spl, scorer, _, Q = study._eval_setup()
        tau1 = json.load(open(SD / "baseline.json"))["tau1"]
    rng = np.random.default_rng(config.SEED + 5)
    uq = np.unique(FB.q_row.values)
    is_dev = np.isin(FB.q_row.values, rng.choice(uq, len(uq) // 10, replace=False))
    dtr = xgb.DMatrix(FB.loc[~is_dev, cols], label=FB.y.values[~is_dev])
    ddv = xgb.DMatrix(FB.loc[is_dev, cols], label=FB.y.values[is_dev])
    bst = xgb.train(dict(S2_PARAMS, device=device(), seed=config.SEED), dtr, 3000, evals=[(ddv, "dev")],
                    early_stopping_rounds=50, verbose_eval=False)
    FE["p2"] = bst.predict(xgb.DMatrix(FE[cols]), iteration_range=(0, bst.best_iteration + 1))
    s2, p2 = per_query_best(Q, FE, "p2")
    f_t, L, U, t2 = _sweep_global(scorer, spl["tune_s1"], Q, tau1, s2, p2)
    qa, sa, share = decide(Q, L, U, tau1, s2, p2, t2)
    rep = scorer.score(spl["report_s1"], qa, sa)
    f_rep = scorer.per_entity(spl["report_s1"], qa, sa)[0]
    A = np.load(V3 / "assign_v3_cascade_p2_full_global.npz")
    refs = {"old CE cascade (300k/fold)": old_systems()["old CE cascade"], "v3 cascade (600k/fold)": (A["qa"], A["sa"])}
    out = dict(mode=mode, ce_pairs_per_fold=n_train, best_iter=int(bst.best_iteration),
               dev_logloss=float(bst.best_score), L=L, U=U, tau2=t2, band=share, tune_f05=f_t, report=rep,
               **link_recall(truth, spl["report_s1"], qa, sa, empty_q))
    for k, (ra, rs) in refs.items():
        out[f"dF vs {k}"] = bootstrap_delta(f_rep, scorer.per_entity(spl["report_s1"], ra, rs)[0])
    json.dump(out, open(V3 / f"ce_control_{mode}_{n_train}.json", "w"), indent=2, default=float)
    log(f"ce_control[{mode}, {n_train:,}/fold]: L={L} U={U} tau2={t2} TUNE F0.5={f_t:.4f} REPORT {fmt(rep)} "
        f"link recall empty={out['link_R_empty']:.3f} non-empty={out['link_R_nonempty']:.4f}")
    for k in refs:
        d, lo, hi = out[f"dF vs {k}"]
        log(f"   dF0.5 vs {k}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")


# ------------------------------------------------------------------ step: fn (false-negative analysis)
def fn_step(system=None):
    s1, q = tables(["country", "name_core", "addr_norm"])
    spl, truth, scorer, PE, Q = eval_setup()
    empty_q = q.addr_norm.values == ""
    T = pd.read_csv(V3 / "decide.csv")
    v3 = T[T.system.str.startswith("v3")]
    system = system or v3.loc[v3.tune_f05.idxmax(), "system"]
    A = np.load(V3 / f"assign_{system.replace(' ', '_').replace('[', '').replace(']', '')}.npz")
    qa, sa = A["qa"], A["sa"]
    in_t = np.zeros(len(s1), bool)
    in_t[spl["tune_s1"]] = True
    tq = np.flatnonzero((truth >= 0) & in_t[np.maximum(truth, 0)])
    assigned = np.full(len(truth), -1, np.int64)
    assigned[qa] = sa
    miss = tq[assigned[tq] != truth[tq]]
    E = read_split("cands", "E", ["q_row", "s1_row"])
    retrieved = pd.Series(True, index=E.q_row.values.astype(np.int64) * len(s1) + E.s1_row.values)
    key = miss.astype(np.int64) * len(s1) + truth[miss]
    ret = retrieved.reindex(key).fillna(False).values.astype(bool)
    P2 = pd.read_parquet(V3 / "p2_E.parquet")
    in_top8 = np.isin(key, P2.q_row.values.astype(np.int64) * len(s1) + P2.s1_row.values)
    band = np.isin(miss, P2.q_row.values)
    p1t = PE.assign(k=PE.q_row.values.astype(np.int64) * len(s1) + PE.s1_row.values).set_index("k").p1.reindex(key).values
    cause = np.select(
        [~ret, assigned[miss] >= 0, band & ~in_top8, band, p1t >= 0.5],
        ["never retrieved", "assigned to another entity", "band query, true S1 outside top 8",
         "band query, not accepted", "stage 1 >= 0.5 but not accepted"], "stage-1 probability < 0.5")
    print(f"system: {system}\nTUNE true links {len(tq):,}; missed {len(miss):,} ({len(miss) / len(tq):.2%})")
    C = pd.Series(cause).value_counts()
    print(pd.DataFrame({"count": C, "share": (C / len(miss)).round(3)}).to_string())
    e = empty_q[miss]
    print(f"\nprofile: empty address {e.mean():.3f} of misses vs {empty_q[tq].mean():.3f} of true links; "
          f"India {(q.country.values[miss] == 'India').mean():.3f} vs {(q.country.values[tq] == 'India').mean():.3f}")
    ns = _n_same_true(s1, truth, tq)
    b = pd.cut(ns, [0, 1, 2, 4, 9, 49, 1e9], labels=["1", "2", "3-4", "5-9", "10-49", "50+"])
    ok = assigned[tq] == truth[tq]
    G = pd.DataFrame({"b": b, "empty": empty_q[tq], "ok": ok}).groupby(["empty", "b"], observed=True).ok.agg(["size", "mean"])
    print("\nrecall of TUNE true links by address segment and #S1 sharing the true name:")
    print(G.rename(columns={"size": "n", "mean": "recall"}).round(3).to_string())


def report():
    print(pd.read_csv(V3 / "recall.csv").round(4).to_string(index=False))
    print(open(V3 / "baseline.json").read())
    print(pd.read_csv(V3 / "decide.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["dense", "cands", "recall", "features", "stage1", "ablation", "stage2",
                                     "decide", "ce_control", "fn", "report"])
    ap.add_argument("--system", default=None)
    ap.add_argument("--mode", default="v3", choices=["v3", "old"], help="ce_control: which cascade")
    ap.add_argument("--n", type=int, default=None, help="ce_control: CE training pairs per fold")
    a = ap.parse_args()
    V3.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if a.step == "decide":
        decide_step()
    elif a.step == "ce_control":
        ce_control(a.mode, a.n or (300_000 if a.mode == "v3" else 10 ** 9))
    elif a.step == "fn":
        fn_step(a.system)
    else:
        globals()[a.step]()
