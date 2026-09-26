"""Stage 2: candidate generation (blocking).

Direction: every S2/S3 record ("query") searches the Source-1 index of its own
country (country is treated as an open set of labels). Each S2/S3 record can
belong to at most one S1 entity, so querying from this side and keeping the
top-K S1 records per query is natural.

Representation: sparse TF-IDF over four token channels
  n: name word tokens (v2: + word bigrams)   c: name char 3-grams (typos, domains)
  a: address words (v2: + word bigrams)      d: address number keys
Vocabulary / IDF are fit on S1 of the country; tokens present in too many S1
records are dropped (costly, little signal). Word bigrams keep common-word
names/addresses searchable ('vision enterprises', 'nehru ngr') after pruning.

Modes
  v1: one combined cosine search, top-k.
  v2: three searches -- combined (top KC), name-only (top KN), address-only
      (top KA) -- unioned. Name-only rescues records with empty/garbled
      addresses; address-only rescues garbage/renamed names. Every union pair
      gets all three cosines (score, name_cos, addr_cos) as model features and
      `found` = bitmask of the searches that retrieved it (1 comb, 2 name, 4 addr).
      Rank is by combined score.

Usage: python blocking.py --split train --mode v2
Output: WORK_DIR/{split}_{mode}_cands_{country}.parquet
"""
import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize as l2norm
from sparse_dot_topn import sp_matmul_topn

import config

# channel -> (column, vectorizer kwargs, weight)
CHANNELS = {
    "v1": {
        "n": ("name_core", dict(token_pattern=r"\S+"), 1.0),
        "c": ("name_sq", dict(analyzer="char", ngram_range=(3, 3)), 0.6),
        "a": ("addr_norm", dict(token_pattern=r"\S+"), 0.5),
        "d": ("addr_nums", dict(token_pattern=r"\S+"), 0.8),
    },
    "v2": {
        "n": ("name_core", dict(token_pattern=r"\S+", ngram_range=(1, 2)), 1.0),
        "c": ("name_sq", dict(analyzer="char", ngram_range=(3, 3)), 0.6),
        "a": ("addr_norm", dict(token_pattern=r"\S+", ngram_range=(1, 2)), 0.5),
        "d": ("addr_nums", dict(token_pattern=r"\S+"), 0.8),
    },
}
SPACES = {"comb": "ncad", "name": "nc", "addr": "ad"}
K_V2 = {"comb": 10, "name": 6, "addr": 6}
Q_CHUNK = 400_000


def load_tables(split):
    """S1 table and the concatenated S2+S3 query table (with a `src` column)."""
    s1 = pd.read_parquet(config.prep_path(split, 1))
    s2 = pd.read_parquet(config.prep_path(split, 2))
    s3 = pd.read_parquet(config.prep_path(split, 3))
    s2["src"] = np.int8(2)
    s3["src"] = np.int8(3)
    q = pd.concat([s2, s3], ignore_index=True)
    return s1, q


class ChannelModel:
    """Per-channel vectorizers + pruned IDF weights, fit on one country's S1."""

    def __init__(self, S, channels, max_df_frac=0.003, min_max_df=2000):
        n = len(S)
        max_df = max(min_max_df, int(max_df_frac * n))
        self.parts = {}
        self.S = {}
        for ch, (col, kw, w) in channels.items():
            vec = CountVectorizer(binary=True, lowercase=False, dtype=np.float32, **kw)
            try:
                Xs = vec.fit_transform(S[col].values)
            except ValueError:  # empty vocabulary
                continue
            df = np.asarray(Xs.sum(0)).ravel()
            keep = np.where(df <= max_df)[0]
            idf = (np.log((n + 1) / (df[keep] + 1)) + 1).astype(np.float32) * w
            self.parts[ch] = (col, vec, keep, sp.diags(idf))
            self.S[ch] = (Xs[:, keep] @ self.parts[ch][3]).tocsr()

    def transform(self, Q):
        out = {}
        for ch, (col, vec, keep, D) in self.parts.items():
            out[ch] = (vec.transform(Q[col].values)[:, keep] @ D).tocsr()
        return out

    @staticmethod
    def space(mats, chans):
        m = [mats[c] for c in chans if c in mats]
        return l2norm(sp.hstack(m).tocsr()).astype(np.float32)


def _search(Mq, MsT, k):
    R = sp_matmul_topn(Mq, MsT, top_n=k, threshold=1e-4, sort=True, n_threads=config.N_JOBS).tocsr()
    sizes = np.diff(R.indptr)
    qi = np.repeat(np.arange(R.shape[0], dtype=np.int64), sizes)
    return qi, R.indices.astype(np.int64), R.data.astype(np.float32)


def _rowdot(A, B, qi, si, step=1_000_000):
    """Cosine of row pairs (A[qi], B[si]) for L2-normalized sparse rows."""
    out = np.empty(len(qi), np.float32)
    for j in range(0, len(qi), step):
        out[j:j + step] = np.asarray(A[qi[j:j + step]].multiply(B[si[j:j + step]]).sum(1)).ravel()
    return out


def block_country_v1(S, Q, k):
    cm = ChannelModel(S, CHANNELS["v1"])
    MsT = cm.space(cm.S, SPACES["comb"]).T.tocsr()
    parts = []
    for start in range(0, len(Q), Q_CHUNK):
        Mq = cm.space(cm.transform(Q.iloc[start:start + Q_CHUNK]), SPACES["comb"])
        qi, si, sc = _search(Mq, MsT, k)
        parts.append(pd.DataFrame({"q": (qi + start).astype(np.int32), "s": si.astype(np.int32), "score": sc}))
    c = pd.concat(parts, ignore_index=True)
    c["rank"] = c.groupby("q").cumcount().astype(np.int16)
    return c


def block_country_v2(S, Q, ks):
    cm = ChannelModel(S, CHANNELS["v2"])
    Ms = {name: cm.space(cm.S, chans) for name, chans in SPACES.items()}
    MsT = {name: m.T.tocsr() for name, m in Ms.items()}
    n_s = len(S)
    parts = []
    for start in range(0, len(Q), Q_CHUNK):
        t = time.time()
        qm = cm.transform(Q.iloc[start:start + Q_CHUNK])
        tt = [time.time() - t]
        Mq = {name: cm.space(qm, chans) for name, chans in SPACES.items()}
        del qm
        keys, bits = [], []
        for b, name in ((1, "comb"), (2, "name"), (4, "addr")):
            qi, si, _ = _search(Mq[name], MsT[name], ks[name])
            keys.append(qi * n_s + si)
            bits.append(np.full(len(qi), b, np.int8))
            tt.append(time.time() - t)
        keys, bits = np.concatenate(keys), np.concatenate(bits)
        uk, inv = np.unique(keys, return_inverse=True)
        found = np.zeros(len(uk), np.int8)
        np.bitwise_or.at(found, inv, bits)
        qi, si = uk // n_s, uk % n_s
        cos = {name: _rowdot(Mq[name], Ms[name], qi, si) for name in SPACES}
        order = np.lexsort((-cos["comb"], qi))
        c = pd.DataFrame({"q": (qi[order] + start).astype(np.int32), "s": si[order].astype(np.int32),
                          "score": cos["comb"][order], "name_cos": cos["name"][order],
                          "addr_cos": cos["addr"][order], "found": found[order]})
        c["rank"] = c.groupby("q").cumcount().astype(np.int16)
        parts.append(c)
        del Mq
        print(f"    chunk {start:,}: transform {tt[0]:.0f}s comb {tt[1]:.0f}s name {tt[2]:.0f}s "
              f"addr {tt[3]:.0f}s total {time.time() - t:.0f}s", flush=True)
    return pd.concat(parts, ignore_index=True)


def cands_path(split, mode, country):
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(country))
    return config.WORK_DIR / f"{split}_{mode}_cands_{safe}.parquet"


def load_cands(split, mode, k=None):
    """All countries' candidates; each query's rows are contiguous and rank-ordered."""
    files = sorted(config.WORK_DIR.glob(f"{split}_{mode}_cands_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no candidates for {split}/{mode} in {config.WORK_DIR}")
    filt = [("rank", "<", k)] if k else None
    return pd.concat([pd.read_parquet(f, filters=filt) for f in files], ignore_index=True)


def run(split, mode, k):
    t0 = time.time()
    s1, q = load_tables(split)
    print(f"loaded s1={len(s1):,} q={len(q):,} in {time.time() - t0:.0f}s", flush=True)
    for country in sorted(set(s1.country.unique()) | set(q.country.unique())):
        t = time.time()
        out = cands_path(split, mode, country)
        if out.exists():
            print(f"[{country}] exists, skipping")
            continue
        s_idx = np.where(s1.country.values == country)[0]
        q_idx = np.where(q.country.values == country)[0]
        if len(s_idx) == 0 or len(q_idx) == 0:
            print(f"[{country}] skipped (s1={len(s_idx)}, q={len(q_idx)})")
            continue
        S, Q = s1.iloc[s_idx], q.iloc[q_idx]
        c = block_country_v1(S, Q, k) if mode == "v1" else block_country_v2(S, Q, K_V2)
        c.insert(0, "q_row", q_idx[c.pop("q").values].astype(np.int32))
        c.insert(1, "s1_row", s_idx[c.pop("s").values].astype(np.int32))
        c.to_parquet(out, index=False)
        print(f"[{country}] s1={len(s_idx):,} q={len(q_idx):,} pairs={len(c):,} "
              f"({len(c) / len(q_idx):.1f}/query) in {time.time() - t:.0f}s", flush=True)
        del c
    if split == "train":
        report_recall(load_cands(split, mode), s1, q)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def load_truth(s1, q):
    """Array truth[q_row] = s1_row of the true S1 match, or -1 if unmatched."""
    gt = pd.read_csv(config.DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype=str,
                     keep_default_na=False)
    s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
    q_pos = pd.Series(np.arange(len(q)), index=q.entity_id.values)
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    ex = ex[ex.m != ""]
    truth = np.full(len(q), -1, dtype=np.int32)
    truth[q_pos.loc[ex.m.values].values] = s1_pos.loc[ex.source1_entity_id.values].values
    return truth


def report_recall(cand, s1, q):
    truth = load_truth(s1, q)
    np.save(config.WORK_DIR / "train_truth.npy", truth)
    n_pos = int((truth >= 0).sum())
    is_hit = truth[cand.q_row.values] == cand.s1_row.values
    hit = cand[is_hit]
    print(f"true pairs: {n_pos:,}  candidates/query: {len(cand) / len(q):.1f}")
    print(f"  recall (all candidates): {len(hit) / n_pos:.4f}")
    for kk in (1, 2, 3, 5, 10, 15, 20, 30):
        if kk <= cand["rank"].max() + 1:
            print(f"  recall@{kk}: {(hit['rank'] < kk).sum() / n_pos:.4f}")
    if "found" in cand:
        for b, name in ((1, "comb"), (2, "name"), (4, "addr")):
            print(f"  recall via {name}: {((hit.found & b) > 0).sum() / n_pos:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--mode", default="v2", choices=["v1", "v2"])
    ap.add_argument("--k", type=int, default=20, help="top-k for v1")
    a = ap.parse_args()
    run(a.split, a.mode, a.k)
