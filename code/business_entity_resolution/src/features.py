"""Stage 3: pairwise features for (S2/S3 query, S1 candidate) pairs.

All features are country-agnostic similarity scores (no country one-hot), so
the model transfers to countries unseen in training (France).

Groups:
  * string similarity (rapidfuzz, multi-threaded) on names and addresses
  * IDF-weighted token overlap (IDF from Source-1), number/house-number agreement
  * blocking context: how this pair's score compares to the query's other
    candidates and to the S1 entity's other candidate queries
"""
import math
import os
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

import config

CHUNK = 2_000_000


# ---------------------------------------------------------------- IDF tables
def build_idf(s1):
    """IDF of name-core tokens and address tokens, fit on Source-1."""
    n = len(s1)
    cn, ca = Counter(), Counter()
    for s in s1.name_core.values:
        cn.update(set(s.split()))
    for s in s1.addr_norm.values:
        ca.update(set(s.split()))
    return ({t: math.log(n / c) for t, c in cn.items()},
            {t: math.log(n / c) for t, c in ca.items()},
            math.log(n))


# ------------------------------------------------------ python-loop features
_IDF = {}


def _init(idf_n, idf_a, idf_max):
    _IDF.update(n=idf_n, a=idf_a, max=idf_max)


def _wsets(a, b, idf, dflt):
    """IDF-weighted jaccard and containment of a's tokens in b."""
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return np.nan, np.nan, np.nan
    w = lambda s: sum(idf.get(t, dflt) for t in s)
    inter = w(sa & sb)
    union = w(sa | sb)
    return inter / union if union else 0.0, inter / (w(sa) or 1.0), inter / (w(sb) or 1.0)


def _loop_chunk(args):
    nc_a, nc_b, ad_a, ad_b, nu_a, nu_b, lg_a, lg_b = args
    idf_n, idf_a, dflt = _IDF["n"], _IDF["a"], _IDF["max"]
    out = np.full((len(nc_a), 11), np.nan, dtype=np.float32)
    for i in range(len(nc_a)):
        out[i, 0:3] = _wsets(nc_a[i], nc_b[i], idf_n, dflt)
        out[i, 3:6] = _wsets(ad_a[i], ad_b[i], idf_a, dflt)
        na, nb = set(nu_a[i].split()), set(nu_b[i].split())
        if na and nb:
            inter = len(na & nb)
            out[i, 6] = inter / len(na | nb)
            out[i, 7] = inter
            out[i, 8] = inter / len(na)
        la, lb = lg_a[i], lg_b[i]
        if la and lb:
            out[i, 9] = float(la == lb)
            out[i, 10] = float(bool(set(la.split()) & set(lb.split())))
    return out


LOOP_NAMES = ["n_wjac", "n_wcont_s1", "n_wcont_q", "a_wjac", "a_wcont_s1", "a_wcont_q",
              "num_jac", "num_inter", "num_cont_s1", "legal_eq", "legal_overlap"]


# ------------------------------------------------------------------- main API
def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _first_num(addr):
    return addr.str.extract(r"(?:^|\s)(\d+)(?:\s|$)", expand=False)


class Context:
    """Blocking-score context computed once over the FULL candidate table.

    cands must be sorted by (q_row, rank). Query-side: score vs the query's best
    and second-best candidate. S1-side: how many queries list this S1, how many
    have it as their top-1, and this pair's rank among the S1's queries.
    Implemented with numpy (bincount / lexsort) to stay memory-friendly on
    ~100M+ pairs.
    """

    def __init__(self, cands):
        q = cands.q_row.values
        s = cands.s1_row.values
        sc = cands.score.values
        rk = cands["rank"].values
        n = len(cands)
        first = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1]
        sizes = np.diff(np.r_[first, n])
        grp_first = np.repeat(first, sizes)
        self.top1 = sc[grp_first]
        nxt = np.minimum(grp_first + 1, n - 1)
        self.second = np.where(np.repeat(sizes, sizes) > 1, sc[nxt], 0).astype(np.float32)
        self.q_ncand = np.repeat(sizes, sizes).astype(np.int16)
        n_s1 = int(s.max()) + 1
        self.s1_ncand = np.bincount(s, minlength=n_s1).astype(np.int32)
        self.s1_ntop1 = np.bincount(s[rk == 0], minlength=n_s1).astype(np.int32)
        s1_max = np.zeros(n_s1, np.float32)
        np.maximum.at(s1_max, s, sc)
        self.s1_max = s1_max
        order = np.lexsort((-sc, s))
        ss = s[order]
        st = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
        grp = np.repeat(st, np.diff(np.r_[st, n]))
        s1_rank = np.empty(n, np.int32)
        s1_rank[order] = np.arange(n) - grp
        self.s1_rank = s1_rank
        del order, ss, grp

    def features(self, pairs):
        """Context features for a subset of rows (pairs.index = row ids in cands)."""
        i = pairs.index.values
        sc, rk, s = pairs.score.values, pairs["rank"].values, pairs.s1_row.values
        f = pd.DataFrame(index=pairs.index)
        f["score"] = sc
        f["rank"] = rk
        f["q_ncand"] = self.q_ncand[i]
        f["top1"] = self.top1[i]
        f["gap_top1"] = self.top1[i] - sc
        f["margin"] = np.where(rk == 0, sc - self.second[i], sc - self.top1[i])
        f["s1_ncand"] = self.s1_ncand[s]
        f["s1_n_top1"] = self.s1_ntop1[s]
        f["s1_gap_top"] = self.s1_max[s] - sc
        f["s1_rank"] = self.s1_rank[i]
        for extra in ("name_cos", "addr_cos", "found"):  # present for blocking v2
            if extra in pairs:
                f[extra] = pairs[extra].values
        return f.astype(np.float32)


def add_first_num(df):
    """Cache the first standalone number of the address (house number)."""
    if "first_num" not in df:
        df["first_num"] = _first_num(df.addr_norm)


def compute_features(pairs, s1, q, pool, context):
    """pairs: rows of the candidate table (q_row, s1_row, score, rank).

    Returns a float32 feature DataFrame aligned with pairs.index.
    """
    ctx = context.features(pairs)
    add_first_num(s1)
    add_first_num(q)
    s1_first, q_first = s1.first_num, q.first_num
    blocks = []
    for start in range(0, len(pairs), CHUNK):
        p = pairs.iloc[start:start + CHUNK]
        ia, ib = p.s1_row.values, p.q_row.values
        col = lambda df, c, idx: df[c].values[idx].tolist()
        A = {c: col(s1, c, ia) for c in ("name_core", "name_sq", "name_full", "addr_norm", "addr_nums", "legal")}
        B = {c: col(q, c, ib) for c in ("name_core", "name_sq", "name_full", "addr_norm", "addr_nums", "legal")}
        f = {}
        f["n_ratio"] = _cp(A["name_core"], B["name_core"], fuzz.ratio)
        f["n_tset"] = _cp(A["name_core"], B["name_core"], fuzz.token_set_ratio)
        f["n_tsort"] = _cp(A["name_core"], B["name_core"], fuzz.token_sort_ratio)
        f["n_partial"] = _cp(A["name_core"], B["name_core"], fuzz.partial_ratio)
        f["sq_jw"] = _cp(A["name_sq"], B["name_sq"], JaroWinkler.normalized_similarity)
        f["sq_ratio"] = _cp(A["name_sq"], B["name_sq"], fuzz.ratio)
        f["sq_partial"] = _cp(A["name_sq"], B["name_sq"], fuzz.partial_ratio)
        f["full_tset"] = _cp(A["name_full"], B["name_full"], fuzz.token_set_ratio)
        f["a_ratio"] = _cp(A["addr_norm"], B["addr_norm"], fuzz.ratio)
        f["a_tset"] = _cp(A["addr_norm"], B["addr_norm"], fuzz.token_set_ratio)
        f["a_tsort"] = _cp(A["addr_norm"], B["addr_norm"], fuzz.token_sort_ratio)
        f["a_partial"] = _cp(A["addr_norm"], B["addr_norm"], fuzz.partial_ratio)
        a_empty = np.array([not s for s in B["addr_norm"]])
        for k in ("a_ratio", "a_tset", "a_tsort", "a_partial"):
            f[k][a_empty] = np.nan
        f["q_addr_empty"] = a_empty.astype(np.float32)
        f["n_len_s1"] = np.fromiter((len(s) for s in A["name_sq"]), np.float32, len(ia))
        f["n_len_q"] = np.fromiter((len(s) for s in B["name_sq"]), np.float32, len(ia))
        f["a_len_s1"] = np.fromiter((len(s) for s in A["addr_norm"]), np.float32, len(ia))
        f["a_len_q"] = np.fromiter((len(s) for s in B["addr_norm"]), np.float32, len(ia))
        fa = pd.Series(s1_first.values[ia])
        fb = pd.Series(q_first.values[ib])
        has = (fa.notna() & fb.notna()).values
        f["house_eq"] = np.where(has, (fa.fillna("") == fb.fillna("")).values, np.nan).astype(np.float32)
        f["q_is_domain"] = q.is_domain.values[ib].astype(np.float32)
        f["src"] = q.src.values[ib].astype(np.float32)

        # python-loop features in parallel
        step = 50_000
        jobs = [tuple(x[j:j + step] for x in (A["name_core"], B["name_core"], A["addr_norm"], B["addr_norm"],
                                              A["addr_nums"], B["addr_nums"], A["legal"], B["legal"]))
                for j in range(0, len(ia), step)]
        loop = np.vstack(pool.map(_loop_chunk, jobs))
        for j, name in enumerate(LOOP_NAMES):
            f[name] = loop[:, j]
        blk = pd.DataFrame(f, index=p.index)
        blocks.append(blk)
        del A, B
    F = pd.concat(blocks)
    F = pd.concat([ctx, F], axis=1)
    F["n_x_a"] = F["n_tset"] * F["a_tset"].fillna(50) / 100.0
    return F.astype(np.float32)


def make_pool(idf):
    # One BLAS thread per worker: 24 workers x 24 OpenBLAS threads exhausts memory.
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    return Pool(config.N_JOBS, initializer=_init, initargs=idf)
