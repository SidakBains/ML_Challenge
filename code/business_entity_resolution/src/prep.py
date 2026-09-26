"""Stage 1: read raw TSVs, normalize names/addresses in parallel, cache as parquet.

Usage: python prep.py --split train   (or test)
"""
import argparse
import csv
import json
import time
from multiprocessing import Pool

import pandas as pd

import config
import normalize as N

CHUNK = 20000


def read_raw(split, src):
    return pd.read_csv(config.raw_path(split, src), sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, encoding="utf-8")


def _init(translit_dict):
    N.set_translit_dict(translit_dict)


def _work(args):
    names, addrs = args
    rows = []
    for n, a in zip(names, addrs):
        rows.append(N.norm_name(n) + N.norm_addr(a))
    return rows


def normalize_frame(df, pool):
    names, addrs = df.business_name.tolist(), df.business_address.tolist()
    jobs = [(names[i:i + CHUNK], addrs[i:i + CHUNK]) for i in range(0, len(df), CHUNK)]
    rows = [r for part in pool.imap(_work, jobs) for r in part]
    cols = ["name_full", "name_core", "name_sq", "legal", "is_domain", "addr_norm", "addr_nums"]
    out = pd.DataFrame(rows, columns=cols)
    out.insert(0, "entity_id", df.entity_id.values)
    out.insert(1, "country", df.country.values)
    out["is_domain"] = out.is_domain.astype("int8")
    return out


def load_translit_dict():
    p = config.WORK_DIR / "translit_dict.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    args = ap.parse_args()
    tdict = load_translit_dict()
    print(f"translit dict entries: {len(tdict)}")
    with Pool(config.N_JOBS, initializer=_init, initargs=(tdict,)) as pool:
        for src in (1, 2, 3):
            t = time.time()
            df = read_raw(args.split, src)
            out = normalize_frame(df, pool)
            out.to_parquet(config.prep_path(args.split, src), index=False)
            print(f"{args.split} s{src}: {len(out):,} rows in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
