"""Learn a {non-latin token -> latin token} dictionary from TRAINING matches only.

Some S2/S3 records write names/states in Devanagari, Kannada, etc. (e.g.
'लाइफ इन्वेस्टमेंट्स' for S1 'Life Investments'). For every ground-truth pair
where the S2/S3 side has non-latin tokens we count co-occurrences with the S1
side's latin tokens (position-aligned when token counts agree, bag-of-words
otherwise) and pick, per non-latin token, the latin token maximizing
count * (0.5 + similarity to the generic anyascii romanization).

Output: WORK_DIR/translit_dict.json  (used by prep.py)
"""
import json
import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd
from anyascii import anyascii
from rapidfuzz.distance import JaroWinkler

import config
from prep import read_raw

from normalize import ASCII_PUNCT, is_nonlatin

TOK = re.compile(r"[^\s,]+")


def toks(s):
    # Same tokenization as normalize.translit(): split on whitespace/commas,
    # strip ASCII punctuation only (keeps Indic vowel signs intact).
    s = unicodedata.normalize("NFKC", s)
    return [t.strip(ASCII_PUNCT) for t in TOK.findall(s)]


def main():
    s1 = read_raw("train", 1).set_index("entity_id")
    others = []
    for src in (2, 3):
        d = read_raw("train", src)
        mask = ~(d.business_name.map(str.isascii) & d.business_address.map(str.isascii))
        others.append(d[mask])
    q = pd.concat(others).set_index("entity_id")
    print("non-ascii S2/S3 records:", len(q))

    gt = pd.read_csv(config.DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype=str,
                     keep_default_na=False)
    q_to_s1 = {}
    qset = set(q.index)
    for s1id, ids in zip(gt.source1_entity_id, gt.matched_entity_ids):
        for i in ids.split(","):
            if i in qset:
                q_to_s1[i] = s1id

    counts = defaultdict(Counter)
    s1n, s1a = s1.business_name.to_dict(), s1.business_address.to_dict()
    qn, qa = q.business_name.to_dict(), q.business_address.to_dict()
    for qid, s1id in q_to_s1.items():
        for fq, fs in ((qn[qid], s1n[s1id]), (qa[qid], s1a[s1id])):
            tq, ts = toks(fq), [t.lower() for t in toks(fs)]
            ts = [t for t in ts if t and t.isascii()]
            if not ts:
                continue
            aligned = len(tq) == len(ts)
            for i, t in enumerate(tq):
                if not t or not is_nonlatin(t):
                    continue
                if aligned:
                    counts[t][ts[i]] += 3.0
                for u in set(ts):
                    counts[t][u] += 1.0

    out = {}
    for t, c in counts.items():
        rom = anyascii(t).lower()
        best, best_score = None, 0.0
        total = sum(c.values())
        for u, n in c.most_common(30):
            score = n * (0.5 + JaroWinkler.normalized_similarity(rom, u))
            if score > best_score:
                best, best_score = u, score
        if best is not None and (total >= 4 or JaroWinkler.normalized_similarity(rom, best) >= 0.6):
            out[t] = best
    (config.WORK_DIR / "translit_dict.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("dictionary entries:", len(out))
    for k in list(out)[:40]:
        print(f"  {k} -> {out[k]}   (anyascii: {anyascii(k).lower()})")


if __name__ == "__main__":
    main()
