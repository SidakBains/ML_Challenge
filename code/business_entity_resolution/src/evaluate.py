"""Exact challenge metric: per-S1-entity F0.5, macro-averaged over S1 entities.

Singletons (no true matches) score 1.0 when predicted empty, else 0.0.
An entity with true matches but an empty prediction scores 0.0.
"""
import numpy as np


def macro_f05(eval_s1, assigned_q, assigned_s1, truth, beta=0.5):
    """
    eval_s1:     array of S1 row ids to evaluate on
    assigned_q:  query rows that were assigned a match
    assigned_s1: the S1 row each of those queries was assigned to
    truth:       truth[q_row] = true S1 row, or -1
    """
    eval_s1 = np.asarray(eval_s1)
    n = int(max(eval_s1.max(), assigned_s1.max() if len(assigned_s1) else 0, truth.max())) + 1
    in_eval = np.zeros(n, bool)
    in_eval[eval_s1] = True

    ntrue = np.bincount(truth[(truth >= 0)], minlength=n)
    npred = np.bincount(assigned_s1, minlength=n)
    tp = np.bincount(assigned_s1[truth[assigned_q] == assigned_s1], minlength=n)

    nt, npd, t = ntrue[eval_s1], npred[eval_s1], tp[eval_s1].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(npd > 0, t / npd, 0.0)
        r = np.where(nt > 0, t / nt, 0.0)
        b2 = beta * beta
        f = np.where((p + r) > 0, (1 + b2) * p * r / (b2 * p + r), 0.0)
    f = np.where(nt == 0, (npd == 0).astype(float), f)
    return float(f.mean()), dict(
        f05=float(f.mean()), precision=float(p[npd > 0].mean()) if (npd > 0).any() else 0.0,
        recall=float(r[nt > 0].mean()) if (nt > 0).any() else 0.0,
        singleton_acc=float(f[nt == 0].mean()) if (nt == 0).any() else float("nan"),
        n_entities=int(len(eval_s1)),
    )


if __name__ == "__main__":
    # Example from the problem statement: pred 3, truth 2, 2 correct -> 0.714
    truth = np.array([0, -1, 0])  # q0 and q2 are the true matches of s1=0
    f, info = macro_f05(np.array([0]), np.array([0, 1, 2]), np.array([0, 0, 0]), truth)
    print(round(f, 3), info)
