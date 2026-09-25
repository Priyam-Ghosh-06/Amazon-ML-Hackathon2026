"""Stage F: from calibrated edge probabilities to one match set per S1.

1. Many-to-one: every S2/S3 record belongs to at most one S1, so a query only keeps its best edge.
2. Per S1, sort the kept edges by p and take the prefix k maximising expected F0.5:
     E[F | k>0] ~= 1.25 * sum_{i<=k} p_i / (k + 0.25 * N),  N = sum of p over ALL edges into the S1
     E[F | k=0]  = prod(1 - p_i)                               (the S1 is a singleton)
   ponytail: ratio-of-expectations instead of the exact Poisson-binomial expectation. Swap it in if the
   gap to the best fixed threshold on OOF looks like approximation error.
"""
import numpy as np, pandas as pd


def best_edges(qi, si, p):
    o = np.lexsort((-p, qi))
    first = np.r_[True, qi[o][1:] != qi[o][:-1]]
    return o[first]


def choose_expected_f(qi, si, p, n_s1):
    """Boolean mask over edges: selected matches."""
    N = np.bincount(si, weights=p, minlength=n_s1)
    log_empty = np.bincount(si, weights=np.log1p(-np.clip(p, 0, 1 - 1e-7)), minlength=n_s1)
    b = best_edges(qi, si, p)
    o = b[np.lexsort((-p[b], si[b]))]                           # kept edges, grouped by S1, p descending
    s, pp = si[o], p[o]
    start = np.r_[True, s[1:] != s[:-1]]
    gid = np.cumsum(start) - 1
    k = np.arange(len(o)) - np.flatnonzero(start)[gid] + 1
    cs = np.cumsum(pp); cs -= np.r_[0, cs[np.flatnonzero(start)[1:] - 1]][gid]
    ef = 1.25 * cs / (k + 0.25 * N[s])
    best_ef = np.maximum.reduceat(ef, np.flatnonzero(start)) if len(o) else ef
    keep_any = best_ef > np.exp(log_empty[s[start]])
    # first k reaching the group max
    is_best = (ef == best_ef[gid]) & keep_any[gid]
    kk = np.where(is_best, k, np.iinfo(np.int64).max)
    best_k = np.minimum.reduceat(kk, np.flatnonzero(start)) if len(o) else kk
    best_k = np.where(keep_any, best_k, 0)                      # empty set wins -> select nothing
    sel = np.zeros(len(p), bool)
    sel[o[k <= best_k[gid]]] = True
    return sel


def choose_threshold(qi, si, p, tau):
    sel = np.zeros(len(p), bool)
    b = best_edges(qi, si, p)
    sel[b[p[b] >= tau]] = True
    return sel


if __name__ == '__main__':
    # S1 0: two strong edges and one weak -> take the two. S1 1: one weak edge -> empty (likely singleton).
    # Query 3 has a better edge to S1 0 than to S1 1, so its S1 1 edge can never be selected.
    qi = np.array([0, 1, 2, 3, 3]); si = np.array([0, 0, 0, 0, 1]); p = np.array([.95, .9, .2, .3, .25])
    sel = choose_expected_f(qi, si, p, 2)
    assert sel.tolist() == [True, True, False, False, False], sel
    assert choose_threshold(qi, si, p, 0.25).tolist() == [True, True, False, True, False]
    assert choose_expected_f(np.array([0]), np.array([1]), np.array([0.7]), 2).tolist() == [True]
    # two weak edges into one S1: the empty set wins, nothing is selected
    assert not choose_expected_f(np.array([0, 1]), np.array([0, 0]), np.array([0.01, 0.02]), 1).any()
    print('ok')
