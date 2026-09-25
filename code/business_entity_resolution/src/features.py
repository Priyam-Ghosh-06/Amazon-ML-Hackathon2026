"""Stage C: pairwise comparison vector for every (query r, S1 s) candidate edge.
Stage E: leave-one-out view-corroboration features from same-(S1, source) siblings, given stage-1 p.
"""
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from common import nums

LLR_MIN_COUNT = 50   # tokens rarer than this get LLR 0, which also keeps OOF leakage negligible
P_MIN = 0.01         # edges below this stage-1 p get no number-level corroboration features (cost bound)


def _sims(a, b, prefix):
    out = {}
    for nm, sc in [('ratio', fuzz.ratio), ('tset', fuzz.token_set_ratio), ('tsort', fuzz.token_sort_ratio),
                   ('partial', fuzz.partial_ratio), ('jw', JaroWinkler.normalized_similarity)]:
        out[f'{prefix}_{nm}'] = process.cpdist(a, b, scorer=sc, workers=-1).astype(np.float32)
    return out


def _num_rel(nq, ns):
    """common / extra-in-r / missing-from-r counts, and the smallest |shift| between an extra and a missing number."""
    out = np.zeros((len(nq), 5), np.float32)
    for i, (a, b) in enumerate(zip(nq, ns)):
        ex, mi = a - b, b - a
        sh, rt = -1.0, -1.0
        if ex and mi:
            sh, x, y = min((abs(x - y), x, y) for x in ex for y in mi)
            rt = max(x, y) / max(min(x, y), 1)
        out[i] = (len(a & b), len(ex), len(mi), sh, rt)
    return out


def _extra_tokens(tq, ts):
    return [(set(a.split()) - set(b.split()), set(b.split()) - set(a.split())) for a, b in zip(tq, ts)]


def learn_llr(s1, q, pairs, y):
    """Per-token log-likelihood ratio of a token being extra/missing in a true pair vs a candidate non-pair."""
    tabs = {}
    for col in ('name', 'addr'):
        ex = _extra_tokens(q[col].values[pairs['qi']], s1[col].values[pairs['si']])
        for k, kind in enumerate(('extra', 'miss')):
            t = pd.DataFrame({'t': [list(e[k]) for e in ex], 'y': y}).explode('t').dropna()
            c = t.groupby(['t', 'y']).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
            c = c[c.sum(axis=1) >= LLR_MIN_COUNT]
            n0, n1 = (y == 0).sum(), (y == 1).sum()
            tabs[f'{col}_{kind}'] = (np.log((c[1] + 1) / n1) - np.log((c[0] + 1) / n0)).to_dict()
    return tabs


def graph_features(pairs, s1_name_group, q_src):
    """Blocking-graph context; needs the whole edge list, so it is computed once, not per chunk."""
    qi, si, cos = pairs['qi'].values, pairs['si'].values, pairs['cos'].values
    top1 = np.zeros(qi.max() + 1, np.float32); np.maximum.at(top1, qi, cos)
    f = {'cos': cos, 'rank': pairs['rank'].values.astype(np.float32), 'cos_top1': top1[qi]}
    f['cos_gap'] = f['cos_top1'] - cos
    f['n_cand_q'] = np.bincount(qi)[qi].astype(np.float32)
    f['s_degree'] = np.bincount(si)[si].astype(np.float32)
    f['s_top1_degree'] = np.bincount(si[pairs['rank'].values == 0], minlength=si.max() + 1)[si].astype(np.float32)
    f['s_name_group'] = s1_name_group[si].astype(np.float32)
    f['src'] = q_src[qi].astype(np.float32)
    return pd.DataFrame(f)


def pair_features(s1, q, pairs, llr):
    qi, si = pairs['qi'].values, pairs['si'].values
    f = {}
    for col in ('name', 'addr'):
        a, b = q[col].values[qi], s1[col].values[si]
        f.update(_sims(a, b, col))
        f[f'{col}_len_q'] = np.fromiter(map(len, a), np.float32, len(a))
        f[f'{col}_len_s'] = np.fromiter(map(len, b), np.float32, len(b))
        ex = _extra_tokens(a, b)
        for k, kind in enumerate(('extra', 'miss')):
            tab = llr[f'{col}_{kind}']
            ll = [[tab.get(t, 0.0) for t in e[k]] for e in ex]
            f[f'{col}_{kind}_n'] = np.fromiter((len(e[k]) for e in ex), np.float32, len(ex))
            f[f'{col}_{kind}_llr_sum'] = np.fromiter((sum(v) for v in ll), np.float32, len(ll))
            f[f'{col}_{kind}_llr_min'] = np.fromiter((min(v, default=0.0) for v in ll), np.float32, len(ll))
    nr = _num_rel([nums(x) for x in q['addr'].values[qi]], [nums(x) for x in s1['addr'].values[si]])
    for j, nm in enumerate(['num_common', 'num_extra', 'num_miss', 'num_shift', 'num_shift_ratio']):
        f[nm] = nr[:, j]
    return pd.DataFrame(f)


def _explode_nums(ids, addr):
    """(id, number) rows, one per distinct number in each address."""
    s = pd.Series(addr[ids]).str.findall(r'\d+').explode().dropna()
    return pd.DataFrame({'id': ids[s.index.values], 'n': s.str[-15:].astype(np.int64).values}).drop_duplicates()


def corroboration(pairs, p, q, s1, tau=0.5):
    """Stage E, one synchronous (Jacobi) round: every edge reads p from the previous round only, and r is always
    excluded from its own profile, so the output is independent of row order and cannot self-reinforce.

    n_sib_same / n_sib_cross / p_sib_sum   support of the (S1, source) star from confident members (p >= tau)
    p_q_other  best p of the same query to any other S1 (possible-match context)
    num_corr   max p(r'->s) over confident same-source siblings r' whose numbers contain all of r's extra numbers
    contra     max p(r'->s) over confident same-source siblings with no extra numbers and none of r's extras
    num_corr/contra: -1 when r has no extra numbers, -2 when p(r->s) < P_MIN (not computed).
    """
    qi, si = pairs['qi'].values, pairs['si'].values
    src = q['src'].values[qi].astype(np.int64)
    n_e = len(qi)
    conf = p >= tau
    key = si.astype(np.int64) * 2 + src
    ns = np.bincount(key, weights=conf, minlength=2 * (si.max() + 1))[key]
    na = np.bincount(si, weights=conf)[si]
    pa = np.bincount(si, weights=p * conf)[si]
    out = {'n_sib_same': ns - conf, 'n_sib_cross': na - ns, 'p_sib_sum': pa - p * conf}

    o = np.lexsort((-p, qi))                                         # per query, p descending
    first = np.r_[True, qi[o][1:] != qi[o][:-1]]
    top = np.zeros(qi.max() + 1); top[qi[o][first]] = p[o][first]
    sec = np.zeros(qi.max() + 1)
    second = np.r_[False, ~first[1:] & first[:-1]]                   # 2nd row of each query block
    sec[qi[o][second]] = p[o][second]
    is_top = np.zeros(n_e, bool); is_top[o[first]] = True
    out['p_q_other'] = np.where(is_top, sec[qi], top[qi])

    # number-level view corroboration, on edges with p >= P_MIN only
    ev = np.flatnonzero(p >= P_MIN)
    e = pd.DataFrame({'e': ev, 'qi': qi[ev], 'si': si[ev], 'src': src[ev], 'p': p[ev]})
    qn = _explode_nums(np.unique(e['qi'].values), q['addr'].values).rename(columns={'id': 'qi'})
    sn = _explode_nums(np.unique(e['si'].values), s1['addr'].values).rename(columns={'id': 'si'}).assign(in_s1=True)
    x = e.merge(qn, on='qi').merge(sn, on=['si', 'n'], how='left')
    x = x[x['in_s1'].isna()].drop(columns='in_s1')                   # r's numbers absent from its S1
    n_extra = np.bincount(x['e'].values, minlength=n_e)
    x['need'] = n_extra[x['e'].values]

    mem = e[e['p'] >= tau].rename(columns={'qi': 'qj', 'p': 'pj'})[['qj', 'si', 'src', 'pj', 'e']]
    mn = mem.merge(qn.rename(columns={'qi': 'qj'}), on='qj')
    hit = x[['e', 'qi', 'si', 'src', 'n', 'need']].merge(mn[['qj', 'si', 'src', 'pj', 'n']], on=['si', 'src', 'n'])
    hit = hit[hit['qi'] != hit['qj']]
    cnt = hit.groupby(['e', 'qj']).agg(k=('n', 'size'), pj=('pj', 'first'), need=('need', 'first')).reset_index()
    full = cnt[cnt['k'] >= cnt['need']].groupby('e')['pj'].max()

    base = np.full(n_e, -2.0); base[ev] = np.where(n_extra[ev] > 0, 0.0, -1.0)
    out['num_corr'] = base.copy(); out['num_corr'][full.index.values] = full.values

    clean = mem[n_extra[mem['e'].values] == 0][['qj', 'si', 'src', 'pj']]
    c = e[n_extra[ev] > 0][['e', 'qi', 'si', 'src']].merge(clean, on=['si', 'src'])
    c = c[c['qi'] != c['qj']]
    shared = cnt[['e', 'qj']].assign(sh=True)
    c = c.merge(shared, on=['e', 'qj'], how='left')
    cm = c[c['sh'].isna()].groupby('e')['pj'].max()
    out['contra'] = base.copy(); out['contra'][cm.index.values] = cm.values
    return pd.DataFrame(out).astype(np.float32)


if __name__ == '__main__':
    # T6.1: r is excluded from its own profile; corroboration only from a same-source sibling
    s1 = pd.DataFrame({'addr': ['100 main']})
    q = pd.DataFrame({'src': [0, 0, 1, 0], 'addr': ['100 main 7', '100 7 main', '7 100 main', '100 main 9']})
    pairs = pd.DataFrame({'qi': [0, 1, 2, 3], 'si': [0, 0, 0, 0]})
    p = np.array([0.9, 0.8, 0.9, 0.2])
    o = corroboration(pairs, p, q, s1)
    assert np.allclose(o['num_corr'], [0.8, 0.9, 0.0, 0.0]), o          # S3 sibling does not corroborate an S2 record
    assert o['n_sib_same'].tolist() == [1, 1, 0, 2] and o['n_sib_cross'].tolist() == [1, 1, 2, 1]
    assert o['contra'].tolist() == [0, 0, 0, 0]
    q.loc[1, 'addr'] = '100 main'                                      # sibling 1 agrees with S1 and lacks r0's 7
    o = corroboration(pairs, p, q, s1)
    assert np.isclose(o.loc[0, 'contra'], 0.8) and o.loc[0, 'num_corr'] == 0 and o.loc[1, 'num_corr'] == -1
    assert np.isclose(o.loc[3, 'contra'], 0.8)
    # T6.5: row order does not change the output
    perm = np.array([2, 0, 3, 1])
    o2 = corroboration(pairs.iloc[perm].reset_index(drop=True), p[perm], q, s1)
    assert np.allclose(o2.values, o.values[perm])
    # p_q_other: query 0 has two edges
    pr = pd.DataFrame({'qi': [0, 0, 1], 'si': [0, 0, 0]})
    assert np.allclose(corroboration(pr, np.array([0.7, 0.3, 0.1]), q, s1)['p_q_other'], [0.3, 0.7, 0.0])
    assert _num_rel([frozenset({22, 5})], [frozenset({11, 5})]).tolist() == [[1, 1, 1, 11, 2]]
    print('ok')
