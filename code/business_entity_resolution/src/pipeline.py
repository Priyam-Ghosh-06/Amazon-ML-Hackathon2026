"""End-to-end pipeline v1 (docs/2026-pipeline-design-v1.md), stages A B C E F H.

    python src/pipeline.py train   # blocking + features + stage-1/2 OOF on train, fit final models, report OOF F0.5
    python src/pipeline.py test    # same stages on test with the train models, write output/*.tsv

Paths: ER_DATA (dataset dir with train/ and test/), ER_WORK (cache dir), ER_OUT (output dir).
Every expensive step is cached in ER_WORK; delete a file there to recompute that step.
"""
import os, sys, json, time, pickle
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from common import WORK, load_split, load_truth, f05_macro
import indic, blocking, features, decide

OUT = os.environ.get('ER_OUT', 'output')
NF = 4                      # OOF folds, grouped by the query's top-1 S1
TRAIN_ROWS = 6_000_000      # rows sampled per GBDT fit
FEAT_CHUNK = 2_000_000
TAUS = np.round(np.arange(0.3, 0.96, 0.05), 2)
PARAMS = dict(objective='binary', learning_rate=0.08, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
ROUNDS = 800
rng = np.random.default_rng(0)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def cached(name, fn):
    path = f'{WORK}/{name}.pkl'
    if os.path.exists(path):
        return pd.read_pickle(path)
    t = time.time(); obj = fn(); pd.to_pickle(obj, path)
    log(f'{name}: built in {time.time() - t:.0f}s')
    return obj


# ---------------------------------------------------------------- stage A + B
def prepare(split):
    s1, q = load_split(split)
    truth = load_truth(s1, q) if split == 'train' else None
    d = cached('indic_dict', lambda: indic.learn(s1, q, truth))   # learned on train pairs only
    q['name'] = indic.apply(q['name'].values, d)
    q['addr'] = indic.apply(q['addr'].values, d)
    log(f'{split}: S1={len(s1):,} queries={len(q):,} indic dict={len(d):,}')
    return s1, q, truth


def build_X(split, s1, q, pairs, llr):
    """Stage-1 feature matrix as a float32 memmap on disk (N_edges x F)."""
    path, cpath = f'{WORK}/{split}_X1.f32', f'{WORK}/{split}_X1.cols.json'
    if os.path.exists(cpath):
        cols = json.load(open(cpath))
        return np.memmap(path, np.float32, 'r', shape=(len(pairs), len(cols))), cols
    name_group = s1.groupby(['country', 'name'], observed=True)['name'].transform('size').values
    G = features.graph_features(pairs, name_group, q['src'].values)
    X = None
    for a in range(0, len(pairs), FEAT_CHUNK):
        F = pd.concat([G.iloc[a:a + FEAT_CHUNK].reset_index(drop=True),
                       features.pair_features(s1, q, pairs.iloc[a:a + FEAT_CHUNK], llr)], axis=1)
        if X is None:
            cols = list(F.columns)
            X = np.memmap(path, np.float32, 'w+', shape=(len(pairs), len(cols)))
        X[a:a + len(F)] = F.values
        log(f'  features {split}: {a + len(F):,}/{len(pairs):,}')
    X.flush(); json.dump(cols, open(cpath, 'w'))
    return np.memmap(path, np.float32, 'r', shape=(len(pairs), len(cols))), cols


# ---------------------------------------------------------------- stage C/E models
def rows(X, Z, idx):
    a = np.asarray(X[idx])
    return a if Z is None else np.hstack([a, Z[idx]])


def fit(X, Z, y, idx):
    idx = np.sort(rng.choice(idx, min(TRAIN_ROWS, len(idx)), replace=False))
    va = rng.random(len(idx)) < 0.03
    A, b = rows(X, Z, idx), y[idx]
    m = lgb.train(PARAMS, lgb.Dataset(A[~va], b[~va]), ROUNDS, valid_sets=[lgb.Dataset(A[va], b[va])],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    log(f'  fit: rows={len(idx):,} pos={b.mean():.3f} best_iter={m.best_iteration}')
    return m


def predict(m, X, Z, idx):
    out = np.empty(len(idx), np.float32)
    for a in range(0, len(idx), FEAT_CHUNK):
        out[a:a + FEAT_CHUNK] = m.predict(rows(X, Z, idx[a:a + FEAT_CHUNK]), num_iteration=m.best_iteration)
    return out


def oof(X, Z, y, fold):
    p = np.empty(len(y), np.float32)
    for k in range(NF):
        p[fold == k] = predict(fit(X, Z, y, np.flatnonzero(fold != k)), X, Z, np.flatnonzero(fold == k))
    return p


def stage2_inputs(pairs, p1, q, s1):
    Z = features.corroboration(pairs, p1, q, s1)
    Z.insert(0, 'p1', p1)
    return Z


# ---------------------------------------------------------------- stage F + H
def select(pairs, p, rule, n_s1):
    qi, si = pairs['qi'].values, pairs['si'].values
    if rule['kind'] == 'expected_f':
        return decide.choose_expected_f(qi, si, p, n_s1)
    return decide.choose_threshold(qi, si, p, rule['tau'])


def write_lists(path, header, s1_ids, si, ids):
    o = np.argsort(si, kind='stable')
    groups = np.split(ids[o], np.cumsum(np.bincount(si, minlength=len(s1_ids)))[:-1])
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\t'.join(header) + '\n')
        for s, g in zip(s1_ids, groups):
            f.write(f"{s}\t{','.join(g)}\n")


# ---------------------------------------------------------------- runs
def run_train():
    s1, q, truth = prepare('train')
    pairs = cached('train_pairs', lambda: blocking.block(s1, q, log))
    blocking.recall_report(pairs, truth, q, log)
    y = (truth[pairs['qi'].values] == pairs['si'].values).astype(np.int8)

    def _llr():
        s = np.sort(rng.choice(len(pairs), min(3_000_000, len(pairs)), replace=False))
        return features.learn_llr(s1, q, pairs.iloc[s], y[s])
    llr = cached('llr', _llr)
    X, cols = build_X('train', s1, q, pairs, llr)

    top1 = np.full(len(q), -1); r0 = pairs['rank'].values == 0
    top1[pairs['qi'].values[r0]] = pairs['si'].values[r0]
    fold = (np.random.default_rng(1).permutation(len(s1))[top1[pairs['qi'].values]] % NF).astype(np.int8)

    n_s1 = len(s1)
    def score(p, tag):
        best = None
        for tau in TAUS:
            sel = select(pairs, p, {'kind': 'tau', 'tau': tau}, n_s1)
            f = f05_macro(pairs['si'].values[sel], y[sel], truth, n_s1)
            best = max(best or (f, tau), (f, tau))
        sel = select(pairs, p, {'kind': 'expected_f'}, n_s1)
        fe = f05_macro(pairs['si'].values[sel], y[sel], truth, n_s1)
        log(f'OOF {tag}: macro-F0.5 expected-F rule={fe:.4f} | best fixed tau={best[1]} -> {best[0]:.4f}')
        return fe, best

    p1 = cached('train_p1', lambda: oof(X, None, y, fold))
    score(p1, 'stage-1')
    m1 = cached('model1', lambda: fit(X, None, y, np.arange(len(y))))

    Z = cached('train_Z', lambda: stage2_inputs(pairs, p1, q, s1))
    p2 = cached('train_p2', lambda: oof(X, Z.values, y, fold))
    m2 = cached('model2', lambda: fit(X, Z.values, y, np.arange(len(y))))

    sub = rng.choice(len(p2), min(5_000_000, len(p2)), replace=False)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(p2[sub], y[sub])
    pc = iso.predict(p2).astype(np.float32)
    fe, (ft, tau) = score(pc, 'stage-2 (isotonic)')
    rule = {'kind': 'expected_f'} if fe >= ft else {'kind': 'tau', 'tau': float(tau)}
    pickle.dump({'iso': iso, 'rule': rule, 'cols': cols + list(Z.columns)}, open(f'{WORK}/decision.pkl', 'wb'))
    imp = pd.Series(m2.feature_importance('gain'), index=cols + list(Z.columns)).sort_values(ascending=False)
    log('rule:', rule, '\nstage-2 top features (gain):\n' + imp.head(20).round(0).to_string())


def run_test():
    s1, q, _ = prepare('test')
    pairs = cached('test_pairs', lambda: blocking.block(s1, q, log))
    X, cols = build_X('test', s1, q, pairs, pd.read_pickle(f'{WORK}/llr.pkl'))
    m1, m2 = pd.read_pickle(f'{WORK}/model1.pkl'), pd.read_pickle(f'{WORK}/model2.pkl')
    dec = pickle.load(open(f'{WORK}/decision.pkl', 'rb'))
    idx = np.arange(len(pairs))
    p1 = predict(m1, X, None, idx)
    Z = stage2_inputs(pairs, p1, q, s1)
    p = dec['iso'].predict(predict(m2, X, Z.values, idx)).astype(np.float32)
    sel = select(pairs, p, dec['rule'], len(s1))

    os.makedirs(OUT, exist_ok=True)
    s1_ids, q_ids = s1['entity_id'].values, q['entity_id'].values
    si, qi = pairs['si'].values, pairs['qi'].values
    write_lists(f'{OUT}/candidate_pairs.tsv', ['source1_entity_id', 'candidate_entity_ids'], s1_ids, si, q_ids[qi])
    write_lists(f'{OUT}/matching_results.tsv', ['source1_entity_id', 'matched_entity_ids'], s1_ids, si[sel], q_ids[qi[sel]])
    cnt = np.bincount(si[sel], minlength=len(s1))
    log(f'test: edges={len(pairs):,} matches={sel.sum():,} empty S1 share={(cnt == 0).mean():.3f}; by country:')
    log(pd.DataFrame({'c': s1['country'].astype(str), 'n': cnt}).groupby('c')['n'].agg(['mean', lambda x: (x == 0).mean()]).round(3).to_string())


if __name__ == '__main__':
    os.makedirs(WORK, exist_ok=True)
    {'train': run_train, 'test': run_test}[sys.argv[1]]()
