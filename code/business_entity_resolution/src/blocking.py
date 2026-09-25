"""Stage B: candidate graph. Every S2/S3 record (query) retrieves its top-K S1 within its own country.

Word uni+bigram TF-IDF (IDF from S1) with n-grams of document frequency > MAX_DF dropped. Dropping the common
n-grams bounds the sparse-product cost per query; the kept rare tokens/bigrams (business-name words,
"614 shore", ...) carry the ranking. Measured on US: R@5 97.3%, R@20 98.4%, ~0.3 h per 10M queries on 12 threads.
Exact char-3gram cosine was ~100x slower, and a df-pruned char-3gram index collapsed to R@20 43%.
"""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

K = 6            # candidates per query fed to the matcher
MAX_DF = 5000
CHUNK = 500_000


def block(s1, q, log=print):
    ts = (s1['name'] + ' ' + s1['addr']).values
    tq = (q['name'] + ' ' + q['addr']).values
    parts = []
    for c in sorted(set(q['country'].astype(str))):          # open set of countries, France included
        si = np.flatnonzero((s1['country'] == c).values)
        qi = np.flatnonzero((q['country'] == c).values)
        if len(si) == 0 or len(qi) == 0:
            continue
        v = TfidfVectorizer(analyzer='word', token_pattern=r'\S+', ngram_range=(1, 2), dtype=np.float32,
                            sublinear_tf=True, lowercase=False)
        X = v.fit_transform(ts[si])
        keep = np.bincount(X.indices, minlength=X.shape[1]) <= MAX_DF
        XT = X[:, keep].T.tocsr()
        for a in range(0, len(qi), CHUNK):
            b = qi[a:a + CHUNK]
            M = sp_matmul_topn(v.transform(tq[b])[:, keep], XT, top_n=K, threshold=1e-6, n_threads=-1, sort=True)
            n = np.diff(M.indptr)
            parts.append(pd.DataFrame({'qi': np.repeat(b, n).astype(np.int32), 'si': si[M.indices].astype(np.int32),
                                       'cos': M.data.astype(np.float32),
                                       'rank': np.concatenate([np.arange(k) for k in n]).astype(np.int8) if len(n) else []}))
        log(f'  block {c}: S1={len(si):,} queries={len(qi):,} vocab kept={keep.sum():,}/{len(keep):,}')
    return pd.concat(parts, ignore_index=True)


def recall_report(pairs, truth, q, log=print):
    pos = truth >= 0
    hit = np.zeros(len(truth), np.int8) - 1
    t = pairs[truth[pairs['qi'].values] == pairs['si'].values]
    hit[t['qi'].values] = t['rank'].values
    for c in sorted(set(q['country'].astype(str))):
        m = pos & (q['country'] == c).values
        log(f'  recall {c}: ' + ' '.join(f'R@{k}={((hit[m] >= 0) & (hit[m] < k)).mean():.4f}' for k in (1, 3, K)))
    log(f'  pairs/query={len(pairs) / len(truth):.2f}')


if __name__ == '__main__':
    s1 = pd.DataFrame({'name': ['acme corp', 'zen labs', 'acme corp'], 'addr': ['1 main st', '5 elm rd', '9 oak ave'],
                       'country': ['US', 'US', 'France']})
    q = pd.DataFrame({'name': ['acme corporation', 'zen lab', 'acme corp'], 'addr': ['1 main street', '5 elm rd', '9 oak ave'],
                      'country': ['US', 'US', 'France']})
    p = block(s1, q, log=lambda *_: None)
    top = p[p['rank'] == 0].set_index('qi')['si']
    assert top.to_dict() == {0: 0, 1: 1, 2: 2}, p          # France query never retrieves a US S1
    print('ok')
