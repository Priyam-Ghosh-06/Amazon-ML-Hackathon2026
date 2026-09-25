"""Indic-script -> Latin token dictionary, learned from training pairs only (Dice co-occurrence alignment).

S2/S3 India records spell names/states token by token in Devanagari, Bengali, Tamil, ...; S1 is always Latin.
The vocabulary is small, so a word-level dictionary beats generic transliteration (R@1 5% -> ~90% in the EDA).
"""
import numpy as np, pandas as pd
from common import has_nonlatin

MIN_PAIRS = 3


def learn(s1, q, truth):
    rows = []
    pos = np.flatnonzero(truth >= 0)
    for col in ('name', 'addr'):
        qt = q[col].values[pos]
        m = np.fromiter((has_nonlatin(t) for t in qt), bool, len(qt))
        a = pd.DataFrame({'pair': pos[m] * 2 + (col == 'addr'), 'q': qt[m], 's': s1[col].values[truth[pos[m]]]})
        a['q'] = a['q'].str.split(); a['s'] = a['s'].str.split()
        a = a.explode('q').explode('s').dropna()
        a = a[a['q'].map(has_nonlatin) & ~a['s'].map(has_nonlatin)].drop_duplicates()
        rows.append(a)
    a = pd.concat(rows, ignore_index=True)
    c = a.groupby(['q', 's']).size().rename('c').reset_index()
    cq = a.drop_duplicates(['pair', 'q']).groupby('q').size()
    cs = a.drop_duplicates(['pair', 's']).groupby('s').size()
    c['dice'] = 2 * c['c'] / (c['q'].map(cq).values + c['s'].map(cs).values)
    c = c[c['c'] >= MIN_PAIRS].sort_values('dice', ascending=False).drop_duplicates('q')
    return dict(zip(c['q'], c['s']))


def apply(texts, d):
    """Replace every known Indic token; unknown tokens are kept as-is."""
    return [' '.join(d.get(t, t) for t in s.split()) if has_nonlatin(s) else s for s in texts]


if __name__ == '__main__':
    s1 = pd.DataFrame({'name': ['hitech finance', 'ram marketing', 'hitech traders'], 'addr': ['lucknow', 'delhi', 'delhi']})
    q = pd.DataFrame({'name': ['हाईटेक फाइनेंस', 'राम मार्केटिंग', 'हाईटेक ट्रेडर्स'] * 3, 'addr': ['lucknow', 'दिल्ली', 'दिल्ली'] * 3})
    d = learn(s1, q, np.array([0, 1, 2] * 3))
    assert d['हाईटेक'] == 'hitech' and d['दिल्ली'] == 'delhi', d
    assert apply(['हाईटेक फाइनेंस x', 'abc'], d) == ['hitech finance x', 'abc']
    print('ok', len(d))
