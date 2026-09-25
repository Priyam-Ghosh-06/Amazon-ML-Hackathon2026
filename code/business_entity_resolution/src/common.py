"""Shared I/O, text normalisation and the macro-F0.5 metric."""
import os, re, unicodedata
import numpy as np, pandas as pd, regex as rx

DATA = os.environ.get('ER_DATA', 'data/student_resource/dataset')
WORK = os.environ.get('ER_WORK', 'work')

_latin_marks = rx.compile(r'(?<=\p{Latin})\p{Mn}+')     # strip accents on Latin only; Indic matras/viramas survive
_nonword = rx.compile(r'[^\p{L}\p{M}\p{N}]+')
_nonlatin = rx.compile(r'(?!\p{Latin})\p{L}')
_digits = re.compile(r'\d+')
NULLS = {'', 'null', 'n a', 'na', 'none', 'nan'}


def read_tsv(path):
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, quoting=3)


def norm(s):
    s = unicodedata.normalize('NFD', s.lower().replace('&', ' and '))
    s = _nonword.sub(' ', _latin_marks.sub('', s)).strip()
    return '' if s in NULLS else s


def nums(s):
    """Address numbers as a frozenset of ints; leading zeros folded, long runs truncated."""
    return frozenset(int(x[-15:]) for x in _digits.findall(s))


def has_nonlatin(s):
    return _nonlatin.search(s) is not None


def load_split(split):
    """S1 and the S2+S3 query table, with normalised name/address. Cached as pickle."""
    os.makedirs(WORK, exist_ok=True)
    cache = f'{WORK}/{split}_records.pkl'
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    s1 = read_tsv(f'{DATA}/{split}/{split}_source1.tsv')
    q = pd.concat([read_tsv(f'{DATA}/{split}/{split}_source{k}.tsv') for k in (2, 3)], ignore_index=True)
    for df in (s1, q):
        df['name'] = df['business_name'].map(norm)
        df['addr'] = df['business_address'].map(norm)
        df.drop(columns=['business_name', 'business_address'], inplace=True)
        df['country'] = df['country'].astype('category')
    q['src'] = (q['entity_id'].str[1] == '3').astype('int8')   # 0 = S2, 1 = S3
    pd.to_pickle((s1, q), cache)
    return s1, q


def load_truth(s1, q):
    """Array: for every query row, the S1 row index it matches, or -1 (decoy/unmatched)."""
    gt = read_tsv(f'{DATA}/train/train_ground_truth.tsv')
    g = gt.assign(r=gt['matched_entity_ids'].str.split(',')).explode('r')
    g = g[g['r'].fillna('') != '']
    s1_pos = pd.Series(np.arange(len(s1)), index=s1['entity_id'])
    q_pos = pd.Series(np.arange(len(q)), index=q['entity_id'])
    truth = np.full(len(q), -1, np.int64)
    truth[q_pos[g['r']].values] = s1_pos[g['source1_entity_id']].values
    return truth


def f05_macro(sel_si, sel_ok, truth, n_s1):
    """Macro F0.5 over ALL n_s1 S1 entities (singletons included).
    sel_si: S1 index of each predicted match; sel_ok: whether that match is correct; truth: S1 index per query or -1."""
    tp = np.bincount(sel_si, weights=sel_ok, minlength=n_s1)
    npred = np.bincount(sel_si, minlength=n_s1)
    ntrue = np.bincount(truth[truth >= 0], minlength=n_s1)
    den = 1.25 * tp + 0.25 * (ntrue - tp) + (npred - tp)
    f = np.where(ntrue == 0, (npred == 0).astype(float), 1.25 * tp / np.maximum(den, 1e-9))
    return float(f.mean())


if __name__ == '__main__':
    assert norm('Shiv TÉCHNOLOGY & Co.') == 'shiv technology and co'
    assert norm('हाईटेक फाइनेंस') == 'हाईटेक फाइनेंस'          # matras survive
    assert norm('NULL') == '' and nums('Tc 84/2259(1), 007') == {84, 2259, 1, 7}
    # PS worked example: P=2/3, R=1 -> 0.714
    # S1 0: predict 3, 2 correct, 2 true -> 0.714; S1 1: singleton predicted empty -> 1; S1 2: singleton + 1 FP -> 0
    f = f05_macro(np.array([0, 0, 0, 2]), np.array([1, 1, 0, 0]), np.array([0, 0, -1]), 3)
    assert abs(f - (0.7142857 + 1 + 0) / 3) < 1e-6
    print('ok')
