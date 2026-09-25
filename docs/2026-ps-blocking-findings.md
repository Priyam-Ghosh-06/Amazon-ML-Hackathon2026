# 2026 PS (Business Entity Resolution): blocking EDA findings

Measured on a 5% S1 sample of train (110k S1, 382k true pairs) plus the full train S1 as the retrieval index. The recall experiments used 3,300 stratified queries (1,500 US, 1,000 India-Latin, 800 India-Indic-script). Because India-Indic is oversampled, "overall" numbers are not population-weighted.

## Raw facts
- Train: S1 2.21M, S2 5.03M, S3 5.29M. Test: S1 1.73M, S2 4.89M, S3 5.08M. France is only in the test set (259k S1).
- Matches per S1 average about 3.5. Singletons are 5.6% of S1.
- Country agrees on 100% of true pairs, so a hard partition by country is safe.
- Every S2/S3 record matches at most one S1 (0 violations in the samples). The mapping is many-to-one.
- About 26% of S2/S3 records match no S1. These are decoys: a near-copy of some S1 with a perturbed house number (22 vs 11, 9102 vs 9101) and/or an extra modifier token ("East", "Greater", "Overseas", "Riverside"). Their top-1 char-3gram cosine to S1 has median about 0.71 (US).
- S1 names are not unique: 36% of US and 44% of India S1 share their exact normalized name (largest group 253). The (name, address) pair is unique.
- Addresses are empty in about 3.5–4.5% of S2/S3 records. S1 is never empty and never uses a non-Latin script.
- Non-Latin scripts in S2/S3 (India only): Devanagari, Bengali, Tamil, Telugu, Kannada, Gujarati, Malayalam, Gurmukhi, Oriya. They appear in names (about 13% of India S2 records use Devanagari) and in state names inside addresses. The India Indic-name share is about 10% in train and 18% in test (a distribution shift).
- The Indic names are phonetic transliterations of English words, token by token. The vocabulary is small: 1,273 Indic name tokens across 25k pairs.

## Recall results (S2/S3 record as query, retrieve its S1)
- Concatenated char-3gram TF-IDF, recall@50: US 98.2, IN-latin 95.5, IN-indic 85.6.
- Name-only channel on IN-indic. With anyascii transliteration: R@1 5%. With a token dictionary learned from ground-truth pairs (positional alignment): R@1 91.5%. Coverage of test Indic tokens by a dictionary built from the 5% train sample: 93%.
- Pitfall: rank-based recall with ties is optimistic (R@1 97.8% vs 92.5% with honest top-k). The name channel has top-score ties for 61% of queries.
- Missing-aware Pareto skyline over (name sim, address sim), with empty address reduced to name-only ties. Layer 0: recall 97.4%, 2.8 candidates per query. Layers ≤1: recall 98.3%, 9.8 candidates. By group (L0): US 98.3, IN-latin 97.4, IN-indic 95.5.
- Residual misses: mostly empty address plus a shared or generic name. These may be irreducible.

## Bugs / cautions noted
- NFKD plus stripping combining marks deletes Indic matras and viramas. Do not apply it to Indic scripts.
- A dense sparse-matrix product (Y @ X.T) is O(|S1|) per query and blows memory on common n-grams. At test scale, use sparse top-k or an inverted index.
