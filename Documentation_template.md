# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Team Name]
**Team Members:** Priyam Ghosh, [others]
**Submission Date:** [Date]

---

## 1. Executive Summary
We treat the task as **assignment with abstention on a star forest**. Every S2/S3 record belongs to at most one S1, so each record retrieves its top S1 candidates within its country. A two-stage LightGBM scores the edges. The second stage adds *entity-centric view corroboration*: an extra house number in a record is trustworthy when a confident sibling from the same source repeats it. A decoy's perturbed number is never repeated. Finally, a per-S1 decision rule picks the match set that maximises expected F0.5, and that set may be empty, so singletons score 1.0.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the training data (full-train audit notebook: `code/business_entity_resolution/notebooks/ER_phase0_audit.ipynb`):
- Country agrees on 100% of true pairs, so blocking is partitioned by country. Country is treated as an open set, which makes France (test only) work unchanged.
- The mapping is **many-to-one**: no S2/S3 record matches two S1s. About 26% of S2/S3 records match nothing. These are *decoys*: near-copies of an S1 with a shifted house number (22 vs 11) and/or an extra modifier token (East, Greater, Overseas).
- S1 names are not unique: 36% of US and 44% of India S1 records share their normalised name with another S1. Name alone is therefore insufficient.
- The generator is hierarchical for addresses: entity → **per-source address view** → record noise. An extra address number of a true match appears in a same-source sibling 85.5% of the time, versus 1.7% for a random cluster. For decoys the figure is 0.1–0.2%, a likelihood ratio of about 450.
- India S2/S3 names appear in 9 Indic scripts, spelled token by token as phonetic transliterations of English words. The vocabulary is small (about 1.3k tokens). The Indic share rises from 10% (train) to 18% (test).
- Other noise: street abbreviations, typos, legal-suffix changes, DBA/"Mr"/"Sri" prefixes, component reordering, and empty addresses (3.5–4.5% of S2/S3).

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage GBDT (stacked, graph-context features) + expected-utility decision
**Core Innovation:** leave-one-out *view corroboration* over each (S1, source) star, computed in one synchronous (Jacobi) round. This makes it independent of record order and not self-reinforcing. The corroboration is combined with a decision rule that optimises the macro-F0.5 metric directly and includes the empty set.

---

## 3. Candidate Generation (Blocking)
- **Normalisation:** lower-case, `&` → `and`, and accents stripped on Latin letters only, so Indic matras and viramas survive. Letters, marks and digits are kept. NULL/N/A becomes empty. Indic tokens are mapped to Latin with a word dictionary learned from training pairs by Dice co-occurrence alignment.
- **Blocking keys used:** per country, each S2/S3 record queries S1 using **word uni+bigram TF-IDF** of name + address, with IDF computed on S1. N-grams with document frequency > 5,000 are dropped. This bounds the sparse-product cost, while rare tokens and bigrams such as "614 shore" or business-name words carry the ranking. The top K = 6 S1 candidates are kept per record (`sparse_dot_topn`, multithreaded).
- **Why not char n-grams:** exact char-3gram cosine was about 100× slower. A df-pruned char-3gram index collapsed to R@20 = 43%.
- **Candidate pairs generated:** 6 per S2/S3 record (about 60M on test). `output/candidate_pairs.tsv` is exactly the edge set the model scores.
- **How true matches were not lost:** measured on train, US R@6 ≈ 99.4% and India R@6 ≈ 99.3% on the sample. On the full index, US R@20 = 98.4% (see §5). The remaining misses are mostly records with an empty address and a generic name.

---

## 4. Matching Model

**Features used (stage 1, 40 features):**
- Blocking graph: TF-IDF cosine, rank, gap to the query's top-1, number of candidates, S1 in-degree and top-1 in-degree, size of the S1's same-name group, source.
- Name features: RapidFuzz ratio, token-set, token-sort, partial ratio and Jaro–Winkler; lengths; number of extra/missing tokens; learned per-token log-likelihood ratios (LLRs) of a token being extra or missing in a true pair versus a candidate non-pair. These capture legal suffixes and decoy modifiers.
- Address features: the same similarity set and token LLRs. Typed number relations: common, extra and missing numbers, smallest |shift| between an extra and a missing number, and shift ratio (decoy house-number perturbation).
- No country feature (France is unseen).

**Stage 2 (entity-centric) features:** stage-1 p plus:
- `num_corr`: max p over confident same-source siblings whose numbers contain all of the record's extra numbers.
- `contra`: a confident sibling agrees with S1 and lacks the record's extras.
- Star support (confident same-source and cross-source members, summed p).
- `p_q_other`: the record's best competing S1.

**Model type:** LightGBM binary classifier (MIT licence, far below the 8B-parameter limit). Both stages use 4-fold out-of-fold training, grouped by the record's top-1 S1. Stage 2 is trained on stage-1 OOF predictions (stacked sequential learning), so train and test feature distributions match. Stage-2 output is calibrated with isotonic regression.

**Threshold selection method:** (1) each record keeps only its best edge (many-to-one). (2) Per S1, the edges are sorted by p and the prefix *k* maximising expected F0.5 is chosen: `1.25·Σp / (k + 0.25·N)` versus `P(empty) = Π(1−p)`. On OOF this is compared against the best fixed threshold τ, and the better rule is kept.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, OOF):** 5% cluster-sample smoke run: stage 1 = 0.9913, stage 2 = 0.9925 (expected-F rule); best fixed τ = 0.9922. The sample has less inter-S1 competition than the full data, so these figures are optimistic. **Full-train OOF: [to fill after the full run].**
- **Most important stage-2 features:** p1, `num_corr`, `p_q_other`, `contra`. Corroboration is the second-largest gain after stage-1 p.
- **Common false positives (wrong merges):** [to fill after full run: decoys with a shifted number and no same-source sibling; same-name S1s with empty-address records]
- **Common false negatives (missed matches):** [to fill after full run: empty address + generic name; untranslated Indic tokens]

---

## 6. Conclusion
Exploiting the generator's structure pays off more than generic similarity learning. The structure used here is: many-to-one stars, per-source address views and number-shift decoys. Precision on decoys comes from number relations and sibling corroboration, and correct abstention comes from optimising expected F0.5 per S1, singletons included.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`
- `src/pipeline.py`: entry point. `python src/pipeline.py train`, then `python src/pipeline.py test`, writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- `src/common.py` (I/O, normalisation, macro-F0.5), `src/indic.py` (Indic dictionary), `src/blocking.py`, `src/features.py` (pair + corroboration features), `src/decide.py` (decision rule).
- `README.md` (exact run instructions), `requirements.txt` (pinned), `docs/` (design notes), `notebooks/` (full-data audit).

### B. Additional Results
Stages not yet included: a cross-encoder re-ranker on the uncertain band (mDeBERTa-v3 / XLM-R, MIT licence), transductive France token/legal-form tables, and a second corroboration round.
