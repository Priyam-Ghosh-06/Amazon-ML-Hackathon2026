# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** Cursed with dimensionality
**Team Members:** Priyam Ghosh, Krishna Simha,Akshat Rai
**Submission Date:** 27-09-2026

---

## 1. Executive Summary
We treat the task as **assignment with abstention on a star forest**. Every S2/S3 record belongs to at most one S1, and about 26% belong to none. Those are *decoys*: near-copies of an S1 whose only differences are small edits, such as a shifted house number, an added modifier word or a swapped legal form.

The pipeline has five parts:
- **Candidate search:** runs over the full S1 index inside (country, state) buckets, with two channels: char 3-gram TF-IDF and a fine-tuned multilingual bi-encoder.
- **Edge scoring:** a two-stage LightGBM scores each candidate edge. Its features are built to expose those small edits.
- **Cross-encoder:** a multilingual model re-reads the uncertain edges with one token per digit.
- **Sibling features:** evidence from the other records matched to the same S1 in the same source re-prices each edge.
- **Decision:** a per-S1 rule picks the match set that maximises the exact expected F0.5. The set may be empty, so an S1 with no true match scores 1.0.

**Versions.** v2 is the full pipeline (search, features, models, decision). v3 adds honest validation for both countries, a per-country prior-shift correction and label-free tables for France. v4 retrains nothing: it only changes which edges the decision layer keeps, and each change is kept only if it improves held-out EVAL. If no v4 change passes its gate, the v4 submission is identical to v3 (see 4c).

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the training data:
- **Country and state.** Country agrees on 100% of true pairs. State agrees on 100% of US true pairs. In India it agrees once Telangana and Andhra Pradesh are merged: source records keep the pre-2014 state for Hyderabad. **City does not agree:** 14% of US and 7% of India true pairs name a different place (postal city vs township/county vs locality). So state is a hard constraint and city only a soft signal.
- **Many-to-one.** No S2/S3 record matches two S1s. **Decoys** are perturbed copies of an S1. Their house-number shift is signed and comes from a fixed set, D = {1,2,3,4,5,7,9,11,13,21}: a +21 shift is ~1300× more likely in a decoy than in a true match. Decoys copy the parent's city, so the address hierarchy does not separate them.
- **Name sharing.** 36% of US and 44% of India S1 records share their normalised name with another S1.
- **Per-source address view.** The generator gives each source its own view of the address. An extra number in a true match appears in a same-source sibling 85.5% of the time, versus 3.8% for a cross-source sibling and 1.7% for a random cluster.
- **Indic scripts.** India S2/S3 names and addresses use 9 Indic scripts, spelled word by word as transliterations of English.
- **Other noise:** dotted acronyms (`L.L.C.`), fused web-domain names (`wmelitedeli.com`), zero-padded numbers, OCR digits inside words, NULL address pieces, component reordering (only 7% keep S1's order), and empty addresses.

### 2.2 Solution Strategy
**Approach Type:** two-channel blocking + stacked GBDT with a cross-encoder on the uncertain band + graph (sibling) features + expected-utility decision.

**Core ideas:**
1. **Train in a full-competition universe.** Candidate search always uses the full S1 index. Models are fitted on whole geographic regions, so decoys meet their parents exactly as on test.
2. **Keep decoy edits visible.** Only pure noise is rewritten. Numbers, legal forms and modifiers are priced by learned features.
3. **Use the per-source view** through one synchronous, leave-one-out round of sibling corroboration.

---

## 3. Candidate Generation (Blocking)
- **Normalisation:**
  - lower-case; strip accents on Latin letters only (Indic matras survive);
  - keep letters, marks and digits;
  - collapse dotted acronyms, drop NULL address components, fold zero padding, strip web domains, fix OCR digits inside words;
  - split addresses into comma components; keep number suffixes (`593a`, `42rd`).

  Tables learned from training pairs:
  - an Indic → Latin word dictionary (Dice alignment);
  - a state vocabulary per country, learned from S1 only so it also works on France;
  - query-side state spellings (`MH`, `Pennsylvania`, `महाराष्ट्र`);
  - merges for states that true pairs swap.
- **Blocking keys:**
  - (country, state) buckets. S1 records with no state are added to every bucket of their country, and queries with no state search their whole country.
  - A country is bucketed only if confident matches agree on state ≥ 99%. This check needs no labels, so it also decides France.
  - **Channel 1:** char 3-gram TF-IDF, 0.5·cos(name without spaces) + 0.5·cos(address). N-grams in more than 5% of a country's S1 are pruned.
  - **Channel 2:** multilingual-e5-small (MIT) fine-tuned with InfoNCE: batches of up to 4096 grouped by region, one mined hard negative per query, and masking of in-batch false negatives. Search is exact inner product on the GPU.
- **Candidate pairs generated:** the union of the top 6 from each channel, about 8–10 per S2/S3 record. `output/candidate_pairs.tsv` is exactly the edge set the model scores.
- **Recall:** [fill from `eval/report.md` and `work/run.log` "recall" lines].

---

## 4. Matching Model
**Features (stage 1, 65):**
- **Retrieval:** both channels' scores and ranks, gap to the query's best, S1 in-degree and top-1 in-degree, the S1's mean score (hubness), number of S1 sharing the name, number of S1 at the same street and number.
- **Name:** RapidFuzz ratio, token-set, token-sort, partial and Jaro–Winkler, on the full name and on the name with legal forms removed.
  - Token log-likelihood ratios for extra/missing tokens, learned **against decoys vs their near-duplicate S1** (not against arbitrary wrong candidates).
  - Edits at the start or end of a word; legal-form family conflict, extra and missing; exact match.
- **Address:** the same similarity set and token log-likelihood ratios, plus per-level comparison:
  - state relation;
  - place conflict vs place missing on one side (with learned place aliases);
  - street-token Jaccard;
  - landmark words.
- **Numbers:** common, extra and missing counts; signed shift, |shift| and shift ratio; whether the shift is in D (either sign); letter-suffix mismatch; wrong ordinal suffix (`42rd`).

**Cross-encoder:** multilingual-e5-base (MIT) on Ditto-style pairs `[COL] name [VAL] … [COL] addr [VAL] …`, with one token per digit.
- It scores only each query's top-2 edges with 0.02 < p₁ < 0.98, so the cheap model decides the easy edges and the expensive model checks the uncertain ones.
- Two fold models give out-of-fold scores on train; validation and test use their mean.

**Stage-2 (graph) features:** p₁, the cross-encoder logit, and one Jacobi round over each (S1, source) star. That round computes:
- confident same-source and cross-source support;
- number and token corroboration (a confident same-source sibling contains all of the record's extra numbers or address tokens), the share of extras explained, and contradiction;
- the best address similarity to a sibling;
- the competing S1's p.

**Model type:** LightGBM (MIT), 4 folds grouped by the query's top-1 S1, early stopping on a validation region of whole buckets.
- Validation and test predictions use the mean of the fold models.
- Stage-2 output is calibrated with isotonic regression.
- Total parameters are far below the 8B limit (e5-small 118M + e5-base 278M + trees).

**Checkpointing:** the neural models are validated every 100 steps (bi-encoder) or 500 steps (cross-encoder). `best.pt` is saved only after 30% of planned steps and only on a ≥ 0.5% lower validation loss, with an atomic overwrite. Each training loop has a wall-clock cap.

**Threshold selection method:**
1. Each record keeps its best edge (many-to-one).
2. Per S1, choose the prefix of edges (sorted by p) that maximises the **exact** expected F0.5, using Poisson-binomial distributions of true matches inside and outside the selection. The empty set is allowed.
3. This rule is compared with fixed thresholds on the validation region, and the better one is kept.

---

## 4b. v3: what changed after the v2 submission (leaderboard 0.98438)
**v3 reuses v2's search, features, models and caches, and adds three things.** Each was motivated by a measurement.

1. **Honest validation for both countries.**
   - *Problem:* v2's validation region turned out to be 100% US, so India was never measured.
   - *Fix:* v3 searches the 46% of train S1 buckets v2 never used, with the frozen v2 models, and splits them by whole buckets into CAL (calibration, early stopping, rule choice) and EVAL (reporting and gating only).
2. **Per-country prior-shift correction.**
   - *Measurement:* test has 5.75 S2/S3 per S1 vs 4.68 in train. The decoy signature (same street, near-identical name, house number shifted by a decoy step) relative to the positive signature is 1.6–1.8× more frequent in test (US 0.28 → 0.52, India 0.16 → 0.26). So test contains ~1.7× more decoys per true match, and a model calibrated on train over-accepts them.
   - *Correction:*
     - each query's best edge is calibrated per country on CAL;
     - each country's test match rate is estimated from its own predictions (Saerens–Latinne–Decaestecker EM under label shift);
     - the odds are rescaled to that rate;
     - the rule (exact expected-F or a threshold, prior correction on or off) is chosen per country on CAL with unmatched records replicated to the test rate.
3. **France without labels.**
   - Word tables learned from signature pairs match the truth-based tables at Spearman 0.958 (US) / 0.904 (India). On France they find the decoy words v2 could not see: holding, snc, international, distribution, participations, groupe, france.
   - Country names are folded to one token, so India's learned "inserted country name = decoy" transfers.
   - French legal forms (SNC, SCI, SELARL, SCA, SCOP, GIE) are added.
   - Département ↔ région aliases are learned from positive signatures.
   - Street types are canonicalised (ave/avenue, r/rue, bd/boulevard…).

GBDT1/GBDT2 are refitted with these features on hard-negative-weighted rows (all positives, every negative v2 found non-trivial, and weighted easy negatives), early-stopped on CAL. The cross-encoder is reused and only scores new uncertain edges.

**Gates.**
- *Variants:* A = v2 as submitted; B = v2 models + v3 decisions; C = v3 models + v3 decisions.
- *US and India:* per country, the variant with the best EVAL F0.5 at the test decoy density wins, provided plain EVAL does not drop.
- *France:* uses C only if C wins in both labelled countries **and** India passes the "India-as-France" check (India scored with US-only tables plus its own label-free tables must not lose to US-only tables). Otherwise France gets B, or A.

The run is anytime: the v2 file stays in `output/` until a better variant is assembled, and a hard wall-clock guard always leaves time to assemble and validate.

## 4c. v4: decision-layer fixes on top of v3 (models frozen)

v4 retrains no model. It reads the v3 probabilities and calibration (`work/v3/`) and changes only which edges are selected (`src/postfix.py`). Each change is chosen per country on CAL at the test decoy density and kept only if held-out EVAL improves; France follows only if both US and India kept it.

1. **Prior shift split by address presence.**
   - *Measurement:* in train, queries with an empty address (~3% of S2/S3) are 97.8% true matches, against 71.6% of addressed ones. The decoys, of which test has ~1.7× more, carry (shifted) addresses.
   - *Fix:* v3's single EM prior shift per country deflated empty-address edges together with the decoy-heavy addressed ones. v4 runs the EM separately per stratum, each against its own CAL match rate.
   - *Why not guess:* when a query has no address and several S1 records share its exact name (~20% of empty-address queries), no feature separates them. Every namesake has its own members, so leaving the query unassigned beats a random choice: at 2 namesakes, a random assignment costs ~0.05 F in expectation.
2. **Lone-claimant threshold.** A singleton's false merge nearly always comes from a single decoy, the only non-negligible edge into that S1. For such edges the expected-F rule reduces to "p > 0.5"; v4 learns a stricter threshold per country.
3. **Singleton filter.**
   - *Model:* an S1-level LightGBM classifier ("this S1 has no true match"), trained on CAL S1s that currently receive ≥ 1 match.
   - *Features:* the S1's claimants: how many, from which sources (a true entity has ~3.5 members spread over S2 and S3), their probabilities, and how strongly each is pulled to another S1.
   - *Decision:* every match of an S1 whose singleton probability exceeds a threshold (chosen out-of-fold on CAL) is removed, because an empty S1 scores 1.0 if it is a singleton.

Considered and rejected for v4:
- *deepparse for French addresses:* agreed with the comma structure on the city for only 43% of S1 addresses, because it discards the commas.
- *An LLM parse of all French records at inference:* ~10–25 GPU-hours for 1.7M records.
- *Graph-based training augmentation / label-noise removal:* needs retraining, and the labels are synthetic and clean.

---

## 5. Results & Error Analysis
- **Leaderboard:** v2 0.98438 → v3 (C: v3 models + v3 decisions) 0.9858 → v4 [fill after submission; if v4 was identical to v3, say so].
- **v4 held-out results:** [fill from the `[C+]` lines in `work/run.log` and `work/v3/choice.json`: which of the three changes each country kept, and EVAL F0.5 at test density before and after].
- **v3 held-out results** (EVAL F0.5 / at test decoy density):

  | variant | US | India |
  |---|---|---|
  | A: v2 as submitted | 0.98876 / 0.98787 | 0.98970 / 0.98872 |
  | B: v2 models + v3 decisions | 0.98870 / 0.98816 | 0.98982 / 0.98910 |
  | C: v3 models + v3 decisions | 0.98948 / 0.98913 | 0.99010 / 0.98948 |

  Test S1 shares are US 38.3%, India 46.8% and France 15.0%. With the at-test-density scores for US and India, the two leaderboard scores imply France at ≈ 0.962 (v2) and ≈ 0.966 (v3), so France is where most of the remaining error is.
- **F_0.5 Score (macro, validation region):** [fill from `eval/report.md`]
- **By slice:** singletons [ ], matched S1 [ ], US [ ], India [ ]
- **Decoy acceptance rate:** [ ] (v1 trained on a 5% S1 sample accepted ~6% of decoys at full scale; this was the main motivation for full-index training)
- **Most important features:** [fill from the `gbdt1/gbdt2 top features` lines in `work/run.log`]
- **Common false positives:** [fill]
- **Common false negatives:** [fill]

---

## 6. Conclusion
The largest gains come from respecting how the data was generated rather than from generic similarity learning:
- train where decoys meet their parents;
- keep the small edits visible to the model;
- use only the address hierarchy levels that are true constraints (state), and price the rest (city);
- let same-source siblings vouch for a record's extra details;
- optimise the macro-F0.5 metric directly per S1.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/run.py`: entry point (`check`, `smoke`, `train`, `test`, `diagnose`; v3: `v3check`, `smoke3`, `v3`, `assemble`; v4: `v4check`, `smoke4`, `postfix`); `run_v3.sh` / `run_v4.sh` run v3 / v4 under the deadline;
- `src/v3.py` (v3 orchestration, decision layer, gates, assembly), `src/lf.py` (label-free signature tables), `src/postfix.py` (v4 decision-layer fixes);
- `src/text.py` (normalisation, learned tables), `src/blocking.py`, `src/bienc.py`, `src/features.py`, `src/gbdt.py`, `src/crossenc.py`, `src/decide.py`, `src/evaluate.py`, `src/nnutil.py`, `src/config.py`;
- `README.md` (requirements, exact run instructions, troubleshooting), `requirements.txt` (pinned), `run_all.sh`;
- `../../README_v4.md` (how to run v4 on top of a finished v3 folder), `docs/` (design notes and the round-6 measurement of the per-source address view), `notebooks/ER_phase0_audit.ipynb` (phase-0 data audit).

### B. Additional Results
Considered and not used:
- a hard city veto: it would remove 7–14% of true matches;
- geocoding: prohibited by the rules;
- hierarchy graph neural networks: they need parsed, ordered address fields;
- focal loss: it distorts the calibration the decision rule needs;
- Multi-head Latent Attention, speculative decoding and KV caching: these target text-generating models, and our models are encoders on short pairs.
