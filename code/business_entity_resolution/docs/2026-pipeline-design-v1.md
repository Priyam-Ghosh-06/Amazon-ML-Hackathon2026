# Pipeline design v1: entity-centric graph ER (2026-09-25)

Status: DESIGN. Nothing built. The user confirmed that no pipeline code exists. The v1–v6 numbers in `2026-pipeline-loop-log.md` cannot be reproduced from anything on disk, so treat them as hypotheses only. Every fact the design relies on is re-measured on the FULL train by `ER_phase0_audit.ipynb` (cells F1–F7).

## 0. What a sample can and cannot tell us
- **Safe from an S1-cluster sample** (whole clusters kept): per-cluster statistics such as matches per S1, and sibling/view sharing (F7). The sampling unit is the cluster, so the estimate is unbiased (n = 382k positives).
- **Unsafe**: anything involving competition between S1s, such as decoy rate, name sharing, retrieval ranks and precision. On the 5% sample, F6 name sharing comes out at 14–20%, versus the 36–44% believed for the full data, and the decoy share comes out at 34%. The notebook prints both, so the bias is visible.

## 1. Generative picture (the object we invert)
- Latent entity θ_e.
- A per-source address view V_{e,k}, k ∈ {S2, S3}.
- Positives: r ~ N(·|V_{e,k}).
- Decoys: r ~ D(·|θ_e), a perturbed copy that belongs to no S1.
- Label z_r ∈ S1 ∪ {∅}.
- Output: for each S1, the set maximising expected macro-F0.5.
- Graph: a star forest (S1 hubs, S2/S3 leaves). Clustering is given; the task is **assignment with abstention**, not cluster discovery.

## 2. Stages (each with a gate; code is delivered phase by phase after the audit output is reviewed)

| # | Stage | Method | Gate (measured on the held-out S1 half, full competition) |
|---|---|---|---|
| A | Normalise | Keep Unicode letters, marks and digits. Mojibake fix. Legal-form extraction. Number extraction. State tables as *features*, never as filters. Indic token dictionary learned (IBM-1 EM) from training-fold pairs only. | T-A: row counts preserved. Indic matras survive. |
| B | Candidate graph | Buckets (country, state-if-present, else country). Channels: char-3gram TF-IDF of name+address (top-k); squashed-name TF-IDF; **semantic channel**: multilingual-e5-small (MIT) + FAISS, as a union. | Recall of true pairs and candidates/query per channel. The semantic channel stays only if its *unique* recall gain is worth its extra candidates. |
| C | Pairwise comparison vector (Fellegi–Sunter) + GBDT stage-1 | String similarities; typed number relations; learned operator tables (number-shift LLR, inserted-token LLR, legal-form pair LLR), always out-of-fold; blocking scores/ranks. 5-fold OOF, grouped by S1. | OOF AUC / logloss per country. |
| D | Cross-encoder (the "semantic" matcher) | mDeBERTa-v3-base (MIT) or XLM-R-base (MIT) on "name ‖ address" pairs, fine-tuned on blocking candidates (hard negatives = decoys + same-name other S1). Trained on S1 split A, applied to split B + test. Run only on the uncertain band of stage-1 p. Its logit is a stage-2 feature. | Stage-2 F with vs without the CE logit, on both halves. |
| E | Entity-centric graph resolution (Senzing mechanics, batch form) | Section 3. Unrolled T=2, synchronous updates, each round a GBDT trained on the previous round's OOF outputs. | Gain over stage-1+CE on both halves; the shuffled-input output hash is identical. |
| F | Decision | Isotonic calibration (OOF), then a per-query softmax with a null option, then per-S1 expected-F0.5 optimal set (sort by p, evaluate prefixes, Poisson-binomial). | Macro-F0.5 against the best fixed τ. |
| G | France (unseen country) | Operator tables (tokens, legal forms) re-learned transductively on test France, using label functions (the number shift ⇒ decoy rule; a confident no-shift top-1 ⇒ positive). Conservative decision. | Leave-one-country-out proxy (US→IN, IN→US). |
| H | Output | matching_results.tsv ⊆ candidate_pairs.tsv; validator PASS. | validate_submission.py |

## 3. Stage E: Senzing behaviours → batch algorithm
State: assignment a(r) ∈ S1 ∪ {∅}. Profile Φ_s = S1_s ∪ members, kept per source for addresses.

For each round t = 1..T:
1. Features of the edge (r, s) against Φ_s \ {r} (leave-one-out):
   - **union of features** (Senzing record 10): max over members m of name_sim(r, m) and addr_sim(r, m), separately
   - **view corroboration** (discoverable add, record 8): r's numbers/tokens not in S1 but present in a same-source member, weighted by that member's p
   - **contradiction** (un-resolve, record 9): confident same-source members agree with S1 on the component where r deviates
   - **possible-match context** (E5 on slide 10): the competing S1's p for the same query; the members and decoy pressure of the S1
2. p^t = f_t(x, features^t). The model f_t is trained on round-(t−1) OOF outputs (stacked sequential learning), so the train and test feature distributions match.
3. Synchronous (Jacobi) update: every edge in a round uses p^{t−1}. The output is a function of the record multiset, which gives **sequence neutrality**.
4. Laundering guard: members need p ≥ τ_m. Measure how often decoys receive corroboration (target ≤ 1%).

## 4. Rejected / reframed
- **t-SNE / parametric t-SNE / autoencoder clustering.** Unsupervised, so blind to labels. Perplexity sets a neighbourhood scale σ, and decoys sit *inside* σ of their parent S1, so the KL objective pulls them together. Reconstruction losses are dominated by the high-variance "which business" directions, not the low-variance decoy direction. And the clusters are already given by the star structure.
- **FastRP.** Its 1-hop term is a random-projection sketch of the TF-IDF cosine; the sketch noise (~0.09 at d=256) exceeds the decoy margin (~0.03).
- **WCC / transitive closure.** Replaced by the many-to-one star structure. Kept only as a diagnostic (contested regions).
- **Bi-encoder as the final matcher.** An inner product of pooled encodings cannot represent "number differs by +3", so it is used for recall only. The final semantic matcher is a cross-encoder.
