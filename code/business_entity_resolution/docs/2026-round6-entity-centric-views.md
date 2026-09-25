
## Round 6: graph / entity-centric ER. The generator has a per-source ADDRESS view (measured 2026-09-25)

Sources reviewed:
- Senzing-style entity-centric learning: disclosed and discoverable relationships, sequence neutrality
- Neo4j GDS entity resolution: WCC → node similarity → FastRP+kNN
- Graphlet "semantic ER": SBERT blocking plus LLM matching and merging
- Zingg: learned blocking, active learning, classifier, cluster min/max score

The two videos could not be fetched (rate-limited). They were assessed from their known frameworks.

**Theorem (when entity-centric learning can help).** Suppose records are conditionally independent given the latent entity θ_e, and S1 = θ_e exactly. Then P(z_r = e | r, s_e, siblings) = P(z_r = e | r, s_e), so siblings add zero information. Entity-centric learning helps only if (a) S1 is a noisy view of θ_e, or (b) there is shared latent structure below the entity level. Measured on the `_eda_sample` (110k S1 and their complete clusters):

| extra element of a positive (absent from its S1) | found in a same-source sibling | cross-source sibling | random cluster |
|---|---|---|---|
| address number | **0.855** | 0.038 | 0.017 |
| address token (non-numeric) | **0.762** | 0.204 | 0.045 |
| name token (Latin, non-legal) | 0.030 | 0.032 | 0.021 |

→ The generator is hierarchical for addresses: entity → **per-source address view** → record noise. Name noise is independent per record. So the empty-address floor stands: siblings carry no name information beyond S1. (Indic name tokens only look shared because transliteration is deterministic.)

**Decoy vs positive, by number relation to S1** (the "explained" rate is conditional on a same-source sibling existing, which holds for 80% of positives):

| class | n | extra numbers ⊆ same-source siblings' numbers |
|---|---|---|
| positive, shift ∈ D (currently looks like a decoy) | 2,590 | **0.852** |
| positive, other extra number | 61,039 | 0.836 |
| decoy, shift ∈ D (parent sim > 0.5) | 3,360 | **0.001** |
| decoy, other extra number | 6,788 | 0.002 |

LR(explained) ≈ 450 toward a match. LR(unexplained, sibling exists) ≈ 0.15. Decoy-to-decoy corroboration (same parent and source) is 65/3411 = 1.9%. That is small but nonzero, so the laundering risk is real and must be guarded.

**Plan (S4.5, view-corroboration stage, to scrutinise before coding).** For each candidate edge (r, s), let X_r be r's numbers (and address tokens) not in S1. Compute leave-one-out corroboration from same-source candidates r' ≠ r of the same S1, weighted by their stage-2 p (OOF in train):
- num_corr = max p(r'→s)·1[X_r ⊆ N(r')]
- soft fraction explained
- contradiction = a confident same-source sibling agrees with S1 on the slot where r deviates

Updates are one-step and synchronous (Jacobi), which keeps them sequence-neutral and non-self-reinforcing. Mutual corroboration is allowed only in an ablated second iteration.

Tests:
- T6.1: r is excluded from its own profile (assert).
- T6.2: PSI of the corroboration features, OOF-train vs test.
- T6.3: share of decoys with num_corr > 0 is ≤ 1%.
- T6.4: France label-free replication of the same-source vs random gap.
- T6.5: shuffling row order gives byte-identical output.
- Adopt only if E2 gain ≥ +0.03 F and E1 agrees.

Verdicts:
- **FastRP:** its 1-hop term is a Johnson–Lindenstrauss sketch of our exact TF-IDF cosine, so it only adds noise.
- **WCC/transitivity:** replaced by the many-to-one star structure.
- **SBERT/semantic bi-encoders:** decoys are semantically identical by construction, and pooling discards *which* edit happened.
- **Zingg's cluster min-score:** this is the corroboration idea itself.
- **Set-transformer over the (S1, source) star:** the learned version of the corroboration features. Permutation invariance gives sequence neutrality. It is worth an ablation only after the hand features give the rank-0 result.
