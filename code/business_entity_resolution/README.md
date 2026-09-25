# Business Entity Resolution: pipeline v1

This implements `docs/2026-pipeline-design-v1.md`, stages A, B, C, E, F and H. Stage D (cross-encoder) and stage G (France transductive tables) are not built yet.

## Reproduce

```bash
pip install -r requirements.txt
export ER_DATA=/path/to/student_resource/dataset   # contains train/ and test/
export ER_WORK=work                                # cache (~15 GB on the full data)
export ER_OUT=output
python src/pipeline.py train    # prints blocking recall and OOF macro-F0.5; saves models to $ER_WORK
python src/pipeline.py test     # writes $ER_OUT/matching_results.tsv and $ER_OUT/candidate_pairs.tsv
python /path/to/student_resource/utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir $ER_DATA/test
```

Each step caches its result in `$ER_WORK`. Delete a file there to recompute that step and everything after it. For example, delete `train_Z.pkl`, `train_p2.pkl` and `model2.pkl` after changing the stage-E features.

Each module runs its own self-check with `python src/<module>.py` (common, indic, blocking, features, decide).

## Stages

| stage | file | what it does |
|---|---|---|
| A normalise | `common.py`, `indic.py` | Lower-case, `&` → `and`. Strip accents on Latin letters only, so Indic matras and viramas survive. Keep letters, marks and digits. Map NULL/N/A to empty. Map Indic-script tokens to Latin through a word dictionary learned from train pairs (Dice co-occurrence). |
| B blocking | `blocking.py` | Per country (an open set, so France is included), each S2/S3 record queries S1 through word uni+bigram TF-IDF and keeps the top K=6. N-grams with df > 5000 are dropped, which bounds the cost. |
| C pair features | `features.py` | Blocking score, rank, gap and degrees. RapidFuzz name/address similarities. Typed number relations (common, extra, missing, smallest shift, ratio). Token LLR tables for extra/missing tokens, learned on train. No country feature. |
| stage-1 GBDT | `pipeline.py` | LightGBM, 4-fold OOF grouped by the query's top-1 S1. |
| E corroboration | `features.py` | One synchronous round of leave-one-out view corroboration from confident same-(S1, source) siblings: `num_corr`, `contra`, star support, `p_q_other`. The self-test checks that row order does not change the output. |
| stage-2 GBDT | `pipeline.py` | Stage-1 features + p1 + stage-E features, OOF again, then isotonic calibration. |
| F decision | `decide.py` | Each record goes to its single best S1 (many-to-one). Per S1, choose the prefix that maximises expected F0.5, with an empty set allowed (singletons). The rule is compared on OOF against the best fixed threshold τ, and the better one is kept. |
| H output | `pipeline.py` | Writes one row per test S1. The candidate file is exactly the edges the model scored, so matches ⊆ candidates. |
