# Amazon ML Challenge 2026: Business Entity Resolution

The repository follows the final-submission layout from the problem statement:

```
├── output/
│   ├── matching_results.tsv        # final matches (leaderboard file), produced by the pipeline
│   └── candidate_pairs.tsv         # blocking candidate set fed to the model
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all source code (entry point: src/pipeline.py)
│       ├── README.md               # how to reproduce end-to-end
│       ├── requirements.txt        # pinned dependencies
│       ├── docs/                   # design notes (pipeline v1, entity-centric views)
│       └── notebooks/              # phase-0 audit on the full training data
└── Documentation_template.md       # methodology write-up
```

The dataset is **not** in the repo: it is too big for GitHub and belongs to the competition. Put `student_resource/` under `data/` locally, which `.gitignore` excludes. The two `output/*.tsv` files are stored with **Git LFS** because they exceed GitHub's 100 MB file limit.

## Reproduce

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
export ER_DATA=../../data/student_resource/dataset ER_WORK=../../work ER_OUT=../../output
python src/pipeline.py train
python src/pipeline.py test
python ../../data/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../data/student_resource/dataset/test
```

## Build the submission zip

From a checkout with the real outputs (`git lfs pull`), zip exactly the three top-level items:

```bash
zip -r <team_name>_submission.zip output code Documentation_template.md -x "*/__pycache__/*"
```
