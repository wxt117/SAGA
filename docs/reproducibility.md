# Reproducibility

SAGA separates the public decision framework from optional research backends.

## Reproducible from this repository

- Dataset profiling, format bridging, deterministic validation, recipe serialization, traditional augmentation, target-background composition, quality checks, SAR-artifact checks, duplicate checks, leakage checks, and report validation.
- The controlled benchmark logic in `experiments/exp2_intent_skill_planning.py`, `experiments/exp3_recipe_execution_reliability.py`, `experiments/exp4_observer_evidence_repair.py`, `experiments/exp5_large_downstream_benchmark.py`, and `experiments/exp7_ablation_study.py`.

## Requires local or private assets

- `exp1_schema_grounding.py` and the real-data branch of `exp5_downstream_augmentation_benefit.py` use dataset layouts that were not public in the paper's data-availability statement.
- LoRA, GeoDiff-SAR, GAN, style-transfer, Gaussian-splatting, RaySAR, and background-generation skills are wrappers. Their source projects, checkpoints, CUDA stacks, and licenses are not vendored here.
- `exp6_case_study_qualitative.py` expects example images and generated artifacts. It is a traceability illustration, not a substitute for a blinded human study.

All commands in the documentation use the server's `sd3` Conda environment. Expensive skills default to dry-run so that planning and provenance can be inspected before model execution.
