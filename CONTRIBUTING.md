# Contributing

SAGA is a research codebase. Contributions should preserve deterministic behavior, explicit provenance, and the distinction between generated-data quality and downstream task benefit.

Before opening a pull request:

1. Create or update a focused test under `tests/`.
2. Run `conda run -n sd3 python -m pytest`.
3. Run `conda run -n sd3 python scripts/check_publication.py`.
4. Do not add private datasets, model checkpoints, API keys, run directories, or third-party source trees to the repository.

For new skills, document inputs, outputs, controllable parameters, cost, failure modes, and the observer evidence required before claiming downstream benefit.
