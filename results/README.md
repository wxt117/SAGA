# Paper Results

The paper reports seven controlled experiments. The original run outputs are intentionally excluded from the public repository because they contain private-data paths, generated artifacts, and large figures. The experiment scripts under `experiments/` regenerate the controlled reports when their documented inputs are available.

Claims involving private SAR datasets, proprietary model weights, or external simulators cannot be reproduced from this repository alone. See [reproducibility notes](../docs/reproducibility.md).

Selected manuscript-level aggregates:

| Experiment | Full SAGA result |
|---|---:|
| Schema decision accuracy (35 cases) | 97.1% |
| Top-1 skill accuracy (80 cases) | 92.5% |
| Recipe execution success (9 cases) | 100.0% |
| Full-observer invalid-case recall (11 batches) | 90.0% |
| Downstream matched gain (8 tasks x 12 seeds) | +7.4 pp |
| Downstream invalid-sample rate | 1.0% |

These are controlled benchmark results, not guarantees for new datasets. The downstream score aggregates task-specific accuracy, macro-F1, and mAP only for compact reporting.
