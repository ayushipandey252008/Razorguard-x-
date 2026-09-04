# Cost-based threshold experiment (SYNTHETIC_DATASET, validation)

Relative cost units, not currency and not industry-standard.
- FP (legit BLOCK): 10.0
- FN (fraud APPROVE): 100.0
- REVIEW (either class): 2.0

Lowest-cost pair on **validation calibrated probabilities**: review=0.05 (≈ risk 5), block=0.814 (≈ risk 81), expected cost=2392.0.

Apply via `THRESHOLD_REVIEW` / `THRESHOLD_BLOCK` (0–100 risk scale). Do not treat this pair as a certified operating point.

| review_p | block_p | precision | recall | FP_block | FN_approve | cost |
| --- | --- | --- | --- | --- | --- | --- |
| 0.050 | 0.814 | 0.878 | 0.873 | 4 | 23 | 2392.0 |
| 0.050 | 0.200 | 0.878 | 0.873 | 6 | 23 | 2394.0 |
| 0.050 | 0.268 | 0.878 | 0.873 | 6 | 23 | 2394.0 |
| 0.050 | 0.336 | 0.878 | 0.873 | 6 | 23 | 2394.0 |
| 0.050 | 0.405 | 0.878 | 0.873 | 6 | 23 | 2394.0 |
| 0.050 | 0.473 | 0.878 | 0.873 | 6 | 23 | 2394.0 |
| 0.050 | 0.541 | 0.878 | 0.873 | 6 | 23 | 2394.0 |
| 0.050 | 0.609 | 0.878 | 0.873 | 6 | 23 | 2394.0 |
| 0.050 | 0.677 | 0.878 | 0.873 | 6 | 23 | 2394.0 |
| 0.050 | 0.745 | 0.878 | 0.873 | 6 | 23 | 2394.0 |
| 0.050 | 0.882 | 0.878 | 0.873 | 2 | 23 | 2404.0 |
| 0.050 | 0.950 | 0.878 | 0.873 | 2 | 23 | 2404.0 |
