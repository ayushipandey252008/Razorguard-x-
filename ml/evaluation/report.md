# SYNTHETIC_DATASET — evaluation

**Track: SYNTHETIC_DATASET.** Do not mix with REAL_DATASET metrics.

- Version: `xgb-iforest-v1-calibrated`
- Booster: `xgboost`
- Split: 70/15/15 stratified, seed=42 (not a production time split)
- Test n=1200, positive rate=0.150
- Probabilities: isotonic regression on validation booster scores (prefit; not nested CV)

## Calibrated test metrics (0.5 operating point)

| Metric | Calibrated | Uncalibrated |
| --- | --- | --- |
| Precision | 0.966 | 0.959 |
| Recall | 0.783 | 0.783 |
| F1 | 0.865 | 0.862 |
| ROC-AUC | 0.889 | 0.889 |
| PR-AUC | 0.818 | 0.822 |
| FPR | 0.005 | 0.006 |
| Brier | 0.036 | 0.049 |

Confusion (calibrated @0.5): TN=1015 FP=5 FN=39 TP=141

Accuracy is omitted as a primary metric.

The 0.5 cutoff is **not** the product decision threshold. Product APPROVE/REVIEW/BLOCK uses `THRESHOLD_*` on the 0–100 risk score.
