# Machine learning

## Tracks

| Track | What | Artifacts |
| --- | --- | --- |
| SYNTHETIC_DATASET | Product model used by the API | `ml/models/*`, `ml/evaluation/report.md`, `audit.md` |
| REAL_DATASET | Offline ULB/Kaggle credit-card PCA adapter (`ulb-xgb-v1`) | `ml/models/ulb/`, `ml/evaluation/ulb_report.md` |
| IEEE_CIS_OFFLINE | Offline IEEE-CIS candidates (`ieee-xgb-*-v1`) | `ml/models/ieee/`, `docs/ieee-cis-evaluation.md` |

Do not mix the metric tables. The public CSV has no device, IP, merchant, or graph fields, so it cannot drive the product pipeline.

## Features (product)

Defined once in `backend/app/ml/features.py` (`FEATURE_COLUMNS`) so training and inference cannot drift.

Inputs are synthetic payment attributes. **No PANs, CVVs, or authentication secrets.**

## Supervised model (product)

- Algorithm: XGBoost `binary:logistic` (HistGBM fallback)
- Class imbalance: `scale_pos_weight = neg/pos` on the training split
- Split: 70 / 15 / 15, stratified random, seed `42`
- Calibration: isotonic `CalibratedClassifierCV(cv='prefit')` on validation
- Primary metrics: precision, recall, F1, ROC-AUC, PR-AUC, FPR, FNR, Brier, confusion matrix  
  Accuracy is not used as the optimization target.

`ml_probability` is calibrated P(fraud) when the calibrator artifact exists. `ml_score` is that probability × 100 (a risk score). Uncalibrated booster output is stored as `ml_probability_raw` and must not be shown as a percent.

## Anomaly

Isolation Forest is **global** (fit on non-fraud training rows). `anomaly/behavior.py` adds **personalized** explainable checks: amount vs user typical, hour vs typical hour, device familiarity, location familiarity, velocity vs history size. The blended `behavior_score` is 35% forest + 65% personalized flags.

## SHAP

`TreeExplainer` on the uncalibrated booster. Failures are logged and omitted rather than blocking a score.

## Thresholds

Product APPROVE/REVIEW/BLOCK uses `THRESHOLD_REVIEW` / `THRESHOLD_BLOCK` on the combined 0–100 risk. A cost sweep on synthetic validation probabilities is in `ml/evaluation/threshold_calibration.md`. That sweep is an experiment, not an industry operating point.

## Public data

```bash
# REAL_DATASET — ULB offline adapter (does not overwrite the product model)
PYTHONPATH=. python ml/data/scripts/download_ulb.py
PYTHONPATH=. python ml/training/train_ulb.py
```

ULB artifacts: `ml/models/ulb/` (`ulb-xgb-v1`)  
ULB report: `ml/evaluation/ulb_report.md`  
Leakage audit: `ml/evaluation/data_leakage_report.md`

The public CSV has no device/IP/merchant graph fields. The adapter trains a **separate** supervised model and does not replace the product pipeline.

## IEEE-CIS (offline candidates)

```bash
PYTHONPATH=backend:. python ml/training/train_ieee.py
```

Place `train_transaction.csv` and `train_identity.csv` in `IEEE_DATA_DIR` (default `ml/data/ieee/`). The prototype does not download this dataset.

Artifacts: `ml/models/ieee/` (`ieee-xgb-baseline-v1`, `ieee-xgb-combined-v1`, `ieee-xgb-graph-v1`) as **CANDIDATE** only. Live scoring stays on `xgb-iforest-v1-calibrated`.

The IEEE-CIS experiment is an offline public-dataset evaluation. It does not represent production payment-fraud performance.

See `docs/ieee-cis-evaluation.md`.

## Ethics

SYNTHETIC_DATASET metrics recover the generator. They are **not** production fraud-detection performance.
