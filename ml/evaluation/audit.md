# SYNTHETIC_DATASET — ML audit

**Track: SYNTHETIC_DATASET.** Product-model metrics live in `synthetic_metrics.json` / `report.md`.
**Track: REAL_DATASET** is a separate adapter (`public_metrics.json`). Do not mix the two tables.

This document describes how the product model is generated and evaluated. It is not a claim of production fraud-detection performance.

## Dataset generation

`app/services/synthetic.py` `generate_labeled_dataset(n, seed=42)`:

- ~88% `_normal` rows (amount near a per-user typical, known device/location, low velocity).
- ~12% injected patterns: card testing, account takeover, stolen account, velocity attack.
- Then **3% independent label flips** so a pattern is not a deterministic label.

Identifiers are synthetic. No PANs, CVVs, or real PII.

### Class distribution (generator, before split)

Target prevalence is about 12% before flips, then slightly mixed by the 3% flip rate. That is **much higher** than typical card-not-present fraud (~0.1%). The REAL_DATASET track exists so prevalence can be inspected on a public corpus without contaminating this generator.

### Duplicate records

Training logs `duplicate_feature_rows` on the feature frame. Exact feature-vector duplicates are counted, not removed, so evaluation stays honest.

## Features

Shared list `FEATURE_COLUMNS` in `app/ml/features.py` (training and inference):

amount, account_age_days, failed_attempts, transaction_velocity, previous_transaction_count, previous_average_amount, current_device_known, current_location_known, amount_vs_avg_ratio, hour_of_day, day_of_week, payment_method_code, merchant_category_code.

These are legitimate production-style signals. They are also the same knobs the **generator uses to create labels**, so a strong model is partly recovering its own recipe. That is label-process leakage by construction, not a hidden extra column.

Graph, device-sharing, and IP-sharing are **not** model features. They enter the product score through the graph component and rules.

## Target definition

`is_fraud` is the generator flag (plus random flips). It is not a chargeback, not a confirmed fraud case, and not delayed-label fraud.

## Split strategy

70% train / 15% validation / 15% test, **stratified random**, seed 42.

Timestamps are sampled over a recent 14-day window and are **not** a real payment stream. A time-based split would be theatre on this generator. The public-dataset adapter **does** use a time-ordered split because `Time` is meaningful there.

Validation is used for:

- XGBoost `eval_set` (early stopping analogue / monitoring)
- isotonic `CalibratedClassifierCV(cv='prefit')`
- cost-based threshold sweep

Test is used once for reported metrics.

## Label leakage / contamination

| Risk | Status |
| --- | --- |
| Target copied into features | No |
| Train/test row overlap by design | Stratified split of independent draws; duplicate feature rows are counted |
| Temporal leakage | Random split; timestamps are not a causal timeline |
| Process leakage | Yes — labels are functions of the same behavioral knobs as the features |
| Graph features in supervised model | No |

## Overly predictive synthetic features

Binary `current_device_known` / `current_location_known` and `amount_vs_avg_ratio` are strong because the fraud builders set them. Overlap (normals can have some jitter; fraud has label flips) keeps holdout metrics below a perfect 1.0. Metrics were **not** degraded by injecting noise into scores after training.

## Isolation Forest

Fit on non-fraud **training** rows only. Contamination is derived from train prevalence. The forest is **global**, not per-user. Personalized amount/hour/device/location/velocity checks live in `anomaly/behavior.py` and are blended with the forest score.

## Probability vs risk score

- `ml_probability_raw` — booster `predict_proba`
- `ml_probability` — isotonic-calibrated P(fraud) when `calibrator.joblib` exists
- `ml_score` — calibrated P(fraud) × 100, a **risk score**, not a displayed percent unless calibration is on

Uncalibrated output must not be shown as a percentage probability.

## Metrics (this training run)

See `report.md` / `synthetic_metrics.json` for the latest numbers. A recent calibrated test split (n=1200, positive rate 0.15):

- PR-AUC ≈ 0.818, ROC-AUC ≈ 0.889, F1 ≈ 0.865, precision ≈ 0.966, recall ≈ 0.783, FPR ≈ 0.005
- Brier improved vs uncalibrated (0.036 vs 0.049)

Accuracy is not a primary metric.

The 0.5 cutoff in that table is **not** the product APPROVE/REVIEW/BLOCK threshold. Product decisions use `THRESHOLD_REVIEW` / `THRESHOLD_BLOCK` on the 0–100 combined risk score.

## Threshold experiment

`threshold_calibration.md` sweeps review/block probability cutoffs on **validation** calibrated probabilities using relative costs (`COST_FALSE_POSITIVE`, `COST_FALSE_NEGATIVE`, `COST_REVIEW`). That pair is an experiment on this generator, **not** an industry standard.

## Limitations

- Synthetic recovery of known patterns.
- Prevalence and feature distributions are not live acquirer traffic.
- No claim of transfer to REAL_DATASET or production.
- Calibration uses a single validation fold (prefit isotonic regression on booster scores), not nested CV.
