# ULB offline evaluation report

**Track: REAL_DATASET.** These numbers are **offline evaluation** on the public ULB credit-card
dataset. They are not live RazorGuard X scores and must not be mixed with SYNTHETIC_DATASET metrics.

This dataset is **not** Razorpay traffic. It has no user, device, IP, merchant, or location fields.

## Dataset

- Name: ULB Credit Card Fraud Detection
- Identifier: `ULB_CREDIT_CARD_FRAUD`
- Source: ULB Credit Card Fraud Detection
- File: `ml/data/raw/creditcard.csv`
- Rows after exact-duplicate drop: 283726
- Exact duplicates removed: 1081
- Class: 0 = legitimate, 1 = fraud
- Full-file prevalence (cleaned): 0.001667101358352777
- Fraud cases (cleaned): 473
- Legitimate cases (cleaned): 283253

## Preprocessing

- Schema validated in code (`ml/ulb/validate.py`). Malformed files fail loudly.
- Exact duplicates dropped only; fraud outliers are kept.
- Infinite values converted to NaN; median imputation **fitted on train**.
- StandardScaler on Time / Amount / derived columns **fitted on train**.
- V1–V28 passed through (publisher PCA). No extra user/device/IP features.

Cleaning notes:
- Exact duplicates dropped (identical Time, V1–V28, Amount, Class).
- Infinite feature values converted to NaN for train-only imputation later.
- No outlier clipping: fraud may itself be an outlier.
- No full-dataset scaling or imputation.
- Class labels preserved as 0/1.

## Class distribution (official chronological split)

| Split | n | fraud | legitimate | prevalence |
| --- | --- | --- | --- | --- |
| train | 198608 | 366 | 198242 | 0.001843 |
| val | 42559 | 55 | 42504 | 0.001292 |
| test | 42559 | 52 | 42507 | 0.001222 |

## Split methodology

Official strategy: **chronological** (sort by `Time`, 70/15/15). `Time` is seconds elapsed
from the first transaction, so a random split can train on later cards and test on earlier
ones. Chronological split forbids that.

Stratified-random comparison (not the official model):

- Chronological test PR-AUC: 0.757883672773321
- Stratified test PR-AUC: 0.7730157196913747
- Stratified `no_future_in_train`: False

## Feature pipeline

Baseline: Time, V1–V28, Amount.

Derived at inference time only: `log_amount`, `time_of_day_proxy` (`Time % 86400`),
`transaction_time_bucket` (`floor(Time/3600)`).

## Model

- Family: `xgboost`
- Version: `ulb-xgb-v1`
- Trained at (UTC): 2026-09-03T06:31:06.357521+00:00
- Does **not** overwrite the synthetic product model.

## Class imbalance strategy

- Training distribution is the **original** chronological train prevalence (no SMOTE / no resampling).
- `scale_pos_weight = n_neg / n_pos` on train = 541.6448087431694
- HistGBM fallback would use `class_weight='balanced'` instead.
- Threshold tuning uses **validation** only; test remains original prevalence.

## Metrics (chronological test, original class distribution)

PR-AUC is the headline metric.

### Default cutoff 0.5

| Metric | Value |
| --- | --- |
| PR-AUC | 0.757883672773321 |
| ROC-AUC | 0.9829638919200638 |
| Precision | 0.3076923076923077 |
| Recall | 0.7692307692307693 |
| F1 | 0.43956043956043955 |
| FPR | 0.002117298327334321 |
| FNR | 0.23076923076923078 |

Confusion: TN=42417 FP=90 FN=12 TP=40  
Test n=42559 fraud=52 legitimate=42507 prevalence=0.001221833219765502

Accuracy is omitted as a primary metric.

### Validation-tuned F1 threshold (0.9)

| Metric | Value |
| --- | --- |
| PR-AUC | 0.757883672773321 |
| Precision | 0.7358490566037735 |
| Recall | 0.75 |
| F1 | 0.7428571428571429 |
| FPR | 0.0003293575175853389 |

PR-AUC is threshold-free and is the same as the 0.5 table when probabilities are unchanged.

## SHAP analysis

V1–V28 are anonymized PCA components from the dataset publisher. SHAP ranking does not recover a business feature name.

Global mean |SHAP| (top 10):

- V4: 1.326810
- V14: 1.076012
- V12: 0.765563
- V10: 0.611373
- V11: 0.334092
- V8: 0.314489
- V1: 0.285391
- V17: 0.214499
- time_of_day_proxy: 0.208607
- V18: 0.190400

Local examples are stored in `ml/evaluation/ulb_shap.json`. They are PCA attributions, not KYC narratives.

## Leakage checks

See `ml/evaluation/data_leakage_report.md`.

- Train/test exact-row overlap: 0
- Train/val exact-row overlap: 0
- Imputer/scaler fitted on train n=198608
- Resampling applied: none — original training class distribution; scale_pos_weight only

## Limitations

- PCA features have no device/IP/merchant semantics.
- Two-day 2013 European card presentments; not India UPI and not Razorpay.
- Chronological test prevalence can differ from train (concept drift / delayed fraud labels).
- Single prefit split; not nested cross-validation.
- No graph, rules, or investigation agent on this track.

## Reproducibility

```bash
PYTHONPATH=. python ml/data/scripts/download_ulb.py
PYTHONPATH=. python ml/data/scripts/validate_ulb.py
PYTHONPATH=. python ml/training/train_ulb.py
```

Place `creditcard.csv` in `ml/data/raw/` if download is blocked. Raw CSV is gitignored.
