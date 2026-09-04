from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ml.ulb.constants import DATASET_NAME, TRACK


def render_ulb_report(payload: dict) -> str:
    m = payload["metrics_at_half"]
    tuned = payload.get("metrics_at_val_threshold") or {}
    cm = m["confusion_matrix"]
    dist = payload["class_distribution"]
    split = payload["official_split"]
    shap = payload.get("shap") or {}
    leak = payload.get("leakage") or {}
    cmp_ = payload.get("split_comparison") or {}
    return f"""# ULB offline evaluation report

**Track: {TRACK}.** These numbers are **offline evaluation** on the public ULB credit-card
dataset. They are not live RazorGuard X scores and must not be mixed with SYNTHETIC_DATASET metrics.

This dataset is **not** Razorpay traffic. It has no user, device, IP, merchant, or location fields.

## Dataset

- Name: {DATASET_NAME}
- Identifier: `{payload.get("dataset_id")}`
- Source: {payload.get("source")}
- File: `{payload.get("raw_path")}`
- Rows after exact-duplicate drop: {payload.get("n_rows_cleaned")}
- Exact duplicates removed: {payload.get("duplicates_removed")}
- Class: 0 = legitimate, 1 = fraud
- Full-file prevalence (cleaned): {dist.get("full_prevalence")}
- Fraud cases (cleaned): {dist.get("full_fraud")}
- Legitimate cases (cleaned): {dist.get("full_legitimate")}

## Preprocessing

- Schema validated in code (`ml/ulb/validate.py`). Malformed files fail loudly.
- Exact duplicates dropped only; fraud outliers are kept.
- Infinite values converted to NaN; median imputation **fitted on train**.
- StandardScaler on Time / Amount / derived columns **fitted on train**.
- V1–V28 passed through (publisher PCA). No extra user/device/IP features.

Cleaning notes:
{chr(10).join("- " + n for n in (payload.get("cleaning_notes") or []))}

## Class distribution (official chronological split)

| Split | n | fraud | legitimate | prevalence |
| --- | --- | --- | --- | --- |
| train | {split["train"]["n"]} | {split["train"]["fraud"]} | {split["train"]["legitimate"]} | {split["train"]["prevalence"]:.6f} |
| val | {split["val"]["n"]} | {split["val"]["fraud"]} | {split["val"]["legitimate"]} | {split["val"]["prevalence"]:.6f} |
| test | {split["test"]["n"]} | {split["test"]["fraud"]} | {split["test"]["legitimate"]} | {split["test"]["prevalence"]:.6f} |

## Split methodology

Official strategy: **chronological** (sort by `Time`, 70/15/15). `Time` is seconds elapsed
from the first transaction, so a random split can train on later cards and test on earlier
ones. Chronological split forbids that.

Stratified-random comparison (not the official model):

- Chronological test PR-AUC: {cmp_.get("chronological_pr_auc")}
- Stratified test PR-AUC: {cmp_.get("stratified_pr_auc")}
- Stratified `no_future_in_train`: {cmp_.get("stratified_no_future_in_train")}

## Feature pipeline

Baseline: Time, V1–V28, Amount.

Derived at inference time only: `log_amount`, `time_of_day_proxy` (`Time % 86400`),
`transaction_time_bucket` (`floor(Time/3600)`).

## Model

- Family: `{payload.get("booster_family")}`
- Version: `{payload.get("model_version")}`
- Trained at (UTC): {payload.get("trained_at")}
- Does **not** overwrite the synthetic product model.

## Class imbalance strategy

- Training distribution is the **original** chronological train prevalence (no SMOTE / no resampling).
- `scale_pos_weight = n_neg / n_pos` on train = {payload.get("scale_pos_weight")}
- HistGBM fallback would use `class_weight='balanced'` instead.
- Threshold tuning uses **validation** only; test remains original prevalence.

## Metrics (chronological test, original class distribution)

PR-AUC is the headline metric.

### Default cutoff 0.5

| Metric | Value |
| --- | --- |
| PR-AUC | {m.get("pr_auc")} |
| ROC-AUC | {m.get("roc_auc")} |
| Precision | {m.get("precision")} |
| Recall | {m.get("recall")} |
| F1 | {m.get("f1")} |
| FPR | {m.get("false_positive_rate")} |
| FNR | {m.get("false_negative_rate")} |

Confusion: TN={cm.get("tn")} FP={cm.get("fp")} FN={cm.get("fn")} TP={cm.get("tp")}  
Test n={m.get("n_samples")} fraud={m.get("n_fraud")} legitimate={m.get("n_legitimate")} prevalence={m.get("fraud_prevalence")}

Accuracy is omitted as a primary metric.

### Validation-tuned F1 threshold ({payload.get("val_threshold")})

| Metric | Value |
| --- | --- |
| PR-AUC | {tuned.get("pr_auc")} |
| Precision | {tuned.get("precision")} |
| Recall | {tuned.get("recall")} |
| F1 | {tuned.get("f1")} |
| FPR | {tuned.get("false_positive_rate")} |

PR-AUC is threshold-free and is the same as the 0.5 table when probabilities are unchanged.

## SHAP analysis

{shap.get("limitation") or shap.get("reason") or ""}

Global mean |SHAP| (top 10):

{chr(10).join(
    f"- {row['feature']}: {row['mean_abs_shap']:.6f}"
    for row in (shap.get("global_mean_abs_shap") or [])[:10]
) or "- SHAP unavailable"}

Local examples are stored in `ml/evaluation/ulb_shap.json`. They are PCA attributions, not KYC narratives.

## Leakage checks

See `ml/evaluation/data_leakage_report.md`.

- Train/test exact-row overlap: {leak.get("train_test_overlap")}
- Train/val exact-row overlap: {leak.get("train_val_overlap")}
- Imputer/scaler fitted on train n={leak.get("n_train_fit")}
- Resampling applied: {leak.get("resampling")}

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
"""


def render_leakage_report(payload: dict) -> str:
    leak = payload["leakage"]
    return f"""# Data leakage audit — ULB track

Generated {datetime.now(timezone.utc).isoformat()}. Official split: chronological 70/15/15.

| Check | Result |
| --- | --- |
| Duplicate leakage | Exact duplicates dropped **before** split ({payload.get("duplicates_removed")} rows). Remaining train/test exact overlap = {leak.get("train_test_overlap")}. |
| Preprocessing leakage | Imputer and StandardScaler are fit on **train only** (n={leak.get("n_train_fit")}). Derived features are row-wise. |
| Target leakage | `Class` is not in the feature matrix. No chargeback-derived extra columns. |
| Train/test contamination | Hash overlap train∩test = {leak.get("train_test_overlap")}; train∩val = {leak.get("train_val_overlap")}. Chronological Time(train) max ≤ Time(val/test) min: {leak.get("chronological_order_ok")}. |
| Scaling leakage | Scaler means/scales come from train. Test transform uses those parameters. |
| Oversampling leakage | No SMOTE or resampling. `{leak.get("resampling")}`. Evaluation uses original test prevalence. |
| Feature selection leakage | No univariate/model-based selection on the full dataset. Feature list is fixed a priori (V1–V28, Time, Amount + three row-wise derived columns). |

Random stratified split is reported only as a comparison. It can put later `Time` values in train and earlier values in test (`no_future_in_train={payload.get("split_comparison", {}).get("stratified_no_future_in_train")}`), so it is **not** the official model.

If any overlap count is non-zero, do not treat the metrics as a clean holdout.
"""


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
