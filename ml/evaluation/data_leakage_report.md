# Data leakage audit — ULB track

Generated 2026-09-03T06:31:06.364515+00:00. Official split: chronological 70/15/15.

| Check | Result |
| --- | --- |
| Duplicate leakage | Exact duplicates dropped **before** split (1081 rows). Remaining train/test exact overlap = 0. |
| Preprocessing leakage | Imputer and StandardScaler are fit on **train only** (n=198608). Derived features are row-wise. |
| Target leakage | `Class` is not in the feature matrix. No chargeback-derived extra columns. |
| Train/test contamination | Hash overlap train∩test = 0; train∩val = 0. Chronological Time(train) max ≤ Time(val/test) min: True. |
| Scaling leakage | Scaler means/scales come from train. Test transform uses those parameters. |
| Oversampling leakage | No SMOTE or resampling. `none — original training class distribution; scale_pos_weight only`. Evaluation uses original test prevalence. |
| Feature selection leakage | No univariate/model-based selection on the full dataset. Feature list is fixed a priori (V1–V28, Time, Amount + three row-wise derived columns). |

Random stratified split is reported only as a comparison. It can put later `Time` values in train and earlier values in test (`no_future_in_train=False`), so it is **not** the official model.

If any overlap count is non-zero, do not treat the metrics as a clean holdout.
