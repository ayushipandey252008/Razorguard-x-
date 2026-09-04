# Data

RazorGuard X has **three isolated tracks**. Do not mix their metrics or schemas.

| Track | Used by | Schema |
| --- | --- | --- |
| SYNTHETIC_DATASET | Live API, graph, rules, agent, dashboard traffic | user/device/IP/merchant payments |
| REAL_DATASET (ULB) | Offline supervised evaluation only | `Time`, `V1`–`V28`, `Amount`, `Class` |
| IEEE_CIS_OFFLINE | Offline IEEE-CIS candidate models only | `train_transaction.csv` + `train_identity.csv` |

The ULB file is **not** Razorpay data. Do not invent user, device, IP, or merchant columns for it.

IEEE-CIS is also **not** Razorpay data and is **not** routed to live scoring. See `docs/ieee-cis-evaluation.md`.

Place IEEE files in `ml/data/ieee/` (`IEEE_DATA_DIR`). They are gitignored. This prototype does not download them.

## ULB Credit Card Fraud Detection

| | |
| --- | --- |
| Dataset | ULB Credit Card Fraud Detection (Kaggle / Université Libre de Bruxelles) |
| Source | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) or the TensorFlow-hosted copy |
| Download URL used by the script | `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv` |
| Expected filename | `creditcard.csv` |
| Expected path | `ml/data/raw/creditcard.csv` (legacy: `ml/data/creditcard.csv`) |
| Schema | `Time`, `V1` … `V28`, `Amount`, `Class` |
| Target | `Class`: `0` = legitimate, `1` = fraud |
| Size | ~285k rows, ~150MB — **not committed** |
| Imbalance | ~0.17% fraud (hundreds of positives) |
| License / ethics | Research dataset of anonymized European card presentments (Sep 2013). PCA features are not interpretable business fields. |

### Limitations

- No device, IP, merchant, location, or account graph.
- Two days of traffic; `Time` is seconds from the first transaction, not a wall clock.
- Cannot drive RazorGuard X live scoring, rules, or investigations.
- Extreme class imbalance: **do not use accuracy** as the primary metric (PR-AUC is the headline).

### How to download / place the file

```bash
PYTHONPATH=. python ml/data/scripts/download_ulb.py
```

If download fails, copy `creditcard.csv` into `ml/data/raw/` yourself. Raw CSVs are gitignored.

### How preprocessing is reproduced

Nothing is edited by hand.

```bash
PYTHONPATH=. python ml/data/scripts/validate_ulb.py
PYTHONPATH=. python ml/data/scripts/preprocess_ulb.py
PYTHONPATH=. python ml/training/train_ulb.py
```

1. Load raw CSV.  
2. Validate required columns, dtypes, target ∈ {0,1}, class counts. Fail loudly if malformed.  
3. Drop **exact** duplicates only (count documented). Do not drop fraud outliers.  
4. Convert infinities to NaN.  
5. Split **chronologically** on `Time` (70/15/15).  
6. Fit imputer + scaler on **train only**.  
7. Train `ulb-xgb-v1` into `ml/models/ulb/` — never `ml/models/xgb_fraud.joblib`.

Adapter: `ml.ulb.adapter.ULBFraudDatasetAdapter` (`load`, `validate`, `preprocess`, `train`, `evaluate`).

Reports: `ml/evaluation/ulb_report.md`, `ml/evaluation/data_leakage_report.md`.
