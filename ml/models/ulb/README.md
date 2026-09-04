# ULB offline model artifacts (`ulb-xgb-v1`)

Offline **REAL_DATASET** track only. Not the live RazorGuard X product model.

Joblib binaries are gitignored and can be reproduced with `PYTHONPATH=. python ml/training/train_ulb.py` after placing `creditcard.csv` under `ml/data/raw/` (see `ml/data/README.md`).

Metadata JSON may be committed. These files must never replace `ml/models/xgb_fraud.joblib`.
