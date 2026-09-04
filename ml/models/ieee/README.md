# IEEE-CIS offline candidates (`ieee-xgb-*`)

Offline **IEEE_CIS_OFFLINE** track only. Status is always **CANDIDATE**.

Examples: `ieee-xgb-baseline-v1`, `ieee-xgb-combined-v1`, `ieee-xgb-graph-v1`.

Joblib binaries are gitignored and can be reproduced with `PYTHONPATH=backend:. python ml/training/train_ieee.py` after placing IEEE CSVs under `ml/data/ieee/` (see `ml/data/ieee/README.md` and `docs/ieee-cis-evaluation.md`).

JSON sidecars may be committed. These files must never replace `ml/models/xgb_fraud.joblib` or ULB models.
