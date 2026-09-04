# Model artifacts

RazorGuard X keeps three isolated model tracks. **Do not mix their metrics or overwrite the live files.**

## LIVE / PRODUCT-STYLE PIPELINE

| | |
| --- | --- |
| Version | `xgb-iforest-v1-calibrated` |
| Files | `xgb_fraud.joblib`, `calibrator.joblib`, `iforest.joblib`, `feature_columns.json`, `metrics.json`, `version.txt` |
| Trained on | Synthetic payment generator (`SYNTHETIC_DATASET`) |
| Used by | Live FastAPI scoring, UI, graph, rules, investigation agent |
| Git | Committed (small; required to run the API without retraining) |

Regenerate only if you intend to replace the live prototype model:

```bash
PYTHONPATH=backend python ml/training/train_baseline.py
```

## OFFLINE ULB (`ulb-xgb-v1`)

| | |
| --- | --- |
| Directory | `ml/models/ulb/` |
| Files | `model.joblib`, `preprocessor.joblib`, `probability_calibrator.joblib`, metadata JSON |
| Used by | Offline ULB evaluation only |
| Git | Joblibs gitignored (reproducible). Metadata JSON / README may be committed. |

```bash
PYTHONPATH=. python ml/training/train_ulb.py
```

These files must **never** replace `ml/models/xgb_fraud.joblib`.

## OFFLINE IEEE (`ieee-xgb-*`)

| | |
| --- | --- |
| Directory | `ml/models/ieee/` |
| Candidates | `ieee-xgb-baseline-v1`, `ieee-xgb-combined-v1`, `ieee-xgb-graph-v1` |
| Status | Always **CANDIDATE** — never auto-activated |
| Git | Joblibs gitignored (reproducible). JSON sidecars / README may be committed. |

```bash
PYTHONPATH=backend:. python ml/training/train_ieee.py
```

IEEE artifacts must **never** replace the live product model or ULB artifacts.

## Feedback candidates

`ml/models/feedback/<version>/` — offline CANDIDATE only. No activation endpoint. Joblibs are gitignored.
