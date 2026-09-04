# IEEE-CIS evaluation

The IEEE-CIS experiment is an offline public-dataset evaluation. It does not represent production payment-fraud performance.

This is an independent student prototype, not an official Razorpay product. Live scoring continues to use `xgb-iforest-v1-calibrated`. IEEE models are **OFFLINE CANDIDATE** artifacts and are never auto-activated.

## 1. Dataset

IEEE-CIS Fraud Detection (Kaggle / Vesta). Expected local files:

- `train_transaction.csv`
- `train_identity.csv`

Configurable path: `IEEE_DATA_DIR` (default `ml/data/ieee/`). Raw CSVs are gitignored and are **not** downloaded automatically.

If the files are missing, the adapter prints a setup message. Tests use a tiny synthetic fixture that mimics table *shape* only. Fixture metrics are **not** IEEE-CIS results.

Do not mix IEEE-CIS rows into ULB evaluation. Do not mix synthetic scenario labels into IEEE-CIS evaluation.

## 2. Data audit

Before training the pipeline writes:

- `ml/evaluation/ieee_data_audit.json`
- `ml/evaluation/ieee_data_audit.md`

The audit reports row/column counts, target distribution, missingness, duplicate rows and TransactionIDs, numerical vs categorical vs identity vs time columns, memory usage, and identity-join coverage. Suspicious columns are listed, not dropped silently.

## 3. Join

Identity is left-joined onto transaction on the official key `TransactionID`. The target `isFraud` comes only from the transaction table. The join is never performed on a target-derived key.

The join report records rows before/after, unmatched identity keys, duplicate keys, and one-to-many identity keys. `identity_present` flags whether identity attributes were available.

## 4. Temporal split

Official evaluation uses a chronological split on `TransactionDT` (default 70% / 15% / 15%).

Verified:

- `max(train_time) < min(validation_time)`
- `max(validation_time) < min(test_time)`

A random stratified split is **not** the official evaluation. Preprocessing is fit on train only.

## 5. Leakage controls

`backend/app/ml/ieee/leakage.py` checks target leakage, TransactionID leakage, raw-time leakage, train/test overlap, duplicate IDs across splits, preprocessing fit scope, target encoding (not used), and time-aware graph features.

Every excluded feature has a documented reason in `ml/evaluation/ieee_leakage_report.md`.

## 6. Feature engineering

Families:

| Family | Examples |
| --- | --- |
| Transaction | amount, ProductCD, hour-of-day proxy, selected C/D/M |
| Card | card1–card6 available before prediction |
| Address | addr1, addr2, dist1 |
| Identity | device / OS / browser / id_* / identity_present |
| Email | P/R email domain |
| Behavioral | prior-only entity counts and amount deviation |
| Graph | time-aware entity degree and prior-label counts |

Categorical encoding is **frequency encoding fitted on train only**. Unseen categories map to 0. Target encoding is not used (it would need nested temporal out-of-fold protection). One-hot is avoided for high-cardinality columns.

Missingness: train-median impute plus missingness indicators. Actual zeros are not treated as missing. Unavailable identity is `identity_present=0`, not a blanket zero-fill.

## 7. Behavioral features

Transaction-level history uses only rows with `TransactionDT < T`:

- prior count and mean amount on card1
- amount ratio vs that baseline
- seconds since previous card transaction
- prior email / device counts

Future transactions are not used.

## 8. Graph features

Offline graph-derived features are **not** a replacement for live NetworkX / optional Neo4j.

Construction: scan in `TransactionDT` order. Emit features from the current entity dictionaries, **then** update those dictionaries. Prior fraud counts use historical labels only.

IEEE-CIS identity has no raw IP. `graph_ip_entity_unavailable=1`; IP counts are not fabricated. `addr1` is a coarse geo/billing entity, not an IP.

## 9. Model experiments

Same preprocessor and chronological test for:

- A transaction-only → `ieee-xgb-baseline-v1`
- B transaction + card
- C transaction + identity
- D transaction + behavioral
- E transaction + graph → `ieee-xgb-graph-v1`
- F combined → `ieee-xgb-combined-v1`

Class imbalance: `scale_pos_weight` / HistGBM `class_weight`. SMOTE is not used.

## 10. Calibration

Raw vs sigmoid/Platt vs isotonic, fit on validation scores only. Selection follows the ULB robustness lesson: do not pick isotonic only because in-sample Brier is lower. A low unique-probability count prefers sigmoid.

`model_probability` is kept separate from the live product `final_risk_score`. It is not a production fraud probability.

## 11. Thresholds

APPROVE / REVIEW / BLOCK cutoffs are chosen on validation with documented relative costs and evaluated **once** on the frozen chronological test. They are not industry-standard operating points.

## 12. Results

Measured numbers live in `ml/evaluation/ieee_experiment_manifest.json` when the public CSVs are present. If the dataset is missing, that file records setup status and **does not** invent IEEE-CIS PR-AUC.

Headline metrics: PR-AUC (primary), ROC-AUC, precision, recall, F1, FPR, FNR, confusion matrix, fraud catch rate, prevalence.

## 13. Ablations

Graph ablation compares combined features with vs without graph columns on the same frozen test. If graph features do not improve PR-AUC / recall / precision / F1 / FPR, the report says so.

## 14. Limitations

- Public contest data, not Razorpay traffic and not live payments.
- Different schema, time span, and prevalence than ULB; metrics are not equivalent.
- Anonymized `id_*` / `V*` names are not business meanings unless contest docs define them.
- SHAP is a model explanation, not a causal explanation.
- Full IEEE-CIS can exceed laptop memory. Set `IEEE_MAX_ROWS` rather than fabricating a completed run.
- Candidate artifacts must not be read as production fraud accuracy.

## 15. Candidate model handling

Artifacts: `ml/models/ieee/<version>.joblib` and `.json`.

Status: **CANDIDATE**. `deployed_to_live_pipeline=False`. The live application keeps `xgb-iforest-v1-calibrated`. There is no activation endpoint for IEEE models.

Train:

```bash
PYTHONPATH=backend:. python ml/training/train_ieee.py
```

Optional `--fixture` is for tests/demo only and is labeled `SYNTHETIC_FIXTURE_NOT_IEEE_CIS`.
