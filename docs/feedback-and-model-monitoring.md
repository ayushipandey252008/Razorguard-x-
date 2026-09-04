# Feedback and model monitoring (Phase 7)

This is an independent student prototype and is not an official Razorpay product.

Analyst feedback, drift scores, and scenario catch rates are **lab measurements**. They are not real-world fraud accuracy and not production model governance.

## Three evaluation tracks (do not mix)

| Track | What it is | What it is not |
| --- | --- | --- |
| LIVE / PRODUCT-STYLE PIPELINE | Synchronous scoring on synthetic payments (`xgb-iforest-v1-calibrated`) | A production fraud engine |
| PUBLIC DATASET EVALUATION | Frozen ULB holdout + calibration artifacts | Live scores, scenario labels, or Razorpay data |
| IEEE-CIS OFFLINE EVALUATION | Chronological IEEE-CIS candidates under `ml/models/ieee/` | Live scoring or production payment-fraud accuracy |
| SYNTHETIC SCENARIO EVALUATION | Generator ground-truth vs the live pipeline | ULB Class, confirmed fraud, or real-world performance |

Never treat scenario catch rate as ULB PR-AUC. Never treat analyst `CONFIRM_FRAUD` as a ULB label.

## Analyst feedback

`POST /api/v1/feedback` records an **observation** about a decision:

- `CONFIRM_FRAUD` → `actual_outcome=FRAUD` (training label 1)
- `CONFIRM_LEGITIMATE` → `actual_outcome=LEGITIMATE` (training label 0)
- `NEEDS_REVIEW` → no training label (not treated as fraud)

The historical `risk_assessments` row is not updated. One feedback row per investigation (409 on duplicates). Analyst identity is the app-user id only.

Flow:

```
DB transaction
  persist analyst_feedback
  insert outbox AnalystFeedbackRecorded (correlation_id preserved)
commit
→ outbox worker → EventBus
```

Kafka is still optional. Feedback is not published while the domain transaction is open.

## Offline retraining

`backend/app/ml/train_feedback.py` (also `POST /api/v1/ml/train-feedback`, ADMIN):

1. Load defined-outcome feedback only
2. Drop `eval_*` scenario tags so scenario labels never train the candidate
3. Temporal split by `feedback.created_at` (later slice is eval, untouched)
4. Train a new booster
5. Compare against the current live model on that eval slice
6. Write `ml/models/feedback/<version>/` with status **CANDIDATE**

The live artifact `xgb-iforest-v1-calibrated` is not overwritten. There is **no activation endpoint**. A candidate stays offline until a later phase selects it explicitly.

## Model registry

`model_versions` now stores `model_id`, dataset, feature set, row counts, metrics, artifact path, and status:

- `ACTIVE` — currently loaded live scorer
- `CANDIDATE` — offline feedback model
- `RETIRED` — unused in this phase

## Drift

`GET /api/v1/ml/drift` computes **PSI** on amount, velocity, hour, device/location known flags, ml score, and final risk score.

Windows: earlier half vs later half of scored transactions by timestamp.

Prototype interpretation (not a production standard):

- PSI < 0.10 → LOW
- 0.10–0.25 → MODERATE
- > 0.25 → HIGH

Overall: `stable` / `warning` / `drift` / `insufficient`.

If overall is warning or drift, a `model-drift-detected` outbox event is emitted **at most once per cooldown** (`DRIFT_ALERT_COOLDOWN_SECONDS`). Recommendation: **"Review drift and evaluate retraining."** The API does not retrain.

## Scenario evaluation

`POST /api/v1/simulation/evaluate` runs named generators through the **same** live pipeline:

normal_payment, stolen_account, card_testing, high_velocity, unusual_amount, new_device, shared_device, shared_ip, device_farm, fraud_ring.

Tags are `eval_<name>` so they cannot enter the feedback training set. Graph scenarios also report cluster_found / size / backend. Optional investigation runs measure **grounding** (tool trace, limitations, evidence), not LLM accuracy.

Deterministic seeds. Metrics (precision, recall, F1, FPR, catch/block/review/approve rates) are vs generator labels only.

## Leakage protection

- Future feedback timestamps cannot enter training
- Eval slice is later than train
- Scenario-eval tags are excluded from feedback datasets
- Candidate training writes a new directory and checks live `xgb_fraud.joblib` mtime
- ULB evaluation artifacts are not rewritten by this phase

## Limitations

- Prototype only. Tiny labeled feedback, PSI on a local DB, synthetic scenarios.
- Not exactly-once, not production drift monitoring, not a champion/challenger platform.
- No Kubernetes, GNNs, feature store, multi-agent swarm, or RAG.
