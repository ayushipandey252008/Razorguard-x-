# IEEE-CIS results

The IEEE-CIS experiment is an offline public-dataset evaluation. It does not represent production payment-fraud performance.

**Status:** COMPLETED
**Official IEEE-CIS result:** True
**Label:** OFFLINE PUBLIC DATASET EVALUATION
**Source:** `IEEE_CIS_CSV`
**Limited sample:** False (max_rows=None)
**ACTIVE MODEL:** `xgb-iforest-v1-calibrated`
**IEEE-CIS:** OFFLINE CANDIDATE (auto_activated=False)

## Dataset

- Transaction rows/columns: 590540 / 394
- Identity rows/columns: 144233 / 41
- Fraud / legitimate / prevalence: 20663 / 569877 / 0.03499000914417313
- Duplicate txn IDs: 0; duplicate identity IDs: 0
- Identity join coverage: 0.2442391709283029
- Unmatched identity rows: 0
- Numerical / categorical columns (joined): 390 / 43
- TransactionDT range (train min → test max): 86400 → 15811131
- TransactionDT is a contest timedelta in seconds, not a wall clock.
- Memory bytes (txn / identity / joined): 1279380966 / 201546271 / 1914751117

## Chronological split (70 / 15 / 15)

| Split | Rows | Fraud | Prevalence | time_min | time_max |
| --- | --- | --- | --- | --- | --- |
| train | 413378 | 14538 | 0.0351688 | 86400 | 10437996 |
| validation | 88581 | 3042 | 0.0343415 | 10438003 | 13151840 |
| test | 88581 | 3083 | 0.0348043 | 13151880 | 15811131 |

- max(train) < min(validation): True
- max(validation) < min(test): True

## Leakage

- All checks passed: True
- Excluded features: 409
- [PASS] `target_leakage` — isFraud is not a covariate.
- [PASS] `transaction_id_leakage` — TransactionID is not a covariate.
- [PASS] `raw_time_leakage` — Raw TransactionDT excluded; hour_of_day_proxy may be used.
- [PASS] `target_encoding_leakage` — Target encoding is not used.
- [PASS] `preprocessing_fit_scope` — train_only
- [PASS] `train_test_exact_overlap` — exact overlapping hashes=0
- [PASS] `train_val_exact_overlap` — exact overlapping hashes=0
- [PASS] `duplicate_ids_across_splits` — TransactionID overlap train/test=0
- [PASS] `temporal_split_order` — max(train) < min(val) < max(val) < min(test)
- [PASS] `graph_temporal_safety` — Graph features recomputed from strict past for a late test row.
- [PASS] `join_not_on_target` — Join key is TransactionID, not a target-derived key.
- [PASS] `post_outcome_fields` — No chargeback/outcome columns are in the IEEE-CIS public train schema used here.

## Experiments (frozen chronological TEST, threshold 0.5, uncalibrated scores)

| Experiment | Features | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_transaction_only | transaction | 0.333317 | 0.81556 | 0.113102 | 0.669478 | 0.193512 | 0.189303 |
| B_transaction_card | transaction+card | 0.349929 | 0.832217 | 0.123131 | 0.649043 | 0.206993 | 0.166671 |
| C_transaction_identity | transaction+identity | 0.365525 | 0.816071 | 0.125047 | 0.643853 | 0.209421 | 0.162448 |
| D_transaction_behavioral | transaction+behavioral | 0.312714 | 0.821402 | 0.110981 | 0.674019 | 0.190581 | 0.194695 |
| E_transaction_graph | transaction+graph | 0.374836 | 0.865177 | 0.125572 | 0.730133 | 0.214289 | 0.183338 |
| F_combined | transaction+card+address+identity+email+behavioral+graph | 0.345505 | 0.869226 | 0.113296 | 0.773597 | 0.197646 | 0.218321 |
| ablation_no_graph | transaction+card+address+identity+email+behavioral | 0.33381 | 0.833571 | 0.113383 | 0.672721 | 0.194058 | 0.189689 |

## Graph ablation (combined families, TEST @ 0.5, uncalibrated)

- Without graph PR-AUC / recall / precision / F1 / FPR: 0.33381 / 0.672721 / 0.113383 / 0.194058 / 0.189689
- With graph PR-AUC / recall / precision / F1 / FPR: 0.345505 / 0.773597 / 0.113296 / 0.197646 / 0.218321
- Improved PR-AUC: True; recall: True; precision: False; F1: True; FPR: False

## Calibration (fit on validation only)

- Selected: isotonic
- Justification: Selected isotonic using validation diagnostics with an isotonic uniqueness guard (Brier=0.02458479).

| Method | Val Brier | Val log loss | Val ECE | unique preds |
| --- | --- | --- | --- | --- |
| isotonic | 0.0245848 | 0.10041 | 3.8458e-18 | 68 |
| sigmoid | 0.0254482 | 0.103504 | 0.00531245 | 85435 |
| raw | 0.138 | 0.432477 | 0.289238 | 85435 |

- TEST raw Brier / log loss / ECE: 0.159766 / 0.494708 / 0.312868
- TEST calibrated Brier / log loss / ECE: 0.0302087 / 0.117352 / 0.0125923
- Validation calibrator PR-AUC/ROC-AUC: not available

## Thresholds (validation-selected, applied once to TEST)

- APPROVE below: 0.020004
- REVIEW: 0.020004 to 0.511628
- BLOCK above: 0.511628
- Source: validation_only
- TEST policy counts: {'APPROVE': 62927, 'REVIEW': 23809, 'BLOCK': 1845}
- TEST fraud catch (REVIEW or BLOCK): 0.8076548816088226
- model_probability is not final_risk_score.

## Frozen chronological TEST (selected calibrator, threshold 0.5)

- PR-AUC: 0.337152
- ROC-AUC: 0.868842
- Precision / recall / F1: 0.393396 / 0.270516 / 0.320584
- FPR / FNR / catch rate: 0.0150413 / 0.729484 / 0.270516
- Confusion: {'tn': 84212, 'fp': 1286, 'fn': 2249, 'tp': 834}
- Prevalence: 0.0348043

## SHAP (model explanation, not causality)

- Available: True
- Rows explained: 24
- `graph_card_prior_fraud_count`: 1.46892
- `graph_addr_account_count`: 0.310122
- `graph_suspicious_cluster`: 0.307018
- `graph_card_cluster_size`: 0.257918
- `TransactionAmt`: 0.246885
- `card_prior_count`: 0.21887
- `C1`: 0.207477
- `D1`: 0.186462
- `C2`: 0.156478
- `card6_freq`: 0.151966

## Candidates

- Status: OFFLINE CANDIDATE. Live model unchanged: `xgb-iforest-v1-calibrated`.
- `ieee-xgb-baseline-v1` families=['transaction'] features=9 path=`./ml/models/ieee/ieee-xgb-baseline-v1.joblib`
- `ieee-xgb-combined-v1` families=['transaction', 'card', 'address', 'identity', 'email', 'behavioral', 'graph'] features=49 path=`./ml/models/ieee/ieee-xgb-combined-v1.joblib`
- `ieee-xgb-graph-v1` families=['transaction', 'graph'] features=18 path=`./ml/models/ieee/ieee-xgb-graph-v1.joblib`

## Cross-dataset (not equivalent)

| Dataset | Rows | Fraud | Features | PR-AUC | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| ULB Credit Card Fraud Detection | 283726 | 473 | 33 | 0.757884 | 0.982964 |
| IEEE-CIS Fraud Detection | 590540 | 20663 | 49 | 0.337152 | 0.868842 |

Different feature spaces, time periods, fraud prevalence, collection processes, entity information, and evaluation conditions. Metrics are not interchangeable.

## Runtime

- Total seconds: 132.31985795900255
- Stage timers (audit/join/preprocess/graph/train/infer/calibrate): not available / not available / not available / not available / not available / not available / not available
- Peak memory: not available
- Platform: macOS-15.7.7-arm64-arm-64bit

## Integrity

- Live version: `xgb-iforest-v1-calibrated`

## Limitations

- The IEEE-CIS experiment is an offline public-dataset evaluation. It does not represent production payment-fraud performance.
- IEEE-CIS metrics are not comparable to ULB or live synthetic scores.
- model_probability is not the product final_risk_score and is not a production fraud probability.
- IEEE candidates stay CANDIDATE and are not auto-activated.
- Anonymized id_*/V* names are not business meanings unless contest documentation defines them.
- SHAP is a model explanation, not a causal explanation.
