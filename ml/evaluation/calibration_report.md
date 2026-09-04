# ULB probability and threshold calibration

**PROTOTYPE CALIBRATION.** These operating points are not industry-standard thresholds
and are not the live RazorGuard X product policy.

Track: `REAL_DATASET`. Booster: `ulb-xgb-v1`.
Calibrated identity: `ulb-xgb-v1-calibrated`.
Synthetic live model `xgb-iforest-v1-calibrated` was not modified.

Calibrated P(Class=1) is **not** the product `final_risk_score`.

## 1. Dataset split

Official chronological 70/15/15 on Time (unchanged from ULB training).

| split | n | fraud | legitimate | prevalence | time_min | time_max |
| --- | --- | --- | --- | --- | --- | --- |
| train | 198608 | 366 | 198242 | 0.001843 | 0.0 | 132906.0 |
| validation | 42559 | 55 | 42504 | 0.001292 | 132906.0 | 151320.0 |
| test | 42559 | 52 | 42507 | 0.001222 | 151320.0 | 172792.0 |

`no_future_in_train`: True
Split matches `ulb_metrics.json`: True

## 2. Calibration methodology

- TRAIN: existing `ulb-xgb-v1` booster is reused. It is not refit here.
- VALIDATION (42559 rows, 55 positives): fit sigmoid and isotonic on **raw validation scores only**; select method; select thresholds/costs.
- TEST (42559 rows): evaluate the frozen calibrator and frozen thresholds **once**.

| gate | used for fit? | used for selection? |
| --- | --- | --- |
| validation labels | yes (calibrators + thresholds) | yes |
| test labels | False | False / False |

Sigmoid/Platt = `LogisticRegression` on 1-D raw scores.
Isotonic = `IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")`.
Raw = uncalibrated booster probabilities.

Reliability diagrams: `ml/evaluation/figures/ulb_reliability_val_{raw,sigmoid,isotonic}.svg`.

## 3–5. Raw / sigmoid / isotonic (validation)

| method | Brier | log loss | ECE (uniform 10) | ECE (quantile 10) | mean predicted | unique p |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 0.00236261 | 0.014619 | 0.011792 | 0.011792 | 0.013085 | 40677 |
| sigmoid | 0.00049506 | 0.002706 | 0.000696 | 0.000455 | 0.001266 | 40674 |
| isotonic | 0.00030448 | 0.001783 | 0.000000 | 0.000000 | 0.001292 | 11 |

Prevalence on validation is 0.001292. Mean predicted probability
far above that prevalence means the booster is over-confident in the positive direction on average.

### Raw — reliability (uniform 10 bins)

| bin | lo | hi | n | positives | mean predicted | empirical rate | |gap| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.100 | 41644 | 4 | 0.006947 | 0.000096 | 0.006851 |
| 1 | 0.100 | 0.200 | 496 | 2 | 0.139159 | 0.004032 | 0.135127 |
| 2 | 0.200 | 0.300 | 153 | 0 | 0.243382 | 0.000000 | 0.243382 |
| 3 | 0.300 | 0.400 | 60 | 1 | 0.346935 | 0.016667 | 0.330269 |
| 4 | 0.400 | 0.500 | 52 | 1 | 0.442082 | 0.019231 | 0.422852 |
| 5 | 0.500 | 0.600 | 38 | 0 | 0.556025 | 0.000000 | 0.556025 |
| 6 | 0.600 | 0.700 | 34 | 1 | 0.647918 | 0.029412 | 0.618506 |
| 7 | 0.700 | 0.800 | 22 | 2 | 0.737145 | 0.090909 | 0.646236 |
| 8 | 0.800 | 0.900 | 10 | 1 | 0.865575 | 0.100000 | 0.765575 |
| 9 | 0.900 | 1.000 | 50 | 43 | 0.989202 | 0.860000 | 0.129202 |

### Sigmoid — reliability (uniform 10 bins)

| bin | lo | hi | n | positives | mean predicted | empirical rate | |gap| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.100 | 42476 | 9 | 0.000433 | 0.000212 | 0.000222 |
| 1 | 0.100 | 0.200 | 21 | 1 | 0.132205 | 0.047619 | 0.084586 |
| 2 | 0.200 | 0.300 | 4 | 1 | 0.224505 | 0.250000 | 0.025495 |
| 3 | 0.300 | 0.400 | 8 | 1 | 0.348208 | 0.125000 | 0.223208 |
| 4 | 0.400 | 0.500 | 5 | 1 | 0.457332 | 0.200000 | 0.257332 |
| 5 | 0.500 | 0.600 | 20 | 17 | 0.584797 | 0.850000 | 0.265203 |
| 6 | 0.600 | 0.700 | 25 | 25 | 0.601550 | 1.000000 | 0.398450 |
| 7 | 0.700 | 0.800 | 0 | 0 | — | — | — |
| 8 | 0.800 | 0.900 | 0 | 0 | — | — | — |
| 9 | 0.900 | 1.000 | 0 | 0 | — | — | — |

### Isotonic — reliability (uniform 10 bins)

| bin | lo | hi | n | positives | mean predicted | empirical rate | |gap| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.100 | 42498 | 10 | 0.000235 | 0.000235 | 0.000000 |
| 1 | 0.100 | 0.200 | 13 | 2 | 0.153846 | 0.153846 | 0.000000 |
| 2 | 0.200 | 0.300 | 7 | 2 | 0.285714 | 0.285714 | 0.000000 |
| 3 | 0.300 | 0.400 | 0 | 0 | — | — | — |
| 4 | 0.400 | 0.500 | 0 | 0 | — | — | — |
| 5 | 0.500 | 0.600 | 0 | 0 | — | — | — |
| 6 | 0.600 | 0.700 | 0 | 0 | — | — | — |
| 7 | 0.700 | 0.800 | 0 | 0 | — | — | — |
| 8 | 0.800 | 0.900 | 0 | 0 | — | — | — |
| 9 | 0.900 | 1.000 | 41 | 41 | 1.000000 | 1.000000 | 0.000000 |

## 6. Selected method and justification

**Selected: `isotonic`**

Lowest validation Brier is isotonic (Brier=0.00030448, ECE_uniform=3.619654260617094e-20).

- Selection uses validation Brier first, then uniform-bin ECE, then log loss.
- Test labels are not used for selection.
- On ~0.12% prevalence, Brier is dominated by negatives; ECE bins are sparse.
- Isotonic and Platt are both fit on validation, so validation Brier/ECE give isotonic an in-sample advantage. Test Brier is the confirmatory check.
- Isotonic produced 11 distinct probabilities. That stepwise map can improve Brier while compressing ranking (PR-AUC).

## 7. Reliability metrics

See the method table above. Quantile ECE is included because uniform bins are almost empty
except near 0 under extreme imbalance. Neither ECE variant was computed on test for selection.

## 8. Threshold analysis (validation, calibrated probabilities)

Full sweep is in `calibration_metrics.json` (`binary_threshold_sweep_validation`).
Excerpt (5-point steps plus a few operating points):

| threshold | TP | FP | TN | FN | precision | recall | F1 | FPR | FNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.010 | 49 | 186 | 42318 | 6 | 0.2085 | 0.8909 | 0.3379 | 0.004376 | 0.1091 |
| 0.020 | 47 | 58 | 42446 | 8 | 0.4476 | 0.8545 | 0.5875 | 0.001365 | 0.1455 |
| 0.050 | 46 | 26 | 42478 | 9 | 0.6389 | 0.8364 | 0.7244 | 0.000612 | 0.1636 |
| 0.100 | 45 | 16 | 42488 | 10 | 0.7377 | 0.8182 | 0.7759 | 0.000376 | 0.1818 |
| 0.150 | 44 | 10 | 42494 | 11 | 0.8148 | 0.8000 | 0.8073 | 0.000235 | 0.2000 |
| 0.200 | 43 | 5 | 42499 | 12 | 0.8958 | 0.7818 | 0.8350 | 0.000118 | 0.2182 |
| 0.250 | 43 | 5 | 42499 | 12 | 0.8958 | 0.7818 | 0.8350 | 0.000118 | 0.2182 |
| 0.300 | 41 | 0 | 42504 | 14 | 1.0000 | 0.7455 | 0.8542 | 0.000000 | 0.2545 |
| 0.350 | 41 | 0 | 42504 | 14 | 1.0000 | 0.7455 | 0.8542 | 0.000000 | 0.2545 |
| 0.400 | 41 | 0 | 42504 | 14 | 1.0000 | 0.7455 | 0.8542 | 0.000000 | 0.2545 |
| 0.450 | 41 | 0 | 42504 | 14 | 1.0000 | 0.7455 | 0.8542 | 0.000000 | 0.2545 |
| 0.500 | 41 | 0 | 42504 | 14 | 1.0000 | 0.7455 | 0.8542 | 0.000000 | 0.2545 |
| 0.550 | 41 | 0 | 42504 | 14 | 1.0000 | 0.7455 | 0.8542 | 0.000000 | 0.2545 |
| 0.600 | 41 | 0 | 42504 | 14 | 1.0000 | 0.7455 | 0.8542 | 0.000000 | 0.2545 |
| 0.650 | 41 | 0 | 42504 | 14 | 1.0000 | 0.7455 | 0.8542 | 0.000000 | 0.2545 |
| 0.700 | 41 | 0 | 42504 | 14 | 1.0000 | 0.7455 | 0.8542 | 0.000000 | 0.2545 |
| 0.750 | 41 | 0 | 42504 | 14 | 1.0000 | 0.7455 | 0.8542 | 0.000000 | 0.2545 |
| 0.800 | 41 | 0 | 42504 | 14 | 1.0000 | 0.7455 | 0.8542 | 0.000000 | 0.2545 |
| 0.850 | 41 | 0 | 42504 | 14 | 1.0000 | 0.7455 | 0.8542 | 0.000000 | 0.2545 |
| 0.900 | 41 | 0 | 42504 | 14 | 1.0000 | 0.7455 | 0.8542 | 0.000000 | 0.2545 |
| 0.950 | 41 | 0 | 42504 | 14 | 1.0000 | 0.7455 | 0.8542 | 0.000000 | 0.2545 |
| 0.990 | 41 | 0 | 42504 | 14 | 1.0000 | 0.7455 | 0.8542 | 0.000000 | 0.2545 |

Best validation F1 cutoff (not automatically the three-way prototype):
threshold=0.2900, F1=0.8542,
precision=1.0000, recall=0.7455.

## 9. Cost scenarios

Costs are configuration parameters. They are not estimated from Razorpay or any issuer.

### Scenario A

- FN cost: 100.0
- FP cost: 5.0
- Review cost: 2.0

| variant | T_REVIEW | T_BLOCK | expected cost / txn | approve | review | block | fraud catch | feasible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unconstrained | 0.02 | 0.29 | 0.021805 | 0.9975 | 0.0015 | 0.0010 | 0.8545 | true |
| 5% capacity cap | 0.02 | 0.29 | 0.021805 | 0.9975 | 0.0015 | 0.0010 | 0.8545 | true |
| **selected prototype** | **0.02** | **0.29** | 0.021805 | 0.9975 | 0.0015 | 0.0010 | 0.8545 | true |

Selection rule: min expected cost on validation under non-approve rate <= 5%

### Scenario B

- FN cost: 50.0
- FP cost: 10.0
- Review cost: 3.0

| variant | T_REVIEW | T_BLOCK | expected cost / txn | approve | review | block | fraud catch | feasible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unconstrained | 0.04 | 0.29 | 0.012759 | 0.9983 | 0.0007 | 0.0010 | 0.8364 | true |
| 5% capacity cap | 0.04 | 0.29 | 0.012759 | 0.9983 | 0.0007 | 0.0010 | 0.8364 | true |
| **selected prototype** | **0.04** | **0.29** | 0.012759 | 0.9983 | 0.0007 | 0.0010 | 0.8364 | true |

Selection rule: min expected cost on validation under non-approve rate <= 5%

## 10. Selected prototype operating thresholds

Label: **PROTOTYPE CALIBRATION** (Scenario A).

- APPROVE if calibrated p < **0.02**
- REVIEW if **0.02** ≤ p < **0.29**
- BLOCK if p ≥ **0.29**

Cost assumptions: FN=100.0,
FP=5.0,
review=2.0.

Source: min expected cost on validation under non-approve rate <= 5%

Validation mix: approve=0.9975,
review=0.0015,
block=0.0010,
expected cost/txn=0.021805.

These are **not** `THRESHOLD_REVIEW=40` / `THRESHOLD_BLOCK=70` and must not be copied
onto the synthetic product `final_risk_score` without a separate product decision.

## 11. Test evaluation (once, selected calibrator)

Raw vs calibrated probability on the untouched chronological test set:

| quantity | raw probability | calibrated probability (`isotonic`) | risk score |
| --- | --- | --- | --- |
| definition | uncalibrated booster P(Class=1) | validation-fitted map | calibrated p × 100 |
| is a probability? | uncalibrated estimate | calibrated estimate | **no** |
| Brier | 0.00237701 | 0.00044507 | n/a |
| log loss | 0.014995 | 0.004117 | n/a |
| ECE uniform 10 | 0.011500 | 0.000274 | n/a |

Threshold-free ranking on **raw** test probabilities: PR-AUC=0.757884,
ROC-AUC=0.982964.

Threshold-free ranking on **calibrated** test probabilities: PR-AUC=0.719148,
ROC-AUC=0.982079.
A drop in PR-AUC after isotonic is ranking compression, not a reason to quietly switch methods using test.

Binary metrics at 0.5 (reference only):

| precision | recall | F1 | FPR | FNR | Brier | log loss |
| --- | --- | --- | --- | --- | --- | --- |
| 0.8605 | 0.7115 | 0.7789 | 0.000141 | 0.2885 | 0.00044507 | 0.004117 |

Confusion at 0.5: TP=37 FP=6
TN=42501 FN=15.

Binary metrics at prototype **T_BLOCK** (positive = BLOCK):

| precision | recall | F1 | FPR | FNR |
| --- | --- | --- | --- | --- |
| 0.8636 | 0.7308 | 0.7917 | 0.000141 | 0.2692 |

Three-way decisions on test (frozen validation thresholds):

- APPROVE / REVIEW / BLOCK counts: {'APPROVE': 42461, 'REVIEW': 54, 'BLOCK': 44}
- expected cost / txn (Scenario A): 0.033788
- approve rate: 0.9977
- review rate: 0.0013
- block rate: 0.0010
- approved fraud (FN-style): 13
- blocked legit (FP-style): 6
- fraud catch rate (review or block): 0.7500

## Model vs risk score

| signal | meaning |
| --- | --- |
| model probability | calibrated ULB P(Class=1) — offline only |
| behavior score | anomaly / personalized overlays on synthetic traffic |
| rule score | deterministic rule evidence |
| graph score | shared device/IP relationship risk |
| final risk score | weighted RazorGuard decision signal, **not** P(fraud) |

## Limitations

- ULB has no user/device/IP/merchant fields; calibrated P(Class=1) is not a production fraud probability.
- Isotonic validation ECE near zero is in-sample: the map is fit on those same labels.
- Validation has few fraud cases (55); isotonic can overfit and ECE is noisy.
- Isotonic may collapse ranking (fewer unique scores) even when Brier improves.
- Brier score is dominated by the negative class at ~0.12% prevalence.
- Cost weights are prototype configuration, not empirically estimated loss given default.
- A 5% non-approve cap is an operations constraint, not a statistically identified constant.
- Product APPROVE/REVIEW/BLOCK still uses env THRESHOLD_REVIEW/BLOCK on final_risk_score.
- xgb-iforest-v1-calibrated remains the live model; ulb-xgb-v1-calibrated is offline only.

This report does not claim production fraud-detection performance.
