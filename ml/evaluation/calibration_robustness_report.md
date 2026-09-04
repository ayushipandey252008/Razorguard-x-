# Calibration robustness audit

This audit does **not** use chronological test labels for fitting, selection, or the
recommendation. The official test slice was not scored. Phase 2 files
(`calibration_metrics.json`, `calibration_report.md`, `ulb_metrics.json`) were left intact.

The Phase 2 point-estimate choice was isotonic because it had the lowest in-sample
validation Brier. That is not treated as decisive here.

## Split (unchanged)

| split | n | fraud | prevalence |
| --- | --- | --- | --- |
| train | 198608 | 366 | 0.001843 |
| validation | 42559 | 55 | 0.001292 |
| test (not scored) | 42559 | 52 | 0.001222 |

`no_future_in_train`: True
Matches `ulb_metrics.json`: True

Booster scores: train n=198608 (366 positives),
validation n=42559 (55 positives).

## 1. In-sample validation comparison (fit and score on the same 55 fraud cases)

Isotonic and sigmoid are fit on all validation rows, then scored on those same rows.
Isotonic is expected to look best here.

| method | brier | log_loss | ece_uniform_10 | pr_auc | n_unique_predictions |
| --- | --- | --- | --- | --- | --- |
| raw | 0.002363 | 0.014619 | 0.011792 | 0.839099 | 40677 |
| sigmoid | 0.000495 | 0.002706 | 0.000696 | 0.839099 | 40674 |
| isotonic | 0.000304 | 0.001783 | 0.000000 | 0.837019 | 11 |

Mean predicted vs prevalence (raw 0.013085 vs
0.001292).

### Calibration curves (uniform 10 bins, in-sample)

**raw**

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

**sigmoid**

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

**isotonic**

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

Figures: `ml/evaluation/figures/robustness_val_insample_{raw,sigmoid,isotonic}.svg`.

## 2. Uncertainty of frozen-map metrics (stratified bootstrap of validation)

The maps are **not** refit. This interval is sampling noise of Brier / log loss / ECE / PR-AUC
for a fixed map, not a test of whether isotonic overfits.

n_boot=400, stratified=true, seed=42.

**brier**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.002355 | 0.000134 | 0.002112 | 0.002350 | 0.002640 |
| sigmoid | 0.000491 | 0.000053 | 0.000402 | 0.000489 | 0.000595 |
| isotonic | 0.000297 | 0.000063 | 0.000193 | 0.000293 | 0.000427 |
**log_loss**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.014598 | 0.000421 | 0.013798 | 0.014594 | 0.015473 |
| sigmoid | 0.002684 | 0.000385 | 0.002042 | 0.002659 | 0.003563 |
| isotonic | 0.001753 | 0.000376 | 0.001114 | 0.001726 | 0.002575 |
**ece_uniform_10**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.011788 | 0.000220 | 0.011375 | 0.011788 | 0.012238 |
| sigmoid | 0.000723 | 0.000097 | 0.000521 | 0.000728 | 0.000901 |
| isotonic | 0.000094 | 0.000045 | 0.000020 | 0.000091 | 0.000186 |
**pr_auc**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.842509 | 0.042732 | 0.746192 | 0.845938 | 0.910496 |
| sigmoid | 0.842509 | 0.042732 | 0.746192 | 0.845938 | 0.910496 |
| isotonic | 0.840436 | 0.043045 | 0.742764 | 0.843355 | 0.909619 |

Paired isotonic − sigmoid Brier: mean=-0.00019355,
95% [-0.00023699,
-0.00014902].

Paired isotonic − raw PR-AUC: mean=-0.002073,
95% [-0.005723,
-0.000088].

## 3. Nested validation holdout (calibrators refit)

Calibrators are refit on each validation subset and scored on the held-out subset. This removes the in-sample advantage isotonic has when fit and scored on the same 55 positives.

Splits used: 40 / 40
(eval fraction 0.3, skipped 0).

**brier**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.002359 | 0.000220 | 0.002012 | 0.002381 | 0.002737 |
| sigmoid | 0.000532 | 0.000073 | 0.000412 | 0.000533 | 0.000665 |
| isotonic | 0.000353 | 0.000105 | 0.000200 | 0.000355 | 0.000532 |
**log_loss**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.014600 | 0.000713 | 0.013372 | 0.014531 | 0.015859 |
| sigmoid | 0.002865 | 0.000547 | 0.001894 | 0.002874 | 0.003996 |
| isotonic | 0.002957 | 0.002021 | 0.001010 | 0.002455 | 0.007523 |
**ece_uniform_10**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.011761 | 0.000295 | 0.011236 | 0.011705 | 0.012349 |
| sigmoid | 0.000840 | 0.000203 | 0.000513 | 0.000825 | 0.001178 |
| isotonic | 0.000212 | 0.000110 | 0.000050 | 0.000216 | 0.000457 |
**pr_auc**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.841352 | 0.066730 | 0.717421 | 0.843474 | 0.959138 |
| sigmoid | 0.841352 | 0.066730 | 0.717421 | 0.843474 | 0.959138 |
| isotonic | 0.814212 | 0.072061 | 0.664376 | 0.820321 | 0.927090 |
**n_unique**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 12510.750000 | 12.029344 | 12488.950000 | 12514.000000 | 12534.025000 |
| sigmoid | 12510.575000 | 12.127290 | 12488.925000 | 12514.000000 | 12534.025000 |
| isotonic | 11.750000 | 2.519157 | 8.000000 | 11.500000 | 17.025000 |

Win rates (fraction of repeats):
- isotonic Brier < sigmoid: 1.000
- isotonic Brier < raw: 1.000
- sigmoid Brier < raw: 1.000
- isotonic PR-AUC ≥ raw: 0.000
- sigmoid PR-AUC ≥ raw: 1.000

Paired isotonic − sigmoid Brier: mean=-0.00017863,
95% [-0.00024162,
-0.00011636].

Paired isotonic − raw PR-AUC: mean=-0.027140,
95% [-0.053045,
-0.003181].

## 4. K-fold out-of-fold calibration on validation

Each validation row is scored by a calibrator that did not see that row. Pooled OOF Brier/log loss/ECE/PR-AUC are the primary CV estimates.

Folds: 5.

**brier (per-fold)**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.002363 | 0.000380 | 0.001917 | 0.002269 | 0.002799 |
| sigmoid | 0.000502 | 0.000198 | 0.000280 | 0.000508 | 0.000747 |
| isotonic | 0.000388 | 0.000343 | 0.000034 | 0.000348 | 0.000874 |
**log_loss (per-fold)**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.014619 | 0.001162 | 0.013201 | 0.014346 | 0.015831 |
| sigmoid | 0.002761 | 0.001184 | 0.001455 | 0.003056 | 0.004056 |
| isotonic | 0.005101 | 0.006226 | 0.000394 | 0.002318 | 0.014423 |
**ece_uniform_10 (per-fold)**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.011794 | 0.000558 | 0.011025 | 0.011873 | 0.012393 |
| sigmoid | 0.000882 | 0.000340 | 0.000500 | 0.000816 | 0.001242 |
| isotonic | 0.000389 | 0.000279 | 0.000109 | 0.000339 | 0.000798 |
**pr_auc (per-fold)**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.831022 | 0.153540 | 0.682235 | 0.804712 | 0.998052 |
| sigmoid | 0.831022 | 0.153540 | 0.682235 | 0.804712 | 0.998052 |
| isotonic | 0.778631 | 0.220432 | 0.491816 | 0.791308 | 0.997159 |

**Pooled OOF** (concatenated held-out predictions):

| method | brier | log_loss | ece_uniform_10 | pr_auc | n_unique_predictions |
| --- | --- | --- | --- | --- | --- |
| raw | 0.002363 | 0.014619 | 0.011792 | 0.839099 | 40677 |
| sigmoid | 0.000502 | 0.002761 | 0.000677 | 0.800333 | 41909 |
| isotonic | 0.000388 | 0.005101 | 0.000159 | 0.769017 | 46 |

OOF reliability diagrams: `ml/evaluation/figures/robustness_val_oof_{raw,sigmoid,isotonic}.svg`.

### OOF calibration curves

**raw**

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

**sigmoid**

| bin | lo | hi | n | positives | mean predicted | empirical rate | |gap| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.100 | 42476 | 9 | 0.000436 | 0.000212 | 0.000225 |
| 1 | 0.100 | 0.200 | 21 | 1 | 0.132371 | 0.047619 | 0.084752 |
| 2 | 0.200 | 0.300 | 4 | 1 | 0.229307 | 0.250000 | 0.020693 |
| 3 | 0.300 | 0.400 | 8 | 1 | 0.348511 | 0.125000 | 0.223511 |
| 4 | 0.400 | 0.500 | 2 | 0 | 0.426072 | 0.000000 | 0.426072 |
| 5 | 0.500 | 0.600 | 34 | 29 | 0.571307 | 0.852941 | 0.281634 |
| 6 | 0.600 | 0.700 | 14 | 14 | 0.629274 | 1.000000 | 0.370726 |
| 7 | 0.700 | 0.800 | 0 | 0 | — | — | — |
| 8 | 0.800 | 0.900 | 0 | 0 | — | — | — |
| 9 | 0.900 | 1.000 | 0 | 0 | — | — | — |

**isotonic**

| bin | lo | hi | n | positives | mean predicted | empirical rate | |gap| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.100 | 42495 | 11 | 0.000224 | 0.000259 | 0.000035 |
| 1 | 0.100 | 0.200 | 15 | 2 | 0.138664 | 0.133333 | 0.005330 |
| 2 | 0.200 | 0.300 | 2 | 0 | 0.200000 | 0.000000 | 0.200000 |
| 3 | 0.300 | 0.400 | 1 | 1 | 0.369899 | 1.000000 | 0.630101 |
| 4 | 0.400 | 0.500 | 1 | 0 | 0.440408 | 0.000000 | 0.440408 |
| 5 | 0.500 | 0.600 | 1 | 0 | 0.508340 | 0.000000 | 0.508340 |
| 6 | 0.600 | 0.700 | 0 | 0 | — | — | — |
| 7 | 0.700 | 0.800 | 1 | 1 | 0.771097 | 1.000000 | 0.228903 |
| 8 | 0.800 | 0.900 | 0 | 0 | — | — | — |
| 9 | 0.900 | 1.000 | 43 | 40 | 1.000000 | 0.930233 | 0.069767 |

## 5. Staircase / ranking compression

In-sample isotonic vs raw on validation:

- monotone in raw score: True
- unique probabilities: 40677 → 11
- PR-AUC raw: 0.839099
- PR-AUC isotonic: 0.837019
- PR-AUC isotonic + raw-order tie-break: 0.839099
- drop explained by ties: True

Isotonic is monotone in the raw score, so it cannot reverse pairs. A PR-AUC drop that disappears after breaking ties with the raw order is staircase compression, not a rank reversal.

## 6. Cross-validation-style holdout without the test set

Fit calibrators on **train** booster scores (366 positives),
evaluate on **validation**. Train scores are booster-in-sample.

| method | brier | log_loss | ece_uniform_10 | pr_auc | n_unique_predictions |
| --- | --- | --- | --- | --- | --- |
| raw | 0.002363 | 0.014619 | 0.011792 | 0.839099 | 40677 |
| sigmoid | 0.000441 | 0.002515 | 0.000388 | 0.839099 | 40626 |
| isotonic | 0.000469 | 0.008811 | 0.000303 | 0.705833 | 9 |

Calibrator holdout uses more positives (train) than the 55 in validation. Raw train scores are booster-in-sample, so this is not a fully clean calibration CV.

## Recommendation (robustness, not lowest in-sample Brier)

**Recommended calibration method:** `sigmoid/Platt`

Inconclusive flag: True

### Evidence

- In-sample validation Brier: raw=0.002363, sigmoid=0.000495, isotonic=0.000304 (isotonic has an in-sample advantage; 11 unique p).
- Nested val holdout mean Brier: raw=0.002359, sigmoid=0.000532, isotonic=0.000353; isotonic better than sigmoid in 100% of repeats; paired isotonic-sigmoid Brier mean=-0.0001786296096974757, 95% interval [-0.00024161653072603687, -0.00011635533184498395].
- Nested mean PR-AUC: raw=0.8414, sigmoid=0.8414, isotonic=0.8142.
- 5-fold OOF pooled: Brier raw=0.0023626101488038757, sigmoid=0.0005023133670922522, isotonic=0.0003883073190954092; PR-AUC raw=0.8390992199622515, sigmoid=0.8003329015812657, isotonic=0.7690165842221484; isotonic unique p=46.
- Train-fit/val-eval Brier: raw=0.002363, sigmoid=0.000441, isotonic=0.000469; PR-AUC raw=0.8391, sigmoid=0.8391, isotonic=0.7058.
- Staircase: unique 40677→11, monotone=True, PR-AUC drop=0.002080407619329927, after raw-order tie-break drop=0.0 (explained_by_ties=True).

### Rationale

- Nested validation Brier favors isotonic in most repeats; that is a real result, not dismissed because the in-sample Brier was also lowest.
- Isotonic PR-AUC degradation is from plateau ties, not rank reversal. The calibrated score is still a weaker ranking statistic.
- Isotonic stays a coarse step function under resampling (nested mean unique p=11.75).
- When calibrators are fit on train scores and scored on validation, sigmoid has lower Brier than isotonic and does not collapse PR-AUC. That holdout has more positives than the 55-row validation nest.
- Sigmoid is a strictly monotone map (nested PR-AUC matches raw), reduces Brier vs raw, and avoids the staircase. It is the robustness pick when Brier and ranking disagree.
- Inconclusive as a single 'best' map: nested Brier prefers isotonic; ranking, uniqueness, and the train-fit/val-eval holdout prefer sigmoid. The recommended method is the conservative operational map, not the lowest nested Brier.

### Uncertainty

Validation has 55 positives, so ECE and even Brier have wide resampling intervals. Nested holdout used 40 splits (eval fraction 0.3). Frozen-map bootstrap intervals describe in-sample metric noise, not calibrator generalization. Train-fit/val-eval uses 366 train positives but booster-in-sample scores.

### Remaining limitation

The chronological test set was not used. A later one-shot test evaluation of a changed calibrator would still be a single draw at ~52 fraud cases. Product scoring is unchanged. Sigmoid/isotonic fitted on 55 val positives cannot be treated as a production probability of fraud.

---

Recommended calibration method: sigmoid/Platt

Evidence: In-sample validation Brier: raw=0.002363, sigmoid=0.000495, isotonic=0.000304 (isotonic has an in-sample advantage; 11 unique p).; Nested val holdout mean Brier: raw=0.002359, sigmoid=0.000532, isotonic=0.000353; isotonic better than sigmoid in 100% of repeats; paired isotonic-sigmoid Brier mean=-0.0001786296096974757, 95% interval [-0.00024161653072603687, -0.00011635533184498395].; Nested mean PR-AUC: raw=0.8414, sigmoid=0.8414, isotonic=0.8142.; 5-fold OOF pooled: Brier raw=0.0023626101488038757, sigmoid=0.0005023133670922522, isotonic=0.0003883073190954092; PR-AUC raw=0.8390992199622515, sigmoid=0.8003329015812657, isotonic=0.7690165842221484; isotonic unique p=46.; Train-fit/val-eval Brier: raw=0.002363, sigmoid=0.000441, isotonic=0.000469; PR-AUC raw=0.8391, sigmoid=0.8391, isotonic=0.7058.; Staircase: unique 40677→11, monotone=True, PR-AUC drop=0.002080407619329927, after raw-order tie-break drop=0.0 (explained_by_ties=True).

Uncertainty: Validation has 55 positives, so ECE and even Brier have wide resampling intervals. Nested holdout used 40 splits (eval fraction 0.3). Frozen-map bootstrap intervals describe in-sample metric noise, not calibrator generalization. Train-fit/val-eval uses 366 train positives but booster-in-sample scores.

Remaining limitation: The chronological test set was not used. A later one-shot test evaluation of a changed calibrator would still be a single draw at ~52 fraud cases. Product scoring is unchanged. Sigmoid/isotonic fitted on 55 val positives cannot be treated as a production probability of fraud.
