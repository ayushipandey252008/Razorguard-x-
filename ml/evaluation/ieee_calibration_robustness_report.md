# IEEE-CIS calibration robustness audit

Phase 9.1 audit. This is **not** a new model-development run.

The chronological IEEE-CIS **TEST split was not used** to fit calibrators, choose a method,
or set thresholds. XGBoost was **not** retrained. The live model and ULB artifacts were
left unchanged. Candidates remain OFFLINE CANDIDATE.

Phase 9 selected isotonic from in-sample validation diagnostics. That point estimate is
**not** treated as decisive here.

**Decision:** `INCONCLUSIVE_KEEP_CURRENT`

**Operating calibration after this audit:** `isotonic (current, inconclusive)`

Test used for decision: False

## Methodology

- Frozen booster: `ieee-xgb-combined-v1` (`CANDIDATE`, not live).
- Raw scores: saved preprocessor + classifier applied to reconstructed train/validation
  features. Behavioral and graph features for pretest rows use only time < T history.
- Calibrators (raw / sigmoid-Platt / isotonic) are fit on **validation or earlier pretest
  slices only**.
- Nested stratified holdout and 5-fold OOF are on **validation scores**.
- Extra temporal holdouts use **train+validation** with a different time cut. TEST unused.
- Distinctions: **calibration quality** (Brier, log loss, ECE), **ranking quality**
  (PR-AUC / ROC-AUC, tie analysis), **threshold operating performance** (APPROVE/REVIEW/BLOCK
  on validation).
- XGBoost retrained: False
- Frozen TEST scored for this decision: False
- `IEEE_MAX_ROWS` used: False
- Fixture used: False

Booster scores: train n=413378 (14538 positives),
validation n=88581 (3042 positives).

## Split (unchanged; test not scored)

| split | n | fraud | prevalence |
| --- | --- | --- | --- |
| train | 413378 | 14538 | 0.035169 |
| validation | 88581 | 3042 | 0.034341 |
| test (not scored) | 88581 | 3083 | 0.034804 |

Split matches Phase 9 manifest: True

## AUDIT A — In-sample validation calibration

Isotonic and sigmoid are fit on all validation rows, then scored on those same rows.
Isotonic is expected to look best on Brier/ECE here. This is **not** the robustness decision.

| method | brier | log_loss | ece_uniform_10 | pr_auc | n_unique_predictions | mean_predicted | empirical_prevalence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.138000 | 0.432477 | 0.289238 | 0.429994 | 85435 | 0.323579 | 0.034341 |
| sigmoid | 0.025448 | 0.103504 | 0.005312 | 0.429994 | 85435 | 0.034382 | 0.034341 |
| isotonic | 0.024585 | 0.100410 | 0.000000 | 0.422379 | 68 | 0.034341 | 0.034341 |

Observed fraud prevalence: 0.034341.
Mean predicted: raw=0.323579,
sigmoid=0.034382,
isotonic=0.034341.

Unique predicted probabilities: raw=85435,
sigmoid=85435,
isotonic=68.

### Reliability bins (uniform 10, in-sample validation)

**raw**

| bin | lo | hi | n | positives | mean predicted | empirical rate | |gap| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.100 | 15898 | 5 | 0.050143 | 0.000315 | 0.049829 |
| 1 | 0.100 | 0.200 | 12702 | 57 | 0.151599 | 0.004487 | 0.147111 |
| 2 | 0.200 | 0.300 | 15955 | 158 | 0.250827 | 0.009903 | 0.240925 |
| 3 | 0.300 | 0.400 | 14955 | 243 | 0.348288 | 0.016249 | 0.332039 |
| 4 | 0.400 | 0.500 | 11025 | 240 | 0.447015 | 0.021769 | 0.425246 |
| 5 | 0.500 | 0.600 | 7924 | 337 | 0.546845 | 0.042529 | 0.504316 |
| 6 | 0.600 | 0.700 | 4748 | 354 | 0.645050 | 0.074558 | 0.570493 |
| 7 | 0.700 | 0.800 | 2677 | 382 | 0.745526 | 0.142697 | 0.602829 |
| 8 | 0.800 | 0.900 | 1481 | 428 | 0.845714 | 0.288994 | 0.556720 |
| 9 | 0.900 | 1.000 | 1216 | 838 | 0.951537 | 0.689145 | 0.262393 |

**sigmoid / Platt**

| bin | lo | hi | n | positives | mean predicted | empirical rate | |gap| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.100 | 81222 | 1220 | 0.014855 | 0.015021 | 0.000166 |
| 1 | 0.100 | 0.200 | 3758 | 379 | 0.140672 | 0.100852 | 0.039820 |
| 2 | 0.200 | 0.300 | 1431 | 297 | 0.243744 | 0.207547 | 0.036197 |
| 3 | 0.300 | 0.400 | 849 | 258 | 0.345855 | 0.303887 | 0.041969 |
| 4 | 0.400 | 0.500 | 573 | 282 | 0.450399 | 0.492147 | 0.041747 |
| 5 | 0.500 | 0.600 | 748 | 606 | 0.547980 | 0.810160 | 0.262181 |
| 6 | 0.600 | 0.700 | 0 | 0 | — | — | — |
| 7 | 0.700 | 0.800 | 0 | 0 | — | — | — |
| 8 | 0.800 | 0.900 | 0 | 0 | — | — | — |
| 9 | 0.900 | 1.000 | 0 | 0 | — | — | — |

**isotonic**

| bin | lo | hi | n | positives | mean predicted | empirical rate | |gap| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.100 | 83454 | 1414 | 0.016943 | 0.016943 | 0.000000 |
| 1 | 0.100 | 0.200 | 1818 | 231 | 0.127063 | 0.127063 | 0.000000 |
| 2 | 0.200 | 0.300 | 1693 | 406 | 0.239811 | 0.239811 | 0.000000 |
| 3 | 0.300 | 0.400 | 300 | 103 | 0.343333 | 0.343333 | 0.000000 |
| 4 | 0.400 | 0.500 | 381 | 168 | 0.440945 | 0.440945 | 0.000000 |
| 5 | 0.500 | 0.600 | 78 | 41 | 0.525641 | 0.525641 | 0.000000 |
| 6 | 0.600 | 0.700 | 192 | 125 | 0.651042 | 0.651042 | 0.000000 |
| 7 | 0.700 | 0.800 | 242 | 178 | 0.735537 | 0.735537 | 0.000000 |
| 8 | 0.800 | 0.900 | 303 | 262 | 0.864686 | 0.864686 | 0.000000 |
| 9 | 0.900 | 1.000 | 120 | 114 | 0.950000 | 0.950000 | 0.000000 |

## AUDIT B — Robustness (validation / OOF / train-holdout)

### Frozen-map bootstrap (maps not refit)

n_boot=200, stratified=true, seed=42.
This is sampling noise of a fixed map, not calibrator generalization.

**brier**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.137991 | 0.000512 | 0.137067 | 0.137985 | 0.139020 |
| sigmoid | 0.025439 | 0.000193 | 0.025125 | 0.025438 | 0.025796 |
| isotonic | 0.024573 | 0.000238 | 0.024155 | 0.024580 | 0.025023 |

**log_loss**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.432448 | 0.001286 | 0.430259 | 0.432460 | 0.434964 |
| sigmoid | 0.103444 | 0.000900 | 0.101798 | 0.103421 | 0.105112 |
| isotonic | 0.100355 | 0.000927 | 0.098673 | 0.100393 | 0.102140 |

**ece_uniform_10**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.289253 | 0.000666 | 0.287919 | 0.289206 | 0.290441 |
| sigmoid | 0.005435 | 0.000384 | 0.004686 | 0.005451 | 0.006152 |
| isotonic | 0.000897 | 0.000252 | 0.000500 | 0.000864 | 0.001419 |

**pr_auc**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.430447 | 0.009242 | 0.412739 | 0.429704 | 0.446914 |
| sigmoid | 0.430447 | 0.009242 | 0.412739 | 0.429704 | 0.446914 |
| isotonic | 0.422801 | 0.008974 | 0.405949 | 0.422357 | 0.438754 |

Paired isotonic − sigmoid Brier: mean=-0.00086575,
95% [-0.00098341,
-0.00073111].

### Nested validation holdout (calibrators refit)

Calibrators are refit on each validation subset and scored on the held-out subset. Frozen chronological TEST is unused.

Splits used: 40 / 40
(eval fraction 0.3, skipped 0).

**brier**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.137966 | 0.000672 | 0.136645 | 0.137998 | 0.139004 |
| sigmoid | 0.025477 | 0.000247 | 0.025121 | 0.025448 | 0.025899 |
| isotonic | 0.024773 | 0.000328 | 0.024229 | 0.024841 | 0.025415 |

**log_loss**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.432417 | 0.001606 | 0.429450 | 0.432621 | 0.434583 |
| sigmoid | 0.103618 | 0.001458 | 0.101449 | 0.103463 | 0.106023 |
| isotonic | 0.102306 | 0.001788 | 0.099668 | 0.102203 | 0.105079 |

**ece_uniform_10**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.289219 | 0.000895 | 0.287612 | 0.289151 | 0.290737 |
| sigmoid | 0.005462 | 0.000476 | 0.004734 | 0.005470 | 0.006271 |
| isotonic | 0.001829 | 0.000578 | 0.001014 | 0.001733 | 0.003009 |

**pr_auc**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.430035 | 0.012829 | 0.406964 | 0.429002 | 0.449851 |
| sigmoid | 0.430035 | 0.012829 | 0.406964 | 0.429002 | 0.449851 |
| isotonic | 0.415138 | 0.012770 | 0.392519 | 0.413102 | 0.433683 |

**n_unique**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 26265.325000 | 15.152875 | 26239.775000 | 26263.500000 | 26291.250000 |
| sigmoid | 26265.325000 | 15.152875 | 26239.775000 | 26263.500000 | 26291.250000 |
| isotonic | 83.300000 | 7.289613 | 69.975000 | 84.000000 | 99.050000 |

Win rates (fraction of repeats):
- isotonic Brier < sigmoid: 1.000
- isotonic Brier < raw: 1.000
- sigmoid Brier < raw: 1.000
- isotonic PR-AUC ≥ raw: 0.000
- sigmoid PR-AUC ≥ raw: 1.000

Paired isotonic − sigmoid Brier: mean=-0.00070351,
95% [-0.00089403,
-0.00048671].

Paired isotonic − raw PR-AUC: mean=-0.014896.

### 5-fold OOF calibration on validation

Each validation row is scored by a calibrator that did not see that row. Frozen TEST unused.

Folds: 5.

**brier**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.138000 | 0.001049 | 0.136950 | 0.137900 | 0.139267 |
| sigmoid | 0.025460 | 0.000567 | 0.024758 | 0.025459 | 0.026028 |
| isotonic | 0.024786 | 0.000746 | 0.023822 | 0.024855 | 0.025543 |

**log_loss**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.432477 | 0.002634 | 0.429421 | 0.432604 | 0.435703 |
| sigmoid | 0.103540 | 0.003107 | 0.100583 | 0.102144 | 0.106973 |
| isotonic | 0.101994 | 0.002738 | 0.099122 | 0.102273 | 0.104690 |

**ece_uniform_10**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.289238 | 0.001321 | 0.287834 | 0.288874 | 0.291022 |
| sigmoid | 0.005882 | 0.000906 | 0.004574 | 0.006210 | 0.006769 |
| isotonic | 0.002544 | 0.000396 | 0.002221 | 0.002437 | 0.003124 |

**pr_auc**

| method | mean | std | p2.5 | p50 | p97.5 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.430582 | 0.028798 | 0.401443 | 0.425239 | 0.466078 |
| sigmoid | 0.430582 | 0.028798 | 0.401443 | 0.425239 | 0.466078 |
| isotonic | 0.415897 | 0.026278 | 0.389227 | 0.412755 | 0.449281 |

**Pooled OOF**

| method | brier | log_loss | ece_uniform_10 | pr_auc | n_unique_predictions |
| --- | --- | --- | --- | --- | --- |
| raw | 0.138000 | 0.432477 | 0.289238 | 0.429994 | 85435 |
| sigmoid | 0.025460 | 0.103540 | 0.005208 | 0.428297 | 87886 |
| isotonic | 0.024786 | 0.101994 | 0.000561 | 0.421282 | 375 |

### OOF reliability bins

**raw**

| bin | lo | hi | n | positives | mean predicted | empirical rate | |gap| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.100 | 15898 | 5 | 0.050143 | 0.000315 | 0.049829 |
| 1 | 0.100 | 0.200 | 12702 | 57 | 0.151599 | 0.004487 | 0.147111 |
| 2 | 0.200 | 0.300 | 15955 | 158 | 0.250827 | 0.009903 | 0.240925 |
| 3 | 0.300 | 0.400 | 14955 | 243 | 0.348288 | 0.016249 | 0.332039 |
| 4 | 0.400 | 0.500 | 11025 | 240 | 0.447015 | 0.021769 | 0.425246 |
| 5 | 0.500 | 0.600 | 7924 | 337 | 0.546845 | 0.042529 | 0.504316 |
| 6 | 0.600 | 0.700 | 4748 | 354 | 0.645050 | 0.074558 | 0.570493 |
| 7 | 0.700 | 0.800 | 2677 | 382 | 0.745526 | 0.142697 | 0.602829 |
| 8 | 0.800 | 0.900 | 1481 | 428 | 0.845714 | 0.288994 | 0.556720 |
| 9 | 0.900 | 1.000 | 1216 | 838 | 0.951537 | 0.689145 | 0.262393 |

**sigmoid**

| bin | lo | hi | n | positives | mean predicted | empirical rate | |gap| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.100 | 81221 | 1220 | 0.014898 | 0.015021 | 0.000123 |
| 1 | 0.100 | 0.200 | 3767 | 378 | 0.140634 | 0.100345 | 0.040289 |
| 2 | 0.200 | 0.300 | 1439 | 304 | 0.244145 | 0.211258 | 0.032887 |
| 3 | 0.300 | 0.400 | 836 | 256 | 0.346133 | 0.306220 | 0.039913 |
| 4 | 0.400 | 0.500 | 588 | 295 | 0.451071 | 0.501701 | 0.050630 |
| 5 | 0.500 | 0.600 | 729 | 588 | 0.547765 | 0.806584 | 0.258820 |
| 6 | 0.600 | 0.700 | 1 | 1 | 0.601510 | 1.000000 | 0.398490 |
| 7 | 0.700 | 0.800 | 0 | 0 | — | — | — |
| 8 | 0.800 | 0.900 | 0 | 0 | — | — | — |
| 9 | 0.900 | 1.000 | 0 | 0 | — | — | — |

**isotonic**

| bin | lo | hi | n | positives | mean predicted | empirical rate | |gap| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.100 | 83582 | 1435 | 0.017050 | 0.017169 | 0.000119 |
| 1 | 0.100 | 0.200 | 1829 | 247 | 0.134670 | 0.135046 | 0.000376 |
| 2 | 0.200 | 0.300 | 1408 | 341 | 0.236409 | 0.242188 | 0.005779 |
| 3 | 0.300 | 0.400 | 471 | 151 | 0.337947 | 0.320594 | 0.017353 |
| 4 | 0.400 | 0.500 | 375 | 165 | 0.444153 | 0.440000 | 0.004153 |
| 5 | 0.500 | 0.600 | 57 | 35 | 0.547578 | 0.614035 | 0.066457 |
| 6 | 0.600 | 0.700 | 227 | 144 | 0.657869 | 0.634361 | 0.023507 |
| 7 | 0.700 | 0.800 | 207 | 147 | 0.742757 | 0.710145 | 0.032612 |
| 8 | 0.800 | 0.900 | 302 | 263 | 0.862111 | 0.870861 | 0.008750 |
| 9 | 0.900 | 1.000 | 123 | 114 | 0.948918 | 0.926829 | 0.022088 |

### Train-fit / validation-eval

Fit calibrators on **train** booster scores (14538 positives),
evaluate on **validation**. Train scores are booster-in-sample.

| method | brier | log_loss | ece_uniform_10 | pr_auc | n_unique_predictions |
| --- | --- | --- | --- | --- | --- |
| raw | 0.138000 | 0.432477 | 0.289238 | 0.429994 | 85435 |
| sigmoid | 0.025482 | 0.104082 | 0.007424 | 0.429994 | 85435 |
| isotonic | 0.024927 | 0.101786 | 0.003591 | 0.422507 | 149 |

Pre-test only. Raw train scores are booster-in-sample. Frozen TEST unused.

## AUDIT C — Additional temporal pretest holdouts

Only pre-test rows (official train + validation). Different time cuts from the official
70/15 validation boundary. TEST unused.

**Holdout 1** cut_frac=0.5, fit n=250979 (8174 fraud), eval n=250980 (9406 fraud), fit_time_max=5973383 < eval_time_min=5973411, lowest Brier=`isotonic`.

| method | brier | log_loss | ece_uniform_10 | pr_auc | n_unique_predictions |
| --- | --- | --- | --- | --- | --- |
| raw | 0.135907 | 0.426942 | 0.285492 | 0.505025 | 239715 |
| sigmoid | 0.025853 | 0.104093 | 0.010407 | 0.505025 | 239715 |
| isotonic | 0.024955 | 0.101175 | 0.004922 | 0.494329 | 222 |
**Holdout 2** cut_frac=0.75, fit n=376469 (12837 fraud), eval n=125490 (4743 fraud), fit_time_max=9409262 < eval_time_min=9409267, lowest Brier=`isotonic`.

| method | brier | log_loss | ece_uniform_10 | pr_auc | n_unique_predictions |
| --- | --- | --- | --- | --- | --- |
| raw | 0.140399 | 0.438094 | 0.291084 | 0.477764 | 120431 |
| sigmoid | 0.026747 | 0.107583 | 0.008263 | 0.477764 | 120431 |
| isotonic | 0.026036 | 0.104887 | 0.003521 | 0.469151 | 160 |

Winner changes across temporal holdouts: False
Temporal conflicts with isotonic: False

## AUDIT D — Isotonic staircase / 68-level concern

In-sample isotonic vs raw on validation:

- monotone in raw score: True
- unique probabilities: 85435 → 68
- unique ratio: 0.000796
- max plateau n: 9843
- median plateau n: 300.0
- PR-AUC raw: 0.429994
- PR-AUC isotonic: 0.422379
- PR-AUC isotonic + raw-order tie-break: 0.429996
- drop explained by ties: True

Isotonic is monotone in the raw score, so it cannot reverse pairs. A PR-AUC drop that disappears after breaking ties with the raw order is staircase compression, not a rank reversal.

68 distinct isotonic levels (Phase 9 in-sample) is above the uniqueness-guard cutoff of 20,
but it is still a coarse staircase relative to sigmoid (~85k unique values). Whether that
is harmful is decided by nested/OOF ranking and Brier, not by the unique-count alone.

## AUDIT E — Historical frozen chronological TEST (quoted, not recomputed)

These numbers are copied from the Phase 9 IEEE result. They were **not** overwritten and
were **not** used to choose a calibrator.

Selected calibrator at the time: isotonic. Threshold for the binary metrics below: 0.5.

| metric | value |
| --- | --- |
| PR-AUC | 0.337152 |
| ROC-AUC | 0.868842 |
| Precision | 0.393396 |
| Recall | 0.270516 |
| F1 | 0.320584 |
| FPR | 0.01504129 |
| FNR | 0.729484 |
| TP / FP / FN / TN | 834 / 1286 / 2249 / 84212 |

Historical Phase 9 chronological TEST result. Quoted only. This audit did not overwrite ieee_results.json and did not use TEST to select a calibrator.

No new frozen-test metric is reported from this audit.

## AUDIT F — Thresholds (validation only)

Phase 9 validation-selected policy (isotonic val scores):
APPROVE < 0.020004,
REVIEW 0.020004–0.511628,
BLOCK > 0.511628.

Existing policy on **isotonic** validation scores:

- APPROVE/REVIEW/BLOCK counts: 66208 / 21501 / 872
- fraud catch (REVIEW or BLOCK): 0.806049
- val cost (prototype units): 102922.0
- cutpoints: APPROVE < 0.020004, BLOCK > 0.511628

Existing policy on **sigmoid** validation scores (same cutpoints, different map):

- APPROVE/REVIEW/BLOCK counts: 62700 / 25212 / 669
- fraud catch (REVIEW or BLOCK): 0.827416
- val cost (prototype units): 103494.0
- cutpoints: APPROVE < 0.020004, BLOCK > 0.511628

Re-selected three-way policy on validation for the audit operating method
(`isotonic`), still validation-only:

- APPROVE/REVIEW/BLOCK counts: 66208 / 21501 / 872
- fraud catch (REVIEW or BLOCK): 0.806049
- val cost (prototype units): 102922.0
- cutpoints: APPROVE < 0.020004, BLOCK > 0.511628

Phase 9 cutpoints were chosen on isotonic validation scores. This review does not search TEST. If the decision is KEEP_ISOTONIC or INCONCLUSIVE_KEEP_CURRENT, the published 0.020004 / 0.511628 cutpoints remain.

Thresholds were not optimized on TEST.

## Conclusion

**`INCONCLUSIVE_KEEP_CURRENT`**

### Evidence

- In-sample validation Brier: raw=0.138000, sigmoid=0.025448, isotonic=0.024585 (isotonic unique p=68; in-sample advantage).
- Nested val holdout mean Brier: raw=0.13796598608087987, sigmoid=0.025476760210600018, isotonic=0.024773247329344356; isotonic<sigmoid in 100% of repeats.
- Nested mean PR-AUC: raw=0.43003457139348955, sigmoid=0.43003457139348955, isotonic=0.41513813472485117.
- 5-fold OOF pooled Brier raw=0.1379999336135729, sigmoid=0.025459976178104554, isotonic=0.02478615687078338; PR-AUC raw=0.42999390635178036, sigmoid=0.4282974657103745, isotonic=0.42128151537990277; isotonic unique p=375.
- Train-fit/val-eval Brier isotonic=0.024927 sigmoid=0.025482; PR-AUC raw=0.4300 sigmoid=0.4300 isotonic=0.4225.
- Staircase: unique 85435→68 ratio=0.0007959267279218119, monotone=True, PR-AUC drop=0.007614995859687113, after tie-break=-2.1920672563791044e-06.
- Temporal pretest holdout lowest-Brier methods: ['isotonic', 'isotonic'].

### Rationale

- Nested Brier prefers isotonic; ranking/uniqueness prefer sigmoid. Evidence is mixed, so the current isotonic selection is kept.
- In-sample PR-AUC drop is explained by plateau ties, not rank reversal.
- Sigmoid nested PR-AUC matches raw (monotone map).

Live model remains `xgb-iforest-v1-calibrated`. IEEE candidates remain OFFLINE CANDIDATE.
This is not production payment-fraud accuracy.

## Integrity

Protected artifacts unchanged: True
XGBoost retrained: False
Frozen TEST touched: False
New calibration artifact written: False

## Limitations

- Train booster scores are in-sample for XGBoost; train-fit/val-eval is not fully clean CV.
- Nested/OOF still use the official validation time window, not a new test.
- Temporal pretest cuts share the same frozen booster (trained with the official val as
  early stopping / selection context in Phase 9). They are not independent retrains.
- In-sample isotonic ECE near 0 is expected when the map is fit and scored on the same rows.
- 68 isotonic levels can be either a useful piecewise-constant reliability map or harmful
  tie compression; nested PR-AUC vs Brier is the check, not unique-count by itself.
- Product `final_risk_score` is a separate live weighted score. These probabilities are
  offline IEEE candidate scores only.
