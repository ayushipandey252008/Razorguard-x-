"""Markdown report for the IEEE-CIS calibration robustness audit."""

from __future__ import annotations


def _fmt(value, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _sum_row(label: str, s: dict, digits: int = 6) -> str:
    return (
        f"| {label} | {_fmt(s.get('mean'), digits)} | {_fmt(s.get('std'), digits)} | "
        f"{_fmt(s.get('p025'), digits)} | {_fmt(s.get('p50'), digits)} | {_fmt(s.get('p975'), digits)} |"
    )


def _bundle_table(methods: dict, keys: tuple[str, ...]) -> str:
    header = "| method | " + " | ".join(keys) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(keys)) + " |"
    lines = [header, sep]
    for name in ("raw", "sigmoid", "isotonic"):
        row = methods[name]
        cells = " | ".join(_fmt(row.get(k), 6 if k != "n_unique_predictions" else 1) for k in keys)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines)


def _bin_table(rel: dict | None) -> str:
    if not rel:
        return "_no bins_"
    lines = [
        "| bin | lo | hi | n | positives | mean predicted | empirical rate | |gap| |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for b in rel.get("bins") or []:
        lines.append(
            "| {bin} | {lo} | {hi} | {count} | {n_pos} | {mp} | {emp} | {gap} |".format(
                bin=b.get("bin"),
                lo=_fmt(b.get("lo"), 3),
                hi=_fmt(b.get("hi"), 3),
                count=b.get("count"),
                n_pos=b.get("n_positive"),
                mp=_fmt(b.get("mean_predicted"), 6),
                emp=_fmt(b.get("empirical_positive_rate"), 6),
                gap=_fmt(b.get("gap"), 6),
            )
        )
    return "\n".join(lines)


def _metric_block(title: str, methods: dict, metric_names: tuple[str, ...]) -> str:
    chunks = []
    for metric in metric_names:
        lines = [
            f"**{metric}**",
            "",
            "| method | mean | std | p2.5 | p50 | p97.5 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name in ("raw", "sigmoid", "isotonic"):
            lines.append(_sum_row(name, methods[name][metric]))
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks)


def render_ieee_robustness_report(payload: dict) -> str:
    rec = payload["recommendation"]
    split = payload["official_split"]
    meth = payload["methodology"]
    ins = payload["in_sample_validation"]
    stair = payload["staircase_in_sample_validation"]
    boot = payload["bootstrap_frozen_maps"]
    nested = payload["nested_holdout_validation"]
    oof = payload["kfold_oof_validation"]
    th = payload["train_fit_val_eval"]
    hist = payload.get("historical_phase9_frozen_test") or {}
    hist_m = hist.get("metrics") or {}
    hist_cm = hist_m.get("confusion_matrix") or {}
    thr = payload.get("threshold_review") or {}
    existing = thr.get("existing_phase9_thresholds") or {}
    keys = ("brier", "log_loss", "ece_uniform_10", "pr_auc", "n_unique_predictions")
    oof_pool = oof.get("pooled_oof") or {}

    temporal_md = []
    for i, hold in enumerate(payload.get("temporal_pretest_holdouts") or [], 1):
        if hold.get("skipped"):
            temporal_md.append(f"**Holdout {i} skipped:** {hold.get('reason')}")
            continue
        temporal_md.append(
            f"**Holdout {i}** cut_frac={hold.get('cut_frac')}, "
            f"fit n={hold.get('fit_n')} ({hold.get('fit_positives')} fraud), "
            f"eval n={hold.get('eval_n')} ({hold.get('eval_positives')} fraud), "
            f"fit_time_max={hold.get('fit_time_max')} < eval_time_min={hold.get('eval_time_min')}, "
            f"lowest Brier=`{hold.get('lowest_brier_method')}`.\n\n"
            + _bundle_table(hold["methods"], keys)
        )

    oof_fold = []
    if oof.get("fold_summaries"):
        oof_fold.append(
            _metric_block("OOF folds", oof["fold_summaries"], ("brier", "log_loss", "ece_uniform_10", "pr_auc"))
        )

    evidence_md = "\n".join(f"- {item}" for item in rec.get("evidence") or [])
    rationale_md = "\n".join(f"- {item}" for item in rec.get("rationale") or [])
    integrity = payload.get("integrity") or {}
    checks = rec.get("checks") or {}

    return f"""# IEEE-CIS calibration robustness audit

Phase 9.1 audit. This is **not** a new model-development run.

The chronological IEEE-CIS **TEST split was not used** to fit calibrators, choose a method,
or set thresholds. XGBoost was **not** retrained. The live model and ULB artifacts were
left unchanged. Candidates remain OFFLINE CANDIDATE.

Phase 9 selected isotonic from in-sample validation diagnostics. That point estimate is
**not** treated as decisive here.

**Decision:** `{rec.get("decision")}`

**Operating calibration after this audit:** `{rec.get("recommended_calibration_method")}`

Test used for decision: {rec.get("test_used_for_decision")}

## Methodology

- Frozen booster: `{payload.get("booster_model_version")}` (`CANDIDATE`, not live).
- Raw scores: saved preprocessor + classifier applied to reconstructed train/validation
  features. Behavioral and graph features for pretest rows use only time < T history.
- Calibrators (raw / sigmoid-Platt / isotonic) are fit on **validation or earlier pretest
  slices only**.
- Nested stratified holdout and 5-fold OOF are on **validation scores**.
- Extra temporal holdouts use **train+validation** with a different time cut. TEST unused.
- Distinctions: **calibration quality** (Brier, log loss, ECE), **ranking quality**
  (PR-AUC / ROC-AUC, tie analysis), **threshold operating performance** (APPROVE/REVIEW/BLOCK
  on validation).
- XGBoost retrained: {meth.get("xgboost_refit")}
- Frozen TEST scored for this decision: {meth.get("test_scored")}
- `IEEE_MAX_ROWS` used: {meth.get("ieee_max_rows_used")}
- Fixture used: {meth.get("fixture_used")}

Booster scores: train n={meth.get("train_n")} ({meth.get("train_positives")} positives),
validation n={meth.get("validation_n")} ({meth.get("validation_positives")} positives).

## Split (unchanged; test not scored)

| split | n | fraud | prevalence |
| --- | --- | --- | --- |
| train | {split["train"]["n"]} | {split["train"]["fraud"]} | {_fmt(split["train"]["prevalence"], 6)} |
| validation | {split["validation"]["n"]} | {split["validation"]["fraud"]} | {_fmt(split["validation"]["prevalence"], 6)} |
| test (not scored) | {split["test"]["n"]} | {split["test"]["fraud"]} | {_fmt(split["test"]["prevalence"], 6)} |

Split matches Phase 9 manifest: {payload.get("split_matches_phase9_manifest")}

## AUDIT A — In-sample validation calibration

Isotonic and sigmoid are fit on all validation rows, then scored on those same rows.
Isotonic is expected to look best on Brier/ECE here. This is **not** the robustness decision.

{_bundle_table(ins, keys + ("mean_predicted", "empirical_prevalence"))}

Observed fraud prevalence: {_fmt(ins["raw"]["empirical_prevalence"], 6)}.
Mean predicted: raw={_fmt(ins["raw"]["mean_predicted"], 6)},
sigmoid={_fmt(ins["sigmoid"]["mean_predicted"], 6)},
isotonic={_fmt(ins["isotonic"]["mean_predicted"], 6)}.

Unique predicted probabilities: raw={ins["raw"]["n_unique_predictions"]},
sigmoid={ins["sigmoid"]["n_unique_predictions"]},
isotonic={ins["isotonic"]["n_unique_predictions"]}.

### Reliability bins (uniform 10, in-sample validation)

**raw**

{_bin_table(ins["raw"].get("reliability_uniform"))}

**sigmoid / Platt**

{_bin_table(ins["sigmoid"].get("reliability_uniform"))}

**isotonic**

{_bin_table(ins["isotonic"].get("reliability_uniform"))}

## AUDIT B — Robustness (validation / OOF / train-holdout)

### Frozen-map bootstrap (maps not refit)

n_boot={boot.get("n_boot")}, stratified=true, seed={boot.get("seed")}.
This is sampling noise of a fixed map, not calibrator generalization.

{_metric_block("bootstrap", boot["methods"], ("brier", "log_loss", "ece_uniform_10", "pr_auc"))}

Paired isotonic − sigmoid Brier: mean={_fmt(boot["paired_differences"]["brier_isotonic_minus_sigmoid"].get("mean"), 8)},
95% [{_fmt(boot["paired_differences"]["brier_isotonic_minus_sigmoid"].get("p025"), 8)},
{_fmt(boot["paired_differences"]["brier_isotonic_minus_sigmoid"].get("p975"), 8)}].

### Nested validation holdout (calibrators refit)

{nested.get("note")}

Splits used: {nested.get("n_splits_used")} / {nested.get("n_splits_requested")}
(eval fraction {nested.get("test_size")}, skipped {nested.get("n_splits_skipped")}).

{_metric_block("nested", nested["methods"], ("brier", "log_loss", "ece_uniform_10", "pr_auc", "n_unique"))}

Win rates (fraction of repeats):
- isotonic Brier < sigmoid: {_fmt(nested["win_rates"].get("isotonic_brier_lt_sigmoid"), 3)}
- isotonic Brier < raw: {_fmt(nested["win_rates"].get("isotonic_brier_lt_raw"), 3)}
- sigmoid Brier < raw: {_fmt(nested["win_rates"].get("sigmoid_brier_lt_raw"), 3)}
- isotonic PR-AUC ≥ raw: {_fmt(nested["win_rates"].get("isotonic_pr_auc_ge_raw"), 3)}
- sigmoid PR-AUC ≥ raw: {_fmt(nested["win_rates"].get("sigmoid_pr_auc_ge_raw"), 3)}

Paired isotonic − sigmoid Brier: mean={_fmt(nested["paired_differences"]["brier_isotonic_minus_sigmoid"].get("mean"), 8)},
95% [{_fmt(nested["paired_differences"]["brier_isotonic_minus_sigmoid"].get("p025"), 8)},
{_fmt(nested["paired_differences"]["brier_isotonic_minus_sigmoid"].get("p975"), 8)}].

Paired isotonic − raw PR-AUC: mean={_fmt(nested["paired_differences"]["pr_auc_isotonic_minus_raw"].get("mean"), 6)}.

### 5-fold OOF calibration on validation

{oof.get("note")}

Folds: {oof.get("n_splits")}.

{chr(10).join(oof_fold) if oof_fold else "_no fold summaries_"}

**Pooled OOF**

{_bundle_table(oof_pool, keys) if oof_pool else "_k-fold skipped_"}

### OOF reliability bins

**raw**

{_bin_table((oof_pool.get("raw") or {{}}).get("reliability_uniform"))}

**sigmoid**

{_bin_table((oof_pool.get("sigmoid") or {{}}).get("reliability_uniform"))}

**isotonic**

{_bin_table((oof_pool.get("isotonic") or {{}}).get("reliability_uniform"))}

### Train-fit / validation-eval

Fit calibrators on **train** booster scores ({th.get("train_positives")} positives),
evaluate on **validation**. Train scores are booster-in-sample.

{_bundle_table(th["methods"], keys)}

{th.get("note")}

## AUDIT C — Additional temporal pretest holdouts

Only pre-test rows (official train + validation). Different time cuts from the official
70/15 validation boundary. TEST unused.

{chr(10).join(temporal_md) if temporal_md else "_no temporal holdouts_"}

Winner changes across temporal holdouts: {checks.get("temporal_winner_changes")}
Temporal conflicts with isotonic: {checks.get("temporal_conflicts_with_isotonic")}

## AUDIT D — Isotonic staircase / 68-level concern

In-sample isotonic vs raw on validation:

- monotone in raw score: {stair.get("monotone_nondecreasing_in_raw")}
- unique probabilities: {stair.get("n_unique_raw")} → {stair.get("n_unique_calibrated")}
- unique ratio: {_fmt(stair.get("unique_ratio"), 6)}
- max plateau n: {(stair.get("plateaus") or {{}}).get("max_plateau_n")}
- median plateau n: {_fmt((stair.get("plateaus") or {{}}).get("median_plateau_n"), 1)}
- PR-AUC raw: {_fmt(stair.get("pr_auc_raw"), 6)}
- PR-AUC isotonic: {_fmt(stair.get("pr_auc_calibrated"), 6)}
- PR-AUC isotonic + raw-order tie-break: {_fmt(stair.get("pr_auc_calibrated_tiebroken_by_raw"), 6)}
- drop explained by ties: {stair.get("drop_explained_by_ties")}

{stair.get("note")}

68 distinct isotonic levels (Phase 9 in-sample) is above the uniqueness-guard cutoff of 20,
but it is still a coarse staircase relative to sigmoid (~85k unique values). Whether that
is harmful is decided by nested/OOF ranking and Brier, not by the unique-count alone.

## AUDIT E — Historical frozen chronological TEST (quoted, not recomputed)

These numbers are copied from the Phase 9 IEEE result. They were **not** overwritten and
were **not** used to choose a calibrator.

Selected calibrator at the time: isotonic. Threshold for the binary metrics below: 0.5.

| metric | value |
| --- | --- |
| PR-AUC | {_fmt(hist_m.get("pr_auc"), 6)} |
| ROC-AUC | {_fmt(hist_m.get("roc_auc"), 6)} |
| Precision | {_fmt(hist_m.get("precision"), 6)} |
| Recall | {_fmt(hist_m.get("recall"), 6)} |
| F1 | {_fmt(hist_m.get("f1"), 6)} |
| FPR | {_fmt(hist_m.get("false_positive_rate"), 8)} |
| FNR | {_fmt(hist_m.get("false_negative_rate"), 6)} |
| TP / FP / FN / TN | {hist_cm.get("tp")} / {hist_cm.get("fp")} / {hist_cm.get("fn")} / {hist_cm.get("tn")} |

{hist.get("note")}

No new frozen-test metric is reported from this audit.

## AUDIT F — Thresholds (validation only)

Phase 9 validation-selected policy (isotonic val scores):
APPROVE < {existing.get("approve_below")},
REVIEW {existing.get("review_from")}–{existing.get("review_to")},
BLOCK > {existing.get("block_above")}.

Existing policy on **isotonic** validation scores:

{_fmt_policy(thr.get("existing_on_isotonic_val"))}

Existing policy on **sigmoid** validation scores (same cutpoints, different map):

{_fmt_policy(thr.get("existing_on_sigmoid_val"))}

Re-selected three-way policy on validation for the audit operating method
(`{thr.get("reselected_for_operating_method")}`), still validation-only:

{_fmt_policy(thr.get("reselected_operating"))}

{thr.get("note")}

Thresholds were not optimized on TEST.

## Conclusion

**`{rec.get("decision")}`**

### Evidence

{evidence_md}

### Rationale

{rationale_md}

Live model remains `xgb-iforest-v1-calibrated`. IEEE candidates remain OFFLINE CANDIDATE.
This is not production payment-fraud accuracy.

## Integrity

Protected artifacts unchanged: {integrity.get("protected_artifacts_unchanged")}
XGBoost retrained: {meth.get("xgboost_refit")}
Frozen TEST touched: {meth.get("test_scored")}
New calibration artifact written: {integrity.get("new_calibration_artifact_written")}

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
"""


def _fmt_policy(block: dict | None) -> str:
    if not block:
        return "_not computed_"
    dec = block.get("decisions") or {}
    thr = block.get("thresholds") or {}
    return (
        f"- APPROVE/REVIEW/BLOCK counts: {dec.get('APPROVE')} / {dec.get('REVIEW')} / {dec.get('BLOCK')}\n"
        f"- fraud catch (REVIEW or BLOCK): {_fmt(block.get('fraud_catch_rate_review_or_block'), 6)}\n"
        f"- val cost (prototype units): {_fmt(block.get('val_cost'), 1)}\n"
        f"- cutpoints: APPROVE < {_fmt(thr.get('approve_below'), 6)}, "
        f"BLOCK > {_fmt(thr.get('block_above'), 6)}"
    )
