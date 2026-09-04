"""Markdown report for the calibration robustness audit."""

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


def render_robustness_report(payload: dict) -> str:
    rec = payload["recommendation"]
    split = payload["official_split"]
    meth = payload["methodology"]
    ins = payload["in_sample_validation"]
    stair = payload["staircase_in_sample_validation"]
    boot = payload["bootstrap_frozen_maps"]
    nested = payload["nested_holdout_validation"]
    oof = payload["kfold_oof_validation"]
    th = payload["train_fit_val_eval"]
    keys = ("brier", "log_loss", "ece_uniform_10", "pr_auc", "n_unique_predictions")

    nested_tables = []
    for metric in ("brier", "log_loss", "ece_uniform_10", "pr_auc", "n_unique"):
        lines = [
            f"**{metric}**",
            "",
            "| method | mean | std | p2.5 | p50 | p97.5 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name in ("raw", "sigmoid", "isotonic"):
            lines.append(_sum_row(name, nested["methods"][name][metric]))
        nested_tables.append("\n".join(lines))

    boot_tables = []
    for metric in ("brier", "log_loss", "ece_uniform_10", "pr_auc"):
        lines = [
            f"**{metric}**",
            "",
            "| method | mean | std | p2.5 | p50 | p97.5 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name in ("raw", "sigmoid", "isotonic"):
            lines.append(_sum_row(name, boot["methods"][name][metric]))
        boot_tables.append("\n".join(lines))

    oof_fold = []
    if oof.get("fold_summaries"):
        for metric in ("brier", "log_loss", "ece_uniform_10", "pr_auc"):
            lines = [
                f"**{metric} (per-fold)**",
                "",
                "| method | mean | std | p2.5 | p50 | p97.5 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for name in ("raw", "sigmoid", "isotonic"):
                lines.append(_sum_row(name, oof["fold_summaries"][name][metric]))
            oof_fold.append("\n".join(lines))

    oof_pool = oof.get("pooled_oof") or {}
    oof_pool_table = (
        _bundle_table(oof_pool, keys) if oof_pool else "_k-fold skipped_"
    )

    evidence_md = "\n".join(f"- {item}" for item in rec.get("evidence") or [])
    rationale_md = "\n".join(f"- {item}" for item in rec.get("rationale") or [])

    return f"""# Calibration robustness audit

This audit does **not** use chronological test labels for fitting, selection, or the
recommendation. The official test slice was not scored. Phase 2 files
(`calibration_metrics.json`, `calibration_report.md`, `ulb_metrics.json`) were left intact.

The Phase 2 point-estimate choice was isotonic because it had the lowest in-sample
validation Brier. That is not treated as decisive here.

## Split (unchanged)

| split | n | fraud | prevalence |
| --- | --- | --- | --- |
| train | {split['train']['n']} | {split['train']['fraud']} | {_fmt(split['train']['prevalence'], 6)} |
| validation | {split['val']['n']} | {split['val']['fraud']} | {_fmt(split['val']['prevalence'], 6)} |
| test (not scored) | {split['test']['n']} | {split['test']['fraud']} | {_fmt(split['test']['prevalence'], 6)} |

`no_future_in_train`: {split.get('no_future_in_train')}
Matches `ulb_metrics.json`: {payload.get('split_matches_ulb_metrics')}

Booster scores: train n={meth.get('train_n')} ({meth.get('train_positives')} positives),
validation n={meth.get('validation_n')} ({meth.get('validation_positives')} positives).

## 1. In-sample validation comparison (fit and score on the same 55 fraud cases)

Isotonic and sigmoid are fit on all validation rows, then scored on those same rows.
Isotonic is expected to look best here.

{_bundle_table(ins, keys)}

Mean predicted vs prevalence (raw { _fmt(ins['raw']['mean_predicted'], 6) } vs
{ _fmt(ins['raw']['empirical_prevalence'], 6) }).

### Calibration curves (uniform 10 bins, in-sample)

**raw**

{_bin_table(ins['raw'].get('reliability_uniform'))}

**sigmoid**

{_bin_table(ins['sigmoid'].get('reliability_uniform'))}

**isotonic**

{_bin_table(ins['isotonic'].get('reliability_uniform'))}

Figures: `ml/evaluation/figures/robustness_val_insample_{{raw,sigmoid,isotonic}}.svg`.

## 2. Uncertainty of frozen-map metrics (stratified bootstrap of validation)

The maps are **not** refit. This interval is sampling noise of Brier / log loss / ECE / PR-AUC
for a fixed map, not a test of whether isotonic overfits.

n_boot={boot.get('n_boot')}, stratified=true, seed={boot.get('seed')}.

{chr(10).join(boot_tables)}

Paired isotonic − sigmoid Brier: mean={_fmt(boot['paired_differences']['brier_isotonic_minus_sigmoid'].get('mean'), 8)},
95% [{_fmt(boot['paired_differences']['brier_isotonic_minus_sigmoid'].get('p025'), 8)},
{_fmt(boot['paired_differences']['brier_isotonic_minus_sigmoid'].get('p975'), 8)}].

Paired isotonic − raw PR-AUC: mean={_fmt(boot['paired_differences']['pr_auc_isotonic_minus_raw'].get('mean'), 6)},
95% [{_fmt(boot['paired_differences']['pr_auc_isotonic_minus_raw'].get('p025'), 6)},
{_fmt(boot['paired_differences']['pr_auc_isotonic_minus_raw'].get('p975'), 6)}].

## 3. Nested validation holdout (calibrators refit)

{nested.get('note')}

Splits used: {nested.get('n_splits_used')} / {nested.get('n_splits_requested')}
(eval fraction {nested.get('test_size')}, skipped {nested.get('n_splits_skipped')}).

{chr(10).join(nested_tables)}

Win rates (fraction of repeats):
- isotonic Brier < sigmoid: {_fmt(nested['win_rates'].get('isotonic_brier_lt_sigmoid'), 3)}
- isotonic Brier < raw: {_fmt(nested['win_rates'].get('isotonic_brier_lt_raw'), 3)}
- sigmoid Brier < raw: {_fmt(nested['win_rates'].get('sigmoid_brier_lt_raw'), 3)}
- isotonic PR-AUC ≥ raw: {_fmt(nested['win_rates'].get('isotonic_pr_auc_ge_raw'), 3)}
- sigmoid PR-AUC ≥ raw: {_fmt(nested['win_rates'].get('sigmoid_pr_auc_ge_raw'), 3)}

Paired isotonic − sigmoid Brier: mean={_fmt(nested['paired_differences']['brier_isotonic_minus_sigmoid'].get('mean'), 8)},
95% [{_fmt(nested['paired_differences']['brier_isotonic_minus_sigmoid'].get('p025'), 8)},
{_fmt(nested['paired_differences']['brier_isotonic_minus_sigmoid'].get('p975'), 8)}].

Paired isotonic − raw PR-AUC: mean={_fmt(nested['paired_differences']['pr_auc_isotonic_minus_raw'].get('mean'), 6)},
95% [{_fmt(nested['paired_differences']['pr_auc_isotonic_minus_raw'].get('p025'), 6)},
{_fmt(nested['paired_differences']['pr_auc_isotonic_minus_raw'].get('p975'), 6)}].

## 4. K-fold out-of-fold calibration on validation

{oof.get('note')}

Folds: {oof.get('n_splits')}.

{chr(10).join(oof_fold) if oof_fold else '_no fold summaries_'}

**Pooled OOF** (concatenated held-out predictions):

{oof_pool_table}

OOF reliability diagrams: `ml/evaluation/figures/robustness_val_oof_{{raw,sigmoid,isotonic}}.svg`.

### OOF calibration curves

**raw**

{_bin_table((oof_pool.get('raw') or {{}}).get('reliability_uniform'))}

**sigmoid**

{_bin_table((oof_pool.get('sigmoid') or {{}}).get('reliability_uniform'))}

**isotonic**

{_bin_table((oof_pool.get('isotonic') or {{}}).get('reliability_uniform'))}

## 5. Staircase / ranking compression

In-sample isotonic vs raw on validation:

- monotone in raw score: {stair.get('monotone_nondecreasing_in_raw')}
- unique probabilities: {stair.get('n_unique_raw')} → {stair.get('n_unique_calibrated')}
- PR-AUC raw: {_fmt(stair.get('pr_auc_raw'), 6)}
- PR-AUC isotonic: {_fmt(stair.get('pr_auc_calibrated'), 6)}
- PR-AUC isotonic + raw-order tie-break: {_fmt(stair.get('pr_auc_calibrated_tiebroken_by_raw'), 6)}
- drop explained by ties: {stair.get('drop_explained_by_ties')}

{stair.get('note')}

## 6. Cross-validation-style holdout without the test set

Fit calibrators on **train** booster scores ({th.get('train_positives')} positives),
evaluate on **validation**. Train scores are booster-in-sample.

{_bundle_table(th['methods'], keys)}

{th.get('note')}

## Recommendation (robustness, not lowest in-sample Brier)

**Recommended calibration method:** `{rec.get('recommended_calibration_method')}`

Inconclusive flag: {rec.get('inconclusive')}

### Evidence

{evidence_md}

### Rationale

{rationale_md}

### Uncertainty

{rec.get('uncertainty')}

### Remaining limitation

{rec.get('remaining_limitation')}

---

Recommended calibration method: {rec.get('recommended_calibration_method')}

Evidence: {'; '.join(rec.get('evidence') or [])}

Uncertainty: {rec.get('uncertainty')}

Remaining limitation: {rec.get('remaining_limitation')}
"""
