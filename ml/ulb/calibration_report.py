"""Markdown report for ULB probability and threshold calibration."""

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


def _bin_table(rel: dict) -> str:
    lines = [
        "| bin | lo | hi | n | positives | mean predicted | empirical rate | |gap| |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for b in rel.get("bins") or []:
        lines.append(
            "| {bin} | {lo} | {hi} | {count} | {n_positive} | {mean_predicted} | {empirical_positive_rate} | {gap} |".format(
                bin=b.get("bin"),
                lo=_fmt(b.get("lo"), 3),
                hi=_fmt(b.get("hi"), 3),
                count=b.get("count"),
                n_positive=b.get("n_positive"),
                mean_predicted=_fmt(b.get("mean_predicted"), 6),
                empirical_positive_rate=_fmt(b.get("empirical_positive_rate"), 6),
                gap=_fmt(b.get("gap"), 6),
            )
        )
    return "\n".join(lines)


def _method_table(val: dict) -> str:
    lines = [
        "| method | Brier | log loss | ECE (uniform 10) | ECE (quantile 10) | mean predicted | unique p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("raw", "sigmoid", "isotonic"):
        d = val[name]
        lines.append(
            f"| {name} | {_fmt(d['brier'], 8)} | {_fmt(d['log_loss'], 6)} | "
            f"{_fmt(d['ece_uniform_10'], 6)} | {_fmt(d['ece_quantile_10'], 6)} | "
            f"{_fmt(d['mean_predicted'], 6)} | {d['n_unique_predictions']} |"
        )
    return "\n".join(lines)


def _threshold_excerpt(rows: list[dict]) -> str:
    keep = []
    for row in rows:
        t = row["threshold"]
        if abs(t * 100 - round(t * 100)) < 1e-9 and round(t * 100) % 5 == 0:
            keep.append(row)
        elif t in {0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99}:
            keep.append(row)
    # de-dupe by threshold
    seen = set()
    uniq = []
    for row in keep:
        key = round(row["threshold"], 4)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
    lines = [
        "| threshold | TP | FP | TN | FN | precision | recall | F1 | FPR | FNR |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in uniq:
        lines.append(
            f"| {_fmt(row['threshold'], 3)} | {row['tp']} | {row['fp']} | {row['tn']} | {row['fn']} | "
            f"{_fmt(row['precision'], 4)} | {_fmt(row['recall'], 4)} | {_fmt(row['f1'], 4)} | "
            f"{_fmt(row['fpr'], 6)} | {_fmt(row['fnr'], 4)} |"
        )
    return "\n".join(lines)


def _cost_block(name: str, result: dict) -> str:
    u = result["unconstrained"]
    c = result["capacity_capped"]
    s = result["selected"]
    costs = result["costs"]
    lines = [
        f"### {name}",
        "",
        f"- FN cost: {costs['false_negative_cost']}",
        f"- FP cost: {costs['false_positive_cost']}",
        f"- Review cost: {costs['review_cost']}",
        "",
        "| variant | T_REVIEW | T_BLOCK | expected cost / txn | approve | review | block | fraud catch | feasible |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| unconstrained | {_fmt(u.get('t_review'), 2)} | {_fmt(u.get('t_block'), 2)} | "
            f"{_fmt(u.get('expected_cost_per_txn'), 6)} | {_fmt(u.get('approve_rate'), 4)} | "
            f"{_fmt(u.get('review_rate'), 4)} | {_fmt(u.get('block_rate'), 4)} | "
            f"{_fmt(u.get('fraud_catch_rate'), 4)} | {_fmt(u.get('feasible'))} |"
        ),
        (
            f"| 5% capacity cap | {_fmt(c.get('t_review'), 2)} | {_fmt(c.get('t_block'), 2)} | "
            f"{_fmt(c.get('expected_cost_per_txn'), 6)} | {_fmt(c.get('approve_rate'), 4)} | "
            f"{_fmt(c.get('review_rate'), 4)} | {_fmt(c.get('block_rate'), 4)} | "
            f"{_fmt(c.get('fraud_catch_rate'), 4)} | {_fmt(c.get('feasible'))} |"
        ),
        (
            f"| **selected prototype** | **{_fmt(s.get('t_review'), 2)}** | **{_fmt(s.get('t_block'), 2)}** | "
            f"{_fmt(s.get('expected_cost_per_txn'), 6)} | {_fmt(s.get('approve_rate'), 4)} | "
            f"{_fmt(s.get('review_rate'), 4)} | {_fmt(s.get('block_rate'), 4)} | "
            f"{_fmt(s.get('fraud_catch_rate'), 4)} | {_fmt(s.get('feasible'))} |"
        ),
        "",
        f"Selection rule: {result.get('selected_source')}",
    ]
    for note in result.get("notes") or []:
        lines.append(f"- {note}")
    return "\n".join(lines)


def render_calibration_report(payload: dict) -> str:
    split = payload["official_split"]
    val = payload["validation"]
    sel = payload["selected_method"]
    proto = payload["prototype_operating_thresholds"]
    test = payload["test_evaluation"]
    half = test["metrics_at_0_5"]
    blk = test["metrics_at_prototype_t_block"]
    three = test["three_way_at_prototype_thresholds"]
    meth = payload["methodology"]
    best = payload.get("best_f1_threshold_validation") or {}

    parts = f"""# ULB probability and threshold calibration

**PROTOTYPE CALIBRATION.** These operating points are not industry-standard thresholds
and are not the live RazorGuard X product policy.

Track: `{payload.get("track")}`. Booster: `{payload.get("booster_model_version")}`.
Calibrated identity: `{payload.get("calibrated_model_version")}`.
Synthetic live model `xgb-iforest-v1-calibrated` was not modified.

Calibrated P(Class=1) is **not** the product `final_risk_score`.

## 1. Dataset split

Official chronological 70/15/15 on Time (unchanged from ULB training).

| split | n | fraud | legitimate | prevalence | time_min | time_max |
| --- | --- | --- | --- | --- | --- | --- |
| train | {split['train']['n']} | {split['train']['fraud']} | {split['train']['legitimate']} | {_fmt(split['train']['prevalence'], 6)} | {split['train']['time_min']} | {split['train']['time_max']} |
| validation | {split['val']['n']} | {split['val']['fraud']} | {split['val']['legitimate']} | {_fmt(split['val']['prevalence'], 6)} | {split['val']['time_min']} | {split['val']['time_max']} |
| test | {split['test']['n']} | {split['test']['fraud']} | {split['test']['legitimate']} | {_fmt(split['test']['prevalence'], 6)} | {split['test']['time_min']} | {split['test']['time_max']} |

`no_future_in_train`: {split.get("no_future_in_train")}
Split matches `ulb_metrics.json`: {payload.get("split_matches_ulb_metrics")}

## 2. Calibration methodology

- TRAIN: existing `{payload.get("booster_model_version")}` booster is reused. It is not refit here.
- VALIDATION ({meth.get("validation_n")} rows, {meth.get("calibrator_fit_n_positive")} positives): fit sigmoid and isotonic on **raw validation scores only**; select method; select thresholds/costs.
- TEST ({meth.get("test_n")} rows): evaluate the frozen calibrator and frozen thresholds **once**.

| gate | used for fit? | used for selection? |
| --- | --- | --- |
| validation labels | yes (calibrators + thresholds) | yes |
| test labels | {meth.get("test_used_for_calibrator_fit")} | {meth.get("test_used_for_method_selection")} / {meth.get("test_used_for_threshold_selection")} |

Sigmoid/Platt = `LogisticRegression` on 1-D raw scores.
Isotonic = `IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")`.
Raw = uncalibrated booster probabilities.

Reliability diagrams: `ml/evaluation/figures/ulb_reliability_val_{{raw,sigmoid,isotonic}}.svg`.

## 3–5. Raw / sigmoid / isotonic (validation)

{_method_table(val)}

Prevalence on validation is { _fmt(val['raw']['empirical_prevalence'], 6) }. Mean predicted probability
far above that prevalence means the booster is over-confident in the positive direction on average.

### Raw — reliability (uniform 10 bins)

{_bin_table(val["raw"]["reliability_uniform"])}

### Sigmoid — reliability (uniform 10 bins)

{_bin_table(val["sigmoid"]["reliability_uniform"])}

### Isotonic — reliability (uniform 10 bins)

{_bin_table(val["isotonic"]["reliability_uniform"])}

## 6. Selected method and justification

**Selected: `{sel}`**

{payload.get("selection_justification")}

"""
    for note in val.get("selection", {}).get("notes") or []:
        parts += f"- {note}\n"

    parts += f"""
## 7. Reliability metrics

See the method table above. Quantile ECE is included because uniform bins are almost empty
except near 0 under extreme imbalance. Neither ECE variant was computed on test for selection.

## 8. Threshold analysis (validation, calibrated probabilities)

Full sweep is in `calibration_metrics.json` (`binary_threshold_sweep_validation`).
Excerpt (5-point steps plus a few operating points):

{_threshold_excerpt(payload.get("binary_threshold_sweep_validation") or [])}

Best validation F1 cutoff (not automatically the three-way prototype):
threshold={_fmt(best.get("threshold"), 4)}, F1={_fmt(best.get("f1"), 4)},
precision={_fmt(best.get("precision"), 4)}, recall={_fmt(best.get("recall"), 4)}.

## 9. Cost scenarios

Costs are configuration parameters. They are not estimated from Razorpay or any issuer.

{_cost_block("Scenario A", payload["cost_scenarios"]["A"])}

{_cost_block("Scenario B", payload["cost_scenarios"]["B"])}

## 10. Selected prototype operating thresholds

Label: **PROTOTYPE CALIBRATION** (Scenario {proto.get("cost_scenario")}).

- APPROVE if calibrated p < **{proto.get("approve_below")}**
- REVIEW if **{proto.get("review_from")}** ≤ p < **{proto.get("review_to")}**
- BLOCK if p ≥ **{proto.get("block_above")}**

Cost assumptions: FN={proto["cost_assumptions"]["false_negative_cost"]},
FP={proto["cost_assumptions"]["false_positive_cost"]},
review={proto["cost_assumptions"]["review_cost"]}.

Source: {proto.get("source")}

Validation mix: approve={_fmt(proto["validation_mix"].get("approve_rate"), 4)},
review={_fmt(proto["validation_mix"].get("review_rate"), 4)},
block={_fmt(proto["validation_mix"].get("block_rate"), 4)},
expected cost/txn={_fmt(proto["validation_mix"].get("expected_cost_per_txn"), 6)}.

These are **not** `THRESHOLD_REVIEW=40` / `THRESHOLD_BLOCK=70` and must not be copied
onto the synthetic product `final_risk_score` without a separate product decision.

## 11. Test evaluation (once, selected calibrator)

Raw vs calibrated probability on the untouched chronological test set:

| quantity | raw probability | calibrated probability (`{sel}`) | risk score |
| --- | --- | --- | --- |
| definition | uncalibrated booster P(Class=1) | validation-fitted map | calibrated p × 100 |
| is a probability? | uncalibrated estimate | calibrated estimate | **no** |
| Brier | {_fmt(test["raw_probability"]["brier"], 8)} | {_fmt(test["calibrated_probability"]["brier"], 8)} | n/a |
| log loss | {_fmt(test["raw_probability"]["log_loss"], 6)} | {_fmt(test["calibrated_probability"]["log_loss"], 6)} | n/a |
| ECE uniform 10 | {_fmt(test["raw_probability"]["ece_uniform_10"], 6)} | {_fmt(test["calibrated_probability"]["ece_uniform_10"], 6)} | n/a |

Threshold-free ranking on **raw** test probabilities: PR-AUC={_fmt(test["raw_probability"].get("pr_auc"), 6)},
ROC-AUC={_fmt(test["raw_probability"].get("roc_auc"), 6)}.

Threshold-free ranking on **calibrated** test probabilities: PR-AUC={_fmt(half.get("pr_auc"), 6)},
ROC-AUC={_fmt(half.get("roc_auc"), 6)}.
A drop in PR-AUC after isotonic is ranking compression, not a reason to quietly switch methods using test.

Binary metrics at 0.5 (reference only):

| precision | recall | F1 | FPR | FNR | Brier | log loss |
| --- | --- | --- | --- | --- | --- | --- |
| {_fmt(half.get("precision"), 4)} | {_fmt(half.get("recall"), 4)} | {_fmt(half.get("f1"), 4)} | {_fmt(half.get("false_positive_rate"), 6)} | {_fmt(half.get("false_negative_rate"), 4)} | {_fmt(half.get("brier"), 8)} | {_fmt(half.get("log_loss"), 6)} |

Confusion at 0.5: TP={half["confusion_matrix"]["tp"]} FP={half["confusion_matrix"]["fp"]}
TN={half["confusion_matrix"]["tn"]} FN={half["confusion_matrix"]["fn"]}.

Binary metrics at prototype **T_BLOCK** (positive = BLOCK):

| precision | recall | F1 | FPR | FNR |
| --- | --- | --- | --- | --- |
| {_fmt(blk.get("precision"), 4)} | {_fmt(blk.get("recall"), 4)} | {_fmt(blk.get("f1"), 4)} | {_fmt(blk.get("false_positive_rate"), 6)} | {_fmt(blk.get("false_negative_rate"), 4)} |

Three-way decisions on test (frozen validation thresholds):

- APPROVE / REVIEW / BLOCK counts: {test["decision_counts"]}
- expected cost / txn (Scenario {proto.get("cost_scenario")}): {_fmt(three.get("expected_cost_per_txn"), 6)}
- approve rate: {_fmt(three.get("approve_rate"), 4)}
- review rate: {_fmt(three.get("review_rate"), 4)}
- block rate: {_fmt(three.get("block_rate"), 4)}
- approved fraud (FN-style): {three.get("fn_approved_fraud")}
- blocked legit (FP-style): {three.get("fp_blocked_legit")}
- fraud catch rate (review or block): {_fmt(three.get("fraud_catch_rate"), 4)}

## Model vs risk score

| signal | meaning |
| --- | --- |
| model probability | calibrated ULB P(Class=1) — offline only |
| behavior score | anomaly / personalized overlays on synthetic traffic |
| rule score | deterministic rule evidence |
| graph score | shared device/IP relationship risk |
| final risk score | weighted RazorGuard decision signal, **not** P(fraud) |

## Limitations

"""
    for item in payload.get("limitations") or []:
        parts += f"- {item}\n"
    parts += "\nThis report does not claim production fraud-detection performance.\n"
    return parts
