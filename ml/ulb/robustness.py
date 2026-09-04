"""Calibration robustness audit on TRAIN/VALIDATION scores only.

Never scores the chronological test set. Never writes Phase 2 calibration artifacts.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from ml.ulb.calibration import (
    _log_loss,
    calibration_diagnostics,
    clip_proba,
    fit_calibrators,
    reliability_bins,
)

METHODS = ("raw", "sigmoid", "isotonic")


def _as_xy(raw, y) -> tuple[np.ndarray, np.ndarray]:
    return clip_proba(raw), np.asarray(y).astype(int)


def ranking_metrics(y, p) -> dict:
    y = np.asarray(y).astype(int)
    p = clip_proba(p)
    return {
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "n_unique": int(np.unique(np.round(p, 12)).size),
    }


def score_bundle(y, p, label: str) -> dict:
    diag = calibration_diagnostics(y, p, label)
    rank = ranking_metrics(y, p)
    return {
        "label": label,
        "brier": diag["brier"],
        "log_loss": diag["log_loss"],
        "ece_uniform_10": diag["ece_uniform_10"],
        "ece_quantile_10": diag["ece_quantile_10"],
        "mean_predicted": diag["mean_predicted"],
        "empirical_prevalence": diag["empirical_prevalence"],
        "n_unique_predictions": diag["n_unique_predictions"],
        "n_positive": diag["n_positive"],
        "n_samples": diag["n_samples"],
        "pr_auc": rank["pr_auc"],
        "roc_auc": rank["roc_auc"],
        "reliability_uniform": diag["reliability_uniform"],
    }


def apply_method(raw, y_fit, raw_fit, method: str) -> np.ndarray:
    raw = clip_proba(raw)
    if method == "raw":
        return raw
    fitted = fit_calibrators(raw_fit, y_fit)
    return fitted.transform(raw, method)


def pr_auc_tiebroken(y, calibrated, raw) -> float:
    """Preserve raw order inside isotonic plateaus. If this recovers PR-AUC, ties caused the drop."""
    y = np.asarray(y).astype(int)
    cal = clip_proba(calibrated)
    raw = clip_proba(raw)
    span = float(np.ptp(raw)) or 1.0
    scores = cal + (1e-12 * (raw / span))
    return float(average_precision_score(y, scores)) if y.sum() else 0.0


def isotonic_is_monotone(raw, calibrated, atol: float = 1e-12) -> bool:
    raw = clip_proba(raw)
    cal = clip_proba(calibrated)
    order = np.argsort(raw, kind="mergesort")
    return bool(np.all(np.diff(cal[order]) >= -atol))


def staircase_diagnostics(y, raw, calibrated) -> dict:
    raw = clip_proba(raw)
    cal = clip_proba(calibrated)
    y = np.asarray(y).astype(int)
    n_raw = int(np.unique(np.round(raw, 12)).size)
    n_cal = int(np.unique(np.round(cal, 12)).size)
    raw_pr = float(average_precision_score(y, raw)) if y.sum() else 0.0
    cal_pr = float(average_precision_score(y, cal)) if y.sum() else 0.0
    tied_pr = pr_auc_tiebroken(y, cal, raw)
    return {
        "monotone_nondecreasing_in_raw": isotonic_is_monotone(raw, cal),
        "n_unique_raw": n_raw,
        "n_unique_calibrated": n_cal,
        "unique_ratio": float(n_cal / n_raw) if n_raw else None,
        "pr_auc_raw": raw_pr,
        "pr_auc_calibrated": cal_pr,
        "pr_auc_calibrated_tiebroken_by_raw": tied_pr,
        "pr_auc_drop": float(raw_pr - cal_pr),
        "pr_auc_drop_after_tiebreak": float(raw_pr - tied_pr),
        "drop_explained_by_ties": bool(
            (raw_pr - cal_pr) <= 1e-12
            or (raw_pr - tied_pr) <= 0.05 * max(raw_pr - cal_pr, 0.0) + 1e-12
        ),
        "note": (
            "Isotonic is monotone in the raw score, so it cannot reverse pairs. "
            "A PR-AUC drop that disappears after breaking ties with the raw order "
            "is staircase compression, not a rank reversal."
        ),
    }


def _summarize(values: list[float | None]) -> dict:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "p025": None, "p50": None, "p975": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p025": float(np.quantile(arr, 0.025)),
        "p50": float(np.quantile(arr, 0.50)),
        "p975": float(np.quantile(arr, 0.975)),
    }


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y).astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("stratified bootstrap requires both classes")
    bpos = rng.choice(pos, size=len(pos), replace=True)
    bneg = rng.choice(neg, size=len(neg), replace=True)
    return np.concatenate([bpos, bneg])


def bootstrap_frozen_maps(y, maps: dict[str, np.ndarray], n_boot: int, seed: int) -> dict:
    """Uncertainty of metrics for already-fitted maps. Does not refit calibrators."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    draws = {m: {"brier": [], "log_loss": [], "ece_uniform_10": [], "pr_auc": []} for m in maps}
    for _ in range(n_boot):
        idx = stratified_bootstrap_indices(y, rng)
        yb = y[idx]
        for name, p in maps.items():
            pb = clip_proba(p)[idx]
            draws[name]["brier"].append(float(brier_score_loss(yb, pb)))
            draws[name]["log_loss"].append(_log_loss(yb, pb))
            draws[name]["ece_uniform_10"].append(reliability_bins(yb, pb, n_bins=10, strategy="uniform")["ece"])
            draws[name]["pr_auc"].append(float(average_precision_score(yb, pb)) if yb.sum() else 0.0)
    out = {}
    for name in maps:
        out[name] = {metric: _summarize(vals) for metric, vals in draws[name].items()}
    paired = {
        "brier_isotonic_minus_sigmoid": _summarize(
            list(np.array(draws["isotonic"]["brier"]) - np.array(draws["sigmoid"]["brier"]))
        ),
        "brier_isotonic_minus_raw": _summarize(
            list(np.array(draws["isotonic"]["brier"]) - np.array(draws["raw"]["brier"]))
        ),
        "pr_auc_isotonic_minus_raw": _summarize(
            list(np.array(draws["isotonic"]["pr_auc"]) - np.array(draws["raw"]["pr_auc"]))
        ),
        "note": (
            "Paired differences on resampled (y, p) of frozen maps. Negative Brier difference "
            "means isotonic looks better. This is still in-sample for the calibrator fit."
        ),
    }
    return {"n_boot": n_boot, "seed": seed, "stratified": True, "methods": out, "paired_differences": paired}


def nested_holdout(raw, y, n_splits: int, test_size: float, seed: int) -> dict:
    """Fit calibrators on a val subset, evaluate on the held-out val subset. Repeat."""
    raw, y = _as_xy(raw, y)
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    rows: dict[str, dict[str, list]] = {
        m: {"brier": [], "log_loss": [], "ece_uniform_10": [], "pr_auc": [], "n_unique": []} for m in METHODS
    }
    skipped = 0
    used = 0
    for fit_idx, eval_idx in splitter.split(raw, y):
        y_fit, y_eval = y[fit_idx], y[eval_idx]
        if y_fit.sum() < 2 or y_eval.sum() < 2 or len(np.unique(y_fit)) < 2:
            skipped += 1
            continue
        used += 1
        raw_fit, raw_eval = raw[fit_idx], raw[eval_idx]
        for method in METHODS:
            p_eval = apply_method(raw_eval, y_fit, raw_fit, method)
            bundle = score_bundle(y_eval, p_eval, method)
            rows[method]["brier"].append(bundle["brier"])
            rows[method]["log_loss"].append(bundle["log_loss"])
            rows[method]["ece_uniform_10"].append(bundle["ece_uniform_10"])
            rows[method]["pr_auc"].append(bundle["pr_auc"])
            rows[method]["n_unique"].append(bundle["n_unique_predictions"])
    summary = {}
    for method in METHODS:
        summary[method] = {metric: _summarize(vals) for metric, vals in rows[method].items()}
    wins = {
        "isotonic_brier_lt_sigmoid": (
            float(np.mean(np.array(rows["isotonic"]["brier"]) < np.array(rows["sigmoid"]["brier"])))
            if used
            else None
        ),
        "isotonic_brier_lt_raw": (
            float(np.mean(np.array(rows["isotonic"]["brier"]) < np.array(rows["raw"]["brier"])))
            if used
            else None
        ),
        "sigmoid_brier_lt_raw": (
            float(np.mean(np.array(rows["sigmoid"]["brier"]) < np.array(rows["raw"]["brier"])))
            if used
            else None
        ),
        "isotonic_pr_auc_ge_raw": (
            float(np.mean(np.array(rows["isotonic"]["pr_auc"]) >= np.array(rows["raw"]["pr_auc"]) - 1e-12))
            if used
            else None
        ),
        "sigmoid_pr_auc_ge_raw": (
            float(np.mean(np.array(rows["sigmoid"]["pr_auc"]) >= np.array(rows["raw"]["pr_auc"]) - 1e-12))
            if used
            else None
        ),
    }
    paired = {
        "brier_isotonic_minus_sigmoid": _summarize(
            list(np.array(rows["isotonic"]["brier"]) - np.array(rows["sigmoid"]["brier"]))
        )
        if used
        else _summarize([]),
        "pr_auc_isotonic_minus_raw": _summarize(
            list(np.array(rows["isotonic"]["pr_auc"]) - np.array(rows["raw"]["pr_auc"]))
        )
        if used
        else _summarize([]),
        "pr_auc_sigmoid_minus_raw": _summarize(
            list(np.array(rows["sigmoid"]["pr_auc"]) - np.array(rows["raw"]["pr_auc"]))
        )
        if used
        else _summarize([]),
    }
    return {
        "protocol": "stratified_shuffle_split_on_validation",
        "n_splits_requested": n_splits,
        "n_splits_used": used,
        "n_splits_skipped": skipped,
        "test_size": test_size,
        "seed": seed,
        "methods": summary,
        "win_rates": wins,
        "paired_differences": paired,
        "note": (
            "Calibrators are refit on each validation subset and scored on the held-out subset. "
            "This removes the in-sample advantage isotonic has when fit and scored on the same 55 positives."
        ),
    }


def kfold_oof(raw, y, n_splits: int, seed: int) -> dict:
    """K-fold out-of-fold calibration on validation scores. Frozen booster, no test rows."""
    raw, y = _as_xy(raw, y)
    n_splits = min(n_splits, max(int(y.sum()), 2))
    if n_splits < 2:
        return {"skipped": True, "reason": "not enough positives for k-fold"}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = {m: np.zeros(len(y), dtype=float) for m in METHODS}
    fold_rows: dict[str, dict[str, list]] = {
        m: {"brier": [], "log_loss": [], "ece_uniform_10": [], "pr_auc": []} for m in METHODS
    }
    for fit_idx, eval_idx in skf.split(raw, y):
        raw_fit, y_fit = raw[fit_idx], y[fit_idx]
        raw_eval, y_eval = raw[eval_idx], y[eval_idx]
        for method in METHODS:
            p_eval = apply_method(raw_eval, y_fit, raw_fit, method)
            oof[method][eval_idx] = p_eval
            bundle = score_bundle(y_eval, p_eval, method)
            for key in ("brier", "log_loss", "ece_uniform_10", "pr_auc"):
                fold_rows[method][key].append(bundle[key])
    pooled = {m: score_bundle(y, oof[m], m) for m in METHODS}
    # Drop bulky reliability bins from pooled JSON except ECE
    for m in pooled:
        pooled[m] = {k: v for k, v in pooled[m].items() if k != "reliability_uniform"}
        pooled[m]["reliability_uniform"] = {
            "ece": calibration_diagnostics(y, oof[m], m)["ece_uniform_10"],
            "bins": reliability_bins(y, oof[m], n_bins=10, strategy="uniform")["bins"],
        }
    return {
        "protocol": "stratified_kfold_oof_on_validation",
        "n_splits": n_splits,
        "seed": seed,
        "fold_summaries": {m: {k: _summarize(v) for k, v in fold_rows[m].items()} for m in METHODS},
        "pooled_oof": pooled,
        "note": (
            "Each validation row is scored by a calibrator that did not see that row. "
            "Pooled OOF Brier/log loss/ECE/PR-AUC are the primary CV estimates."
        ),
    }


def train_fit_val_eval(raw_train, y_train, raw_val, y_val) -> dict:
    """Fit calibrators on train scores, evaluate on validation. Test unused.

    Train booster scores are in-sample for the XGBoost fit; that limitation is recorded.
    """
    raw_train, y_train = _as_xy(raw_train, y_train)
    raw_val, y_val = _as_xy(raw_val, y_val)
    out = {}
    for method in METHODS:
        p_val = apply_method(raw_val, y_train, raw_train, method)
        bundle = score_bundle(y_val, p_val, method)
        bundle.pop("reliability_uniform", None)
        if method == "isotonic":
            bundle["staircase"] = staircase_diagnostics(y_val, raw_val, p_val)
        out[method] = bundle
    return {
        "protocol": "fit_on_train_scores_eval_on_validation",
        "train_n": int(len(y_train)),
        "train_positives": int(y_train.sum()),
        "val_n": int(len(y_val)),
        "val_positives": int(y_val.sum()),
        "methods": out,
        "note": (
            "Calibrator holdout uses more positives (train) than the 55 in validation. "
            "Raw train scores are booster-in-sample, so this is not a fully clean calibration CV."
        ),
    }


def ci_excludes_zero(summary: dict, positive: bool | None = None) -> bool | None:
    lo, hi = summary.get("p025"), summary.get("p975")
    if lo is None or hi is None:
        return None
    if lo > 0 or hi < 0:
        if positive is None:
            return True
        return (hi < 0) if not positive else (lo > 0)
    return False


def recommend(payload: dict) -> dict:
    """Recommend from nested/OOF robustness, not from a single in-sample Brier."""
    nested = payload["nested_holdout_validation"]
    oof = payload["kfold_oof_validation"]
    train_hold = payload["train_fit_val_eval"]
    stair = payload["staircase_in_sample_validation"]
    in_sample = payload["in_sample_validation"]

    nested_brier_diff = nested["paired_differences"]["brier_isotonic_minus_sigmoid"]
    nested_pr_diff = nested["paired_differences"]["pr_auc_isotonic_minus_raw"]
    oof_pool = oof.get("pooled_oof") or {}
    oof_iso = (oof_pool.get("isotonic") or {}).get("brier")
    oof_sig = (oof_pool.get("sigmoid") or {}).get("brier")
    oof_raw = (oof_pool.get("raw") or {}).get("brier")
    oof_iso_pr = (oof_pool.get("isotonic") or {}).get("pr_auc")
    oof_sig_pr = (oof_pool.get("sigmoid") or {}).get("pr_auc")
    oof_raw_pr = (oof_pool.get("raw") or {}).get("pr_auc")
    oof_iso_nuniq = (oof_pool.get("isotonic") or {}).get("n_unique_predictions")

    train_iso_brier = train_hold["methods"]["isotonic"]["brier"]
    train_sig_brier = train_hold["methods"]["sigmoid"]["brier"]
    train_raw_brier = train_hold["methods"]["raw"]["brier"]
    train_iso_pr = train_hold["methods"]["isotonic"]["pr_auc"]
    train_sig_pr = train_hold["methods"]["sigmoid"]["pr_auc"]
    train_raw_pr = train_hold["methods"]["raw"]["pr_auc"]

    nested_iso_unique = nested["methods"]["isotonic"]["n_unique"]["mean"]
    nested_sig_brier = nested["methods"]["sigmoid"]["brier"]["mean"]
    nested_iso_brier = nested["methods"]["isotonic"]["brier"]["mean"]
    nested_raw_brier = nested["methods"]["raw"]["brier"]["mean"]
    nested_iso_pr = nested["methods"]["isotonic"]["pr_auc"]["mean"]
    nested_sig_pr = nested["methods"]["sigmoid"]["pr_auc"]["mean"]
    nested_raw_pr = nested["methods"]["raw"]["pr_auc"]["mean"]

    iso_brier_win_vs_sig = nested["win_rates"]["isotonic_brier_lt_sigmoid"]
    iso_nested_not_robustly_better = (
        nested_brier_diff.get("p975") is not None and nested_brier_diff["p975"] >= 0
    )
    ranking_hurt = (
        nested_pr_diff.get("mean") is not None and nested_pr_diff["mean"] < -0.005
    ) or (oof_iso_pr is not None and oof_raw_pr is not None and (oof_iso_pr - oof_raw_pr) < -0.005)
    staircase_cause = bool(stair.get("drop_explained_by_ties"))
    degenerate = (oof_iso_nuniq is not None and oof_iso_nuniq < 20) or (
        nested_iso_unique is not None and nested_iso_unique < 20
    )

    sigmoid_helps_brier = (
        nested_sig_brier is not None
        and nested_raw_brier is not None
        and nested_sig_brier < nested_raw_brier
        and (oof_sig is None or oof_raw is None or oof_sig <= oof_raw + 1e-12)
    )
    sigmoid_preserves_ranking = (
        nested_sig_pr is not None
        and nested_raw_pr is not None
        and nested_sig_pr >= nested_raw_pr - 0.005
    )

    evidence = []
    evidence.append(
        f"In-sample validation Brier: raw={in_sample['raw']['brier']:.6f}, "
        f"sigmoid={in_sample['sigmoid']['brier']:.6f}, isotonic={in_sample['isotonic']['brier']:.6f} "
        f"(isotonic has an in-sample advantage; {in_sample['isotonic']['n_unique_predictions']} unique p)."
    )
    evidence.append(
        f"Nested val holdout mean Brier: raw={nested_raw_brier:.6f}, "
        f"sigmoid={nested_sig_brier:.6f}, isotonic={nested_iso_brier:.6f}; "
        f"isotonic better than sigmoid in "
        f"{'n/a' if iso_brier_win_vs_sig is None else f'{iso_brier_win_vs_sig:.0%}'} of repeats; "
        f"paired isotonic-sigmoid Brier mean={nested_brier_diff.get('mean')}, "
        f"95% interval [{nested_brier_diff.get('p025')}, {nested_brier_diff.get('p975')}]."
    )
    evidence.append(
        f"Nested mean PR-AUC: raw={nested_raw_pr:.4f}, sigmoid={nested_sig_pr:.4f}, "
        f"isotonic={nested_iso_pr:.4f}."
    )
    if oof_pool:
        evidence.append(
            f"5-fold OOF pooled: Brier raw={oof_raw}, sigmoid={oof_sig}, isotonic={oof_iso}; "
            f"PR-AUC raw={oof_raw_pr}, sigmoid={oof_sig_pr}, isotonic={oof_iso_pr}; "
            f"isotonic unique p={oof_iso_nuniq}."
        )
    evidence.append(
        f"Train-fit/val-eval Brier: raw={train_raw_brier:.6f}, sigmoid={train_sig_brier:.6f}, "
        f"isotonic={train_iso_brier:.6f}; PR-AUC raw={train_raw_pr:.4f}, "
        f"sigmoid={train_sig_pr:.4f}, isotonic={train_iso_pr:.4f}."
    )
    evidence.append(
        f"Staircase: unique {stair.get('n_unique_raw')}→{stair.get('n_unique_calibrated')}, "
        f"monotone={stair.get('monotone_nondecreasing_in_raw')}, "
        f"PR-AUC drop={stair.get('pr_auc_drop')}, after raw-order tie-break drop="
        f"{stair.get('pr_auc_drop_after_tiebreak')} (explained_by_ties={staircase_cause})."
    )

    method = "sigmoid/Platt"
    rationale = []
    inconclusive = False
    train_iso_beats_sig_brier = train_iso_brier < train_sig_brier
    nested_iso_beats_sig_brier = (
        iso_brier_win_vs_sig is not None
        and iso_brier_win_vs_sig >= 0.8
        and not iso_nested_not_robustly_better
    )

    if nested_iso_beats_sig_brier:
        rationale.append(
            "Nested validation Brier favors isotonic in most repeats; that is a real result, "
            "not dismissed because the in-sample Brier was also lowest."
        )
    if ranking_hurt and staircase_cause:
        rationale.append(
            "Isotonic PR-AUC degradation is from plateau ties, not rank reversal. "
            "The calibrated score is still a weaker ranking statistic."
        )
    if degenerate:
        rationale.append(
            "Isotonic stays a coarse step function under resampling "
            f"(nested mean unique p={nested_iso_unique})."
        )
    if not train_iso_beats_sig_brier:
        rationale.append(
            "When calibrators are fit on train scores and scored on validation, sigmoid has "
            "lower Brier than isotonic and does not collapse PR-AUC. That holdout has more "
            "positives than the 55-row validation nest."
        )

    if nested_iso_beats_sig_brier and not ranking_hurt and train_iso_beats_sig_brier:
        method = "isotonic"
        rationale.append("Isotonic wins nested Brier without a ranking cost and wins the train holdout.")
    elif not sigmoid_helps_brier and ranking_hurt and nested_sig_brier > nested_raw_brier + 1e-6:
        method = "raw probabilities with documented calibration limitation"
        rationale.append("Neither calibrator improves nested Brier over raw without a ranking cost.")
    else:
        method = "sigmoid/Platt"
        rationale.append(
            "Sigmoid is a strictly monotone map (nested PR-AUC matches raw), reduces Brier vs raw, "
            "and avoids the staircase. It is the robustness pick when Brier and ranking disagree."
        )

    if nested_iso_beats_sig_brier and ranking_hurt:
        inconclusive = True
        rationale.append(
            "Inconclusive as a single 'best' map: nested Brier prefers isotonic; ranking, "
            "uniqueness, and the train-fit/val-eval holdout prefer sigmoid. "
            "The recommended method is the conservative operational map, not the lowest nested Brier."
        )

    uncertainty = (
        f"Validation has {in_sample['raw']['n_positive']} positives, so ECE and even Brier "
        f"have wide resampling intervals. Nested holdout used {nested['n_splits_used']} splits "
        f"(eval fraction {nested['test_size']}). Frozen-map bootstrap intervals describe "
        f"in-sample metric noise, not calibrator generalization. Train-fit/val-eval uses "
        f"{train_hold['train_positives']} train positives but booster-in-sample scores."
    )
    remaining = (
        "The chronological test set was not used. A later one-shot test evaluation of a "
        "changed calibrator would still be a single draw at ~52 fraud cases. Product scoring "
        "is unchanged. Sigmoid/isotonic fitted on 55 val positives cannot be treated as a "
        "production probability of fraud."
    )
    return {
        "recommended_calibration_method": method,
        "inconclusive": inconclusive,
        "evidence": evidence,
        "rationale": rationale,
        "uncertainty": uncertainty,
        "remaining_limitation": remaining,
        "checks": {
            "isotonic_nested_brier_not_robustly_better_than_sigmoid": iso_nested_not_robustly_better,
            "isotonic_ranking_hurt": ranking_hurt,
            "staircase_explains_pr_auc_drop": staircase_cause,
            "isotonic_degenerate_unique_values": degenerate,
            "sigmoid_helps_nested_brier": sigmoid_helps_brier,
            "sigmoid_preserves_ranking": sigmoid_preserves_ranking,
        },
    }
