"""IEEE-CIS calibration robustness helpers. Validation/pre-test scores only."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from app.ml.ieee.evaluate import (
    EPS,
    calibration_diagnostics,
    clip_proba,
    fit_calibrators,
    reliability_bins,
)

METHODS = ("raw", "sigmoid", "isotonic")
DECISIONS = ("KEEP_ISOTONIC", "PREFER_SIGMOID", "INCONCLUSIVE_KEEP_CURRENT")


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


def score_bundle(y, p, label: str, include_reliability: bool = False) -> dict:
    diag = calibration_diagnostics(y, p, label)
    rank = ranking_metrics(y, p)
    out = {
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
    }
    if include_reliability:
        out["reliability_uniform"] = diag["reliability_uniform"]
    return out


def apply_method(raw, y_fit, raw_fit, method: str) -> np.ndarray:
    raw = clip_proba(raw)
    if method == "raw":
        return raw
    fitted = fit_calibrators(raw_fit, y_fit)
    return fitted.transform(raw, method)


def pr_auc_tiebroken(y, calibrated, raw) -> float:
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
    plateau_sizes = []
    _, counts = np.unique(np.round(cal, 12), return_counts=True)
    if len(counts):
        plateau_sizes = {
            "max_plateau_n": int(counts.max()),
            "median_plateau_n": float(np.median(counts)),
            "n_plateaus": int(len(counts)),
        }
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
        "plateaus": plateau_sizes or None,
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


def nested_holdout(raw, y, n_splits: int, test_size: float, seed: int) -> dict:
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
    summary = {method: {metric: _summarize(vals) for metric, vals in rows[method].items()} for method in METHODS}
    wins = {
        "isotonic_brier_lt_sigmoid": float(np.mean(np.array(rows["isotonic"]["brier"]) < np.array(rows["sigmoid"]["brier"]))) if used else None,
        "isotonic_brier_lt_raw": float(np.mean(np.array(rows["isotonic"]["brier"]) < np.array(rows["raw"]["brier"]))) if used else None,
        "sigmoid_brier_lt_raw": float(np.mean(np.array(rows["sigmoid"]["brier"]) < np.array(rows["raw"]["brier"]))) if used else None,
        "isotonic_pr_auc_ge_raw": float(np.mean(np.array(rows["isotonic"]["pr_auc"]) >= np.array(rows["raw"]["pr_auc"]) - 1e-12)) if used else None,
        "sigmoid_pr_auc_ge_raw": float(np.mean(np.array(rows["sigmoid"]["pr_auc"]) >= np.array(rows["raw"]["pr_auc"]) - 1e-12)) if used else None,
    }
    paired = {
        "brier_isotonic_minus_sigmoid": _summarize(list(np.array(rows["isotonic"]["brier"]) - np.array(rows["sigmoid"]["brier"]))) if used else _summarize([]),
        "pr_auc_isotonic_minus_raw": _summarize(list(np.array(rows["isotonic"]["pr_auc"]) - np.array(rows["raw"]["pr_auc"]))) if used else _summarize([]),
        "pr_auc_sigmoid_minus_raw": _summarize(list(np.array(rows["sigmoid"]["pr_auc"]) - np.array(rows["raw"]["pr_auc"]))) if used else _summarize([]),
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
        "note": "Calibrators are refit on each validation subset and scored on the held-out subset. Frozen chronological TEST is unused.",
    }


def kfold_oof(raw, y, n_splits: int, seed: int) -> dict:
    raw, y = _as_xy(raw, y)
    n_splits = min(n_splits, max(int(y.sum()), 2))
    if n_splits < 2:
        return {"skipped": True, "reason": "not enough positives for k-fold"}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = {m: np.zeros(len(y), dtype=float) for m in METHODS}
    fold_rows: dict[str, dict[str, list]] = {m: {"brier": [], "log_loss": [], "ece_uniform_10": [], "pr_auc": []} for m in METHODS}
    for fit_idx, eval_idx in skf.split(raw, y):
        raw_fit, y_fit = raw[fit_idx], y[fit_idx]
        raw_eval, y_eval = raw[eval_idx], y[eval_idx]
        for method in METHODS:
            p_eval = apply_method(raw_eval, y_fit, raw_fit, method)
            oof[method][eval_idx] = p_eval
            bundle = score_bundle(y_eval, p_eval, method)
            for key in ("brier", "log_loss", "ece_uniform_10", "pr_auc"):
                fold_rows[method][key].append(bundle[key])
    pooled = {m: score_bundle(y, oof[m], m, include_reliability=True) for m in METHODS}
    for m in pooled:
        bins = (pooled[m].get("reliability_uniform") or {}).get("bins")
        pooled[m]["reliability_uniform"] = {"ece": pooled[m]["ece_uniform_10"], "bins": bins}
    return {
        "protocol": "stratified_kfold_oof_on_validation",
        "n_splits": n_splits,
        "seed": seed,
        "fold_summaries": {m: {k: _summarize(v) for k, v in fold_rows[m].items()} for m in METHODS},
        "pooled_oof": pooled,
        "note": "Each validation row is scored by a calibrator that did not see that row. Frozen TEST unused.",
    }


def train_fit_val_eval(raw_train, y_train, raw_val, y_val) -> dict:
    raw_train, y_train = _as_xy(raw_train, y_train)
    raw_val, y_val = _as_xy(raw_val, y_val)
    out = {}
    for method in METHODS:
        p_val = apply_method(raw_val, y_train, raw_train, method)
        bundle = score_bundle(y_val, p_val, method)
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
        "note": "Pre-test only. Raw train scores are booster-in-sample. Frozen TEST unused.",
    }


def temporal_pretest_holdout(raw_pre, y_pre, times, cut_frac: float, seed: int = 42) -> dict:
    """Fit calibrators on an earlier pretest slice; evaluate on a later pretest slice. No TEST rows."""
    raw_pre, y_pre = _as_xy(raw_pre, y_pre)
    times = np.asarray(times)
    order = np.argsort(times, kind="mergesort")
    n = len(order)
    cut = int(cut_frac * n)
    cut = min(max(cut, 1), n - 1)
    t_cut = int(times[order[cut - 1]])
    while cut < n and int(times[order[cut]]) <= t_cut:
        cut += 1
    if cut <= 0 or cut >= n:
        return {"skipped": True, "reason": "could not form a strict temporal pretest cut"}
    fit_idx, eval_idx = order[:cut], order[cut:]
    y_fit, y_eval = y_pre[fit_idx], y_pre[eval_idx]
    if y_fit.sum() < 2 or y_eval.sum() < 2 or len(np.unique(y_fit)) < 2:
        return {"skipped": True, "reason": "pretest temporal cut missing a class"}
    raw_fit, raw_eval = raw_pre[fit_idx], raw_pre[eval_idx]
    methods = {}
    for method in METHODS:
        p_eval = apply_method(raw_eval, y_fit, raw_fit, method)
        methods[method] = score_bundle(y_eval, p_eval, method)
        if method == "isotonic":
            methods[method]["staircase"] = staircase_diagnostics(y_eval, raw_eval, p_eval)
    winner_brier = min(METHODS, key=lambda m: methods[m]["brier"])
    eval_time_min = int(np.min(times[eval_idx]))
    return {
        "protocol": "temporal_cut_on_pretest_train_plus_validation",
        "cut_frac": cut_frac,
        "seed": seed,
        "fit_n": int(len(y_fit)),
        "fit_positives": int(y_fit.sum()),
        "eval_n": int(len(y_eval)),
        "eval_positives": int(y_eval.sum()),
        "fit_time_max": t_cut,
        "eval_time_min": eval_time_min,
        "strict_temporal": bool(t_cut < eval_time_min),
        "methods": methods,
        "lowest_brier_method": winner_brier,
        "note": "Different temporal boundary from the official 70/15 val cut. TEST unused.",
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
    """Uncertainty of metrics for already-fitted maps. Does not refit calibrators. No TEST."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    draws = {m: {"brier": [], "log_loss": [], "ece_uniform_10": [], "pr_auc": []} for m in maps}
    for _ in range(n_boot):
        idx = stratified_bootstrap_indices(y, rng)
        yb = y[idx]
        for name, p in maps.items():
            pb = clip_proba(p)[idx]
            draws[name]["brier"].append(float(brier_score_loss(yb, pb)))
            draws[name]["log_loss"].append(float(log_loss(yb, np.clip(pb, EPS, 1 - EPS), labels=[0, 1])))
            draws[name]["ece_uniform_10"].append(reliability_bins(yb, pb, n_bins=10, strategy="uniform")["ece"])
            draws[name]["pr_auc"].append(float(average_precision_score(yb, pb)) if yb.sum() else 0.0)
    out = {name: {metric: _summarize(vals) for metric, vals in draws[name].items()} for name in maps}
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


def recommend(payload: dict) -> dict:
    """Map robustness evidence to KEEP_ISOTONIC / PREFER_SIGMOID / INCONCLUSIVE_KEEP_CURRENT."""
    nested = payload["nested_holdout_validation"]
    oof = payload["kfold_oof_validation"]
    train_hold = payload["train_fit_val_eval"]
    stair = payload["staircase_in_sample_validation"]
    in_sample = payload["in_sample_validation"]
    temporal = payload.get("temporal_pretest_holdouts") or []

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

    nested_iso_unique = nested["methods"]["isotonic"]["n_unique"]["mean"]
    nested_sig_brier = nested["methods"]["sigmoid"]["brier"]["mean"]
    nested_iso_brier = nested["methods"]["isotonic"]["brier"]["mean"]
    nested_raw_brier = nested["methods"]["raw"]["brier"]["mean"]
    nested_iso_pr = nested["methods"]["isotonic"]["pr_auc"]["mean"]
    nested_sig_pr = nested["methods"]["sigmoid"]["pr_auc"]["mean"]
    nested_raw_pr = nested["methods"]["raw"]["pr_auc"]["mean"]
    iso_brier_win_vs_sig = nested["win_rates"]["isotonic_brier_lt_sigmoid"]

    train_iso_brier = train_hold["methods"]["isotonic"]["brier"]
    train_sig_brier = train_hold["methods"]["sigmoid"]["brier"]
    train_iso_pr = train_hold["methods"]["isotonic"]["pr_auc"]
    train_sig_pr = train_hold["methods"]["sigmoid"]["pr_auc"]
    train_raw_pr = train_hold["methods"]["raw"]["pr_auc"]

    ranking_hurt = (
        nested_pr_diff.get("mean") is not None and nested_pr_diff["mean"] < -0.005
    ) or (oof_iso_pr is not None and oof_raw_pr is not None and (oof_iso_pr - oof_raw_pr) < -0.005)
    staircase_cause = bool(stair.get("drop_explained_by_ties"))
    degenerate = (oof_iso_nuniq is not None and oof_iso_nuniq < 20) or (
        nested_iso_unique is not None and nested_iso_unique < 20
    )
    heavy_compression = bool(stair.get("unique_ratio") is not None and stair["unique_ratio"] < 0.01)
    nested_iso_beats_sig = (
        iso_brier_win_vs_sig is not None
        and iso_brier_win_vs_sig >= 0.8
        and nested_brier_diff.get("p975") is not None
        and nested_brier_diff["p975"] < 0
    )
    oof_iso_beats_sig = oof_iso is not None and oof_sig is not None and oof_iso < oof_sig
    train_iso_beats_sig = train_iso_brier < train_sig_brier
    sigmoid_preserves_ranking = nested_sig_pr is not None and nested_raw_pr is not None and nested_sig_pr >= nested_raw_pr - 0.005
    temporal_winners = [h.get("lowest_brier_method") for h in temporal if not h.get("skipped")]
    winner_changes = len(set(temporal_winners)) > 1 if temporal_winners else False
    temporal_conflicts_iso = bool(temporal_winners) and any(w != "isotonic" for w in temporal_winners)

    evidence = [
        (
            f"In-sample validation Brier: raw={in_sample['raw']['brier']:.6f}, "
            f"sigmoid={in_sample['sigmoid']['brier']:.6f}, isotonic={in_sample['isotonic']['brier']:.6f} "
            f"(isotonic unique p={in_sample['isotonic']['n_unique_predictions']}; in-sample advantage)."
        ),
        (
            f"Nested val holdout mean Brier: raw={nested_raw_brier}, sigmoid={nested_sig_brier}, "
            f"isotonic={nested_iso_brier}; isotonic<sigmoid in "
            f"{'n/a' if iso_brier_win_vs_sig is None else f'{iso_brier_win_vs_sig:.0%}'} of repeats."
        ),
        f"Nested mean PR-AUC: raw={nested_raw_pr}, sigmoid={nested_sig_pr}, isotonic={nested_iso_pr}.",
    ]
    if oof_pool:
        evidence.append(
            f"5-fold OOF pooled Brier raw={oof_raw}, sigmoid={oof_sig}, isotonic={oof_iso}; "
            f"PR-AUC raw={oof_raw_pr}, sigmoid={oof_sig_pr}, isotonic={oof_iso_pr}; "
            f"isotonic unique p={oof_iso_nuniq}."
        )
    evidence.append(
        f"Train-fit/val-eval Brier isotonic={train_iso_brier:.6f} sigmoid={train_sig_brier:.6f}; "
        f"PR-AUC raw={train_raw_pr:.4f} sigmoid={train_sig_pr:.4f} isotonic={train_iso_pr:.4f}."
    )
    evidence.append(
        f"Staircase: unique {stair.get('n_unique_raw')}→{stair.get('n_unique_calibrated')} "
        f"ratio={stair.get('unique_ratio')}, monotone={stair.get('monotone_nondecreasing_in_raw')}, "
        f"PR-AUC drop={stair.get('pr_auc_drop')}, after tie-break={stair.get('pr_auc_drop_after_tiebreak')}."
    )
    if temporal_winners:
        evidence.append(f"Temporal pretest holdout lowest-Brier methods: {temporal_winners}.")

    rationale = []
    brier_agrees_iso = nested_iso_beats_sig and oof_iso_beats_sig and train_iso_beats_sig
    ranking_ok = not ranking_hurt

    if brier_agrees_iso and ranking_ok and not degenerate and not winner_changes and not temporal_conflicts_iso:
        decision = "KEEP_ISOTONIC"
        rationale.append("Nested, OOF, and train-holdout Brier agree on isotonic without a material ranking cost.")
    elif (ranking_hurt or degenerate) and not nested_iso_beats_sig:
        decision = "PREFER_SIGMOID"
        rationale.append("Isotonic does not win nested Brier robustly and harms ranking or uniqueness.")
    elif ranking_hurt and nested_iso_beats_sig:
        decision = "INCONCLUSIVE_KEEP_CURRENT"
        rationale.append(
            "Nested Brier prefers isotonic; ranking/uniqueness prefer sigmoid. "
            "Evidence is mixed, so the current isotonic selection is kept."
        )
    elif winner_changes or (brier_agrees_iso and temporal_conflicts_iso):
        decision = "INCONCLUSIVE_KEEP_CURRENT"
        rationale.append("Temporal pretest holdouts disagree with each other or with nested Brier. Do not force a change.")
    elif heavy_compression and ranking_hurt:
        decision = "PREFER_SIGMOID"
        rationale.append("Severe isotonic plateau compression explains a PR-AUC drop; sigmoid preserves ranking.")
    else:
        decision = "INCONCLUSIVE_KEEP_CURRENT"
        rationale.append("Robustness signals are mixed or weak. Keep the current isotonic calibration.")

    if degenerate:
        rationale.append(f"Isotonic uniqueness is degenerate (nested mean unique p={nested_iso_unique}, OOF unique={oof_iso_nuniq}).")
    if staircase_cause:
        rationale.append("In-sample PR-AUC drop is explained by plateau ties, not rank reversal.")
    if sigmoid_preserves_ranking:
        rationale.append("Sigmoid nested PR-AUC matches raw (monotone map).")

    return {
        "decision": decision,
        "recommended_calibration_method": (
            "isotonic" if decision == "KEEP_ISOTONIC" else ("sigmoid" if decision == "PREFER_SIGMOID" else "isotonic (current, inconclusive)")
        ),
        "current_phase9_selection": "isotonic",
        "test_used_for_decision": False,
        "evidence": evidence,
        "rationale": rationale,
        "checks": {
            "nested_isotonic_brier_beats_sigmoid": nested_iso_beats_sig,
            "oof_isotonic_brier_beats_sigmoid": oof_iso_beats_sig,
            "train_holdout_isotonic_brier_beats_sigmoid": train_iso_beats_sig,
            "isotonic_ranking_hurt": ranking_hurt,
            "staircase_explains_pr_auc_drop": staircase_cause,
            "isotonic_degenerate_unique_values": degenerate,
            "heavy_probability_compression": heavy_compression,
            "sigmoid_preserves_ranking": sigmoid_preserves_ranking,
            "temporal_winner_changes": winner_changes,
            "temporal_conflicts_with_isotonic": temporal_conflicts_iso,
        },
        "allowed_decisions": list(DECISIONS),
    }
