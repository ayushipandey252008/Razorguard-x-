"""IEEE-CIS metrics, calibration, thresholds, and SHAP. Test labels never fit anything."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.ml.ieee.constants import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, COST_REVIEW

EPS = 1e-15


def clip_proba(p) -> np.ndarray:
    arr = np.asarray(p, dtype=float).reshape(-1)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = clip_proba(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    pr = float(average_precision_score(y_true, y_prob)) if y_true.sum() else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc,
        "pr_auc": pr,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "fraud_catch_rate": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, EPS, 1 - EPS), labels=[0, 1])),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
        "n_legitimate": int((y_true == 0).sum()),
        "fraud_prevalence": float(y_true.mean()) if len(y_true) else None,
    }


def reliability_bins(y_true, y_prob, n_bins: int = 10, strategy: str = "uniform") -> dict:
    y = np.asarray(y_true).astype(int)
    p = clip_proba(y_prob)
    n = len(y)
    if n == 0:
        return {"strategy": strategy, "n_bins": n_bins, "ece": None, "bins": []}
    if strategy == "quantile":
        edges = np.unique(np.round(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)), 12))
        edges[0], edges[-1] = 0.0, 1.0
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (p >= lo) & (p <= hi) if i == len(edges) - 2 else (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin": i, "lo": lo, "hi": hi, "count": 0, "n_positive": 0, "mean_predicted": None, "empirical_positive_rate": None, "gap": None})
            continue
        mean_pred = float(p[mask].mean())
        emp = float(y[mask].mean())
        gap = abs(emp - mean_pred)
        ece += (count / n) * gap
        bins.append(
            {
                "bin": i,
                "lo": lo,
                "hi": hi,
                "count": count,
                "n_positive": int(y[mask].sum()),
                "mean_predicted": mean_pred,
                "empirical_positive_rate": emp,
                "gap": gap,
            }
        )
    return {"strategy": strategy, "n_bins": len(edges) - 1, "ece": float(ece), "bins": bins}


def calibration_diagnostics(y_true, y_prob, label: str) -> dict:
    y = np.asarray(y_true).astype(int)
    p = clip_proba(y_prob)
    uniform = reliability_bins(y, p, n_bins=10, strategy="uniform")
    quantile = reliability_bins(y, p, n_bins=10, strategy="quantile")
    return {
        "label": label,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
        "ece_uniform_10": uniform["ece"],
        "ece_quantile_10": quantile["ece"],
        "mean_predicted": float(p.mean()) if len(p) else None,
        "empirical_prevalence": float(y.mean()) if len(y) else None,
        "n_unique_predictions": int(np.unique(np.round(p, 12)).size),
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "reliability_uniform": uniform,
        "within_unit_interval": bool(np.all((p >= 0.0) & (p <= 1.0))),
    }


@dataclass
class FittedCalibrators:
    sigmoid: LogisticRegression | None
    isotonic: IsotonicRegression | None
    fit_n: int
    fit_n_positive: int
    test_labels_used: bool = False

    def transform(self, raw_prob, method: str) -> np.ndarray:
        raw = clip_proba(raw_prob)
        if method == "raw" or (method == "sigmoid" and self.sigmoid is None) or (method == "isotonic" and self.isotonic is None):
            return raw
        if method == "sigmoid":
            return clip_proba(self.sigmoid.predict_proba(raw.reshape(-1, 1))[:, 1])
        if method == "isotonic":
            return clip_proba(self.isotonic.predict(raw))
        raise ValueError(f"Unknown calibration method: {method}")


def fit_calibrators(raw_val, y_val) -> FittedCalibrators:
    raw = clip_proba(raw_val)
    y = np.asarray(y_val).astype(int)
    if len(np.unique(y)) < 2:
        return FittedCalibrators(None, None, int(len(y)), int(y.sum()), False)
    sigmoid = LogisticRegression(solver="lbfgs", max_iter=1000)
    sigmoid.fit(raw.reshape(-1, 1), y)
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    isotonic.fit(raw, y)
    return FittedCalibrators(sigmoid, isotonic, int(len(y)), int(y.sum()), False)


def select_calibration_method(val_diagnostics: dict) -> dict:
    """ULB lesson: do not pick isotonic only because in-sample Brier is lower."""
    order = ["raw", "sigmoid", "isotonic"]
    rows = []
    for name in order:
        d = val_diagnostics[name]
        rows.append(
            {
                "method": name,
                "brier": d["brier"],
                "log_loss": d["log_loss"],
                "ece_uniform_10": d["ece_uniform_10"],
                "n_unique_predictions": d["n_unique_predictions"],
            }
        )
    ranked = sorted(
        rows,
        key=lambda r: (
            r["brier"],
            r["ece_uniform_10"] if r["ece_uniform_10"] is not None else 1.0,
            r["log_loss"],
        ),
    )
    selected = ranked[0]["method"]
    notes = [
        "Selection uses validation Brier, then ECE, then log loss. Test labels are unused.",
        "Isotonic is fit on the same validation scores it is scored on, so a lower val Brier is not proof of robustness.",
        "ULB calibration showed isotonic can compress ranking (PR-AUC) while improving in-sample Brier.",
    ]
    iso = val_diagnostics["isotonic"]
    if selected == "isotonic" and iso["n_unique_predictions"] < 20:
        selected = "sigmoid" if val_diagnostics["sigmoid"]["n_unique_predictions"] >= 2 else "raw"
        notes.append(
            f"Isotonic had only {iso['n_unique_predictions']} distinct probabilities; sigmoid preferred for robustness."
        )
        ranked = [r for r in rows if r["method"] == selected] + [r for r in rows if r["method"] != selected]
    return {
        "selected_method": selected,
        "ranking": ranked,
        "justification": (
            f"Selected {selected} using validation diagnostics with an isotonic uniqueness guard "
            f"(Brier={val_diagnostics[selected]['brier']:.8f})."
        ),
        "notes": notes,
        "model_probability_is_not_final_risk_score": True,
        "not_production_fraud_probability": True,
    }


def select_three_way_thresholds(
    y_val,
    p_val,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
    cost_review: float = COST_REVIEW,
) -> dict:
    y = np.asarray(y_val).astype(int)
    p = clip_proba(p_val)
    grid = sorted(set(np.round(np.quantile(p, [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]), 6).tolist() + [0.2, 0.4, 0.5, 0.7, 0.9]))
    best = None
    for lo in grid:
        for hi in grid:
            if hi <= lo:
                continue
            cost = 0.0
            n_review = 0
            for yi, pi in zip(y, p):
                if pi < lo:
                    decision = "APPROVE"
                elif pi >= hi:
                    decision = "BLOCK"
                else:
                    decision = "REVIEW"
                    n_review += 1
                if decision == "APPROVE" and yi == 1:
                    cost += cost_fn
                elif decision == "BLOCK" and yi == 0:
                    cost += cost_fp
                elif decision == "REVIEW":
                    cost += cost_review
            rec = {"approve_below": float(lo), "block_above": float(hi), "review_from": float(lo), "review_to": float(hi), "val_cost": float(cost), "val_n_review": int(n_review)}
            if best is None or rec["val_cost"] < best["val_cost"]:
                best = rec
    if best is None:
        best = {"approve_below": 0.2, "block_above": 0.8, "review_from": 0.2, "review_to": 0.8, "val_cost": None, "val_n_review": None}
    best["cost_assumptions"] = {
        "false_negative": cost_fn,
        "false_positive": cost_fp,
        "review": cost_review,
        "units": "relative prototype units, not a bank loss model",
    }
    best["source"] = "validation_only"
    best["notes"] = [
        "Thresholds are chosen on TRAIN/VALIDATION scores only and applied once to the frozen test set.",
        "model_probability is not the live product final_risk_score.",
        "Not an industry-standard operating point.",
    ]
    return best


def apply_policy(p, thresholds: dict) -> list[str]:
    lo = float(thresholds["approve_below"])
    hi = float(thresholds["block_above"])
    out = []
    for pi in clip_proba(p):
        if pi < lo:
            out.append("APPROVE")
        elif pi >= hi:
            out.append("BLOCK")
        else:
            out.append("REVIEW")
    return out


def policy_summary(y_true, p, thresholds: dict) -> dict:
    y = np.asarray(y_true).astype(int)
    decisions = apply_policy(p, thresholds)
    counts = {"APPROVE": 0, "REVIEW": 0, "BLOCK": 0}
    caught = 0
    for yi, d in zip(y, decisions):
        counts[d] += 1
        if yi == 1 and d in {"REVIEW", "BLOCK"}:
            caught += 1
    return {
        "decisions": counts,
        "fraud_catch_rate_review_or_block": float(caught / y.sum()) if y.sum() else None,
        "thresholds": thresholds,
    }


def shap_summary(clf, X, feature_names: list[str], family_of: dict[str, str], max_rows: int = 24) -> dict:
    try:
        import shap
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"shap import failed: {exc}"}
    n = min(max_rows, len(X))
    if n == 0:
        return {"available": False, "reason": "no rows"}
    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X[:n])
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        mean_abs = np.abs(np.asarray(sv)).mean(axis=0)
        order = np.argsort(-mean_abs)
        global_imp = [
            {"feature": feature_names[i] if i < len(feature_names) else str(i), "mean_abs_shap": float(mean_abs[i])}
            for i in order[:20]
        ]
        by_family: dict[str, float] = {}
        for i, mag in enumerate(mean_abs):
            name = feature_names[i] if i < len(feature_names) else str(i)
            fam = family_of.get(name, "other")
            by_family[fam] = by_family.get(fam, 0.0) + float(mag)
        example = []
        if n:
            row = np.asarray(sv[0]).reshape(-1)
            idx = np.argsort(-np.abs(row))[:8]
            example = [
                {
                    "feature": feature_names[i] if i < len(feature_names) else str(i),
                    "shap": float(row[i]),
                    "note": "Model explanation of this score, not a causal claim.",
                }
                for i in idx
            ]
        return {
            "available": True,
            "n_rows_explained": n,
            "global_feature_importance": global_imp,
            "importance_by_family": by_family,
            "example_transaction": example,
            "distinction": {
                "model_explanation": "SHAP attributes the model's score to input features.",
                "causal_explanation": "Not provided. Anonymized IEEE names are not business meanings unless the contest docs define them.",
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}
