"""Train a baseline gradient-boosted fraud classifier + Isolation Forest.

Synthetic training is isolated from the public-dataset evaluation track.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from app.config import REPO_ROOT, get_settings
from app.ml.booster import make_classifier
from app.ml.features import FEATURE_COLUMNS, dataframe_from_records
from app.services.synthetic import generate_labeled_dataset

RANDOM_SEED = 42
TRACK = "SYNTHETIC_DATASET"


def calibrate_scores(iso: IsotonicRegression, raw: np.ndarray) -> np.ndarray:
    return np.clip(iso.predict(raw), 0.0, 1.0)


def classification_metrics(y_true, y_prob, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def expected_cost(y_true, p, t_review, t_block, c_fp, c_fn, c_review) -> float:
    cost = 0.0
    for label, prob in zip(y_true, p):
        if prob >= t_block:
            decision = "BLOCK"
        elif prob >= t_review:
            decision = "REVIEW"
        else:
            decision = "APPROVE"
        if label == 1 and decision == "APPROVE":
            cost += c_fn
        elif label == 1 and decision == "REVIEW":
            cost += c_review
        elif label == 0 and decision == "BLOCK":
            cost += c_fp
        elif label == 0 and decision == "REVIEW":
            cost += c_review
    return float(cost)


def sweep_thresholds(y_true, p, c_fp, c_fn, c_review) -> dict:
    rows = []
    best = None
    for t_review in np.linspace(0.05, 0.6, 12):
        for t_block in np.linspace(max(t_review + 0.05, 0.2), 0.95, 12):
            flagged = p >= t_review
            blocked = p >= t_block
            fp = int(((y_true == 0) & blocked).sum())
            fn = int(((y_true == 1) & (p < t_review)).sum())
            prec = float(precision_score(y_true, flagged, zero_division=0))
            rec = float(recall_score(y_true, flagged, zero_division=0))
            cost = expected_cost(y_true, p, t_review, t_block, c_fp, c_fn, c_review)
            row = {
                "threshold_review": round(float(t_review), 3),
                "threshold_block": round(float(t_block), 3),
                "precision_at_review": prec,
                "recall_at_review": rec,
                "false_positives_block": fp,
                "false_negatives_approve": fn,
                "expected_cost": round(cost, 2),
            }
            rows.append(row)
            if best is None or cost < best["expected_cost"]:
                best = row
    rows.sort(key=lambda r: r["expected_cost"])
    constrained = [r for r in rows if r["threshold_review"] >= 0.25]
    return {
        "best": best,
        "best_with_review_at_least_0.25": constrained[0] if constrained else None,
        "top": rows[:12],
        "cost_units": {"fp": c_fp, "fn": c_fn, "review": c_review},
        "note": (
            "Unconstrained minimum cost often pushes review toward 0 because FN cost "
            "dominates. Product THRESHOLD_* stay separately configurable."
        ),
    }


def train_and_save(n_samples: int = 8000, model_dir: Path | None = None) -> dict:
    settings = get_settings()
    model_dir = Path(model_dir or settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    records = generate_labeled_dataset(n_samples, seed=RANDOM_SEED)
    X = dataframe_from_records(records)
    y = np.array([int(r["is_fraud"]) for r in records])
    duplicate_feature_rows = int(X.duplicated().sum())

    X_train, X_temp, y_train, y_temp = train_test_split(
        X.values, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    pos = max(int(y_train.sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = neg / pos

    clf, family, model_version = make_classifier(scale_pos_weight, RANDOM_SEED)
    if family == "xgboost":
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        clf.fit(X_train, y_train)

    # Prefit isotonic on validation scores. CalibratedClassifierCV(cv='prefit')
    # is deprecated and breaks on the XGBoost sklearn wrapper's tags in sklearn 1.6+.
    raw_val = clf.predict_proba(X_val)[:, 1]
    iso_cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_cal.fit(raw_val, y_val)

    iso = IsolationForest(
        n_estimators=100,
        contamination=min(0.12, max(0.02, float(y_train.mean()) + 0.02)),
        random_state=RANDOM_SEED,
        n_jobs=2,
    )
    iso.fit(X_train[y_train == 0] if (y_train == 0).any() else X_train)

    raw_test = clf.predict_proba(X_test)[:, 1]
    cal_test = calibrate_scores(iso_cal, raw_test)
    raw_metrics = classification_metrics(y_test, raw_test, (raw_test >= 0.5).astype(int))
    cal_metrics = classification_metrics(y_test, cal_test, (cal_test >= 0.5).astype(int))

    cal_val = calibrate_scores(iso_cal, clf.predict_proba(X_val)[:, 1])
    threshold_report = sweep_thresholds(
        y_val,
        cal_val,
        settings.cost_false_positive,
        settings.cost_false_negative,
        settings.cost_review,
    )

    metrics = {
        **cal_metrics,
        "track": TRACK,
        "model_version": model_version + "-calibrated",
        "booster_family": family,
        "scale_pos_weight": float(scale_pos_weight),
        "feature_columns": FEATURE_COLUMNS,
        "duplicate_feature_rows": duplicate_feature_rows,
        "split": {"train": 0.70, "val": 0.15, "test": 0.15, "strategy": "stratified_random", "seed": RANDOM_SEED},
        "probability_calibrated": True,
        "uncalibrated_test": raw_metrics,
        "calibrated_test": cal_metrics,
        "threshold_experiment": threshold_report["best"],
        "note": (
            "SYNTHETIC_DATASET track. Calibrated probabilities are isotonic regression "
            "on validation booster scores (prefit). Risk score is 0–100 from calibrated "
            "P(fraud). Threshold experiment is cost-based on validation, not industry-standard."
        ),
    }

    joblib.dump(clf, model_dir / "xgb_fraud.joblib")
    joblib.dump(iso_cal, model_dir / "calibrator.joblib")
    joblib.dump(iso, model_dir / "iforest.joblib")
    (model_dir / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (model_dir / "version.txt").write_text(metrics["model_version"])

    eval_dir = REPO_ROOT / "ml" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "latest_metrics.json").write_text(json.dumps(metrics, indent=2))
    (eval_dir / "synthetic_metrics.json").write_text(json.dumps(metrics, indent=2))
    (eval_dir / "threshold_calibration.json").write_text(json.dumps(threshold_report, indent=2))
    (eval_dir / "report.md").write_text(_render_report(metrics))
    (eval_dir / "threshold_calibration.md").write_text(_render_thresholds(threshold_report, metrics))
    return metrics


def _render_report(m: dict) -> str:
    cm = m["confusion_matrix"]
    raw = m.get("uncalibrated_test") or {}
    return f"""# SYNTHETIC_DATASET — evaluation

**Track: SYNTHETIC_DATASET.** Do not mix with REAL_DATASET metrics.

- Version: `{m["model_version"]}`
- Booster: `{m.get("booster_family")}`
- Split: 70/15/15 stratified, seed=42 (not a production time split)
- Test n={m["n_samples"]}, positive rate={m["positive_rate"]:.3f}
- Probabilities: isotonic regression on validation booster scores (prefit; not nested CV)

## Calibrated test metrics (0.5 operating point)

| Metric | Calibrated | Uncalibrated |
| --- | --- | --- |
| Precision | {m["precision"]:.3f} | {raw.get("precision", float("nan")):.3f} |
| Recall | {m["recall"]:.3f} | {raw.get("recall", float("nan")):.3f} |
| F1 | {m["f1"]:.3f} | {raw.get("f1", float("nan")):.3f} |
| ROC-AUC | {m["roc_auc"]:.3f} | {raw.get("roc_auc", float("nan")):.3f} |
| PR-AUC | {m["pr_auc"]:.3f} | {raw.get("pr_auc", float("nan")):.3f} |
| FPR | {m["false_positive_rate"]:.3f} | {raw.get("false_positive_rate", float("nan")):.3f} |
| Brier | {m["brier"]:.3f} | {raw.get("brier", float("nan")):.3f} |

Confusion (calibrated @0.5): TN={cm["tn"]} FP={cm["fp"]} FN={cm["fn"]} TP={cm["tp"]}

Accuracy is omitted as a primary metric.

The 0.5 cutoff is **not** the product decision threshold. Product APPROVE/REVIEW/BLOCK uses `THRESHOLD_*` on the 0–100 risk score.
"""


def _render_thresholds(report: dict, metrics: dict) -> str:
    best = report["best"] or {}
    costs = report["cost_units"]
    lines = [
        "# Cost-based threshold experiment (SYNTHETIC_DATASET, validation)",
        "",
        "Relative cost units, not currency and not industry-standard.",
        f"- FP (legit BLOCK): {costs['fp']}",
        f"- FN (fraud APPROVE): {costs['fn']}",
        f"- REVIEW (either class): {costs['review']}",
        "",
        f"Lowest-cost pair on **validation calibrated probabilities**: review={best.get('threshold_review')} "
        f"(≈ risk {float(best.get('threshold_review') or 0)*100:.0f}), "
        f"block={best.get('threshold_block')} (≈ risk {float(best.get('threshold_block') or 0)*100:.0f}), "
        f"expected cost={best.get('expected_cost')}.",
        "",
        "Unconstrained optimum often sets a very low review cutoff (almost everything is reviewed) "
        "because FN cost is 10× FP cost. Product defaults remain `THRESHOLD_REVIEW=40` / "
        "`THRESHOLD_BLOCK=70` so the demo still has an APPROVE band. That is a product choice, "
        "not the cost minimum and not industry-standard.",
        "",
        "| review_p | block_p | precision | recall | FP_block | FN_approve | cost |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["top"]:
        lines.append(
            f"| {row['threshold_review']:.3f} | {row['threshold_block']:.3f} | "
            f"{row['precision_at_review']:.3f} | {row['recall_at_review']:.3f} | "
            f"{row['false_positives_block']} | {row['false_negatives_approve']} | {row['expected_cost']:.1f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    result = train_and_save()
    print(json.dumps({k: result[k] for k in ("model_version", "precision", "recall", "f1", "roc_auc", "pr_auc", "track")}, indent=2))
