"""ULB probability + threshold calibration workflow.

TRAIN → existing fitted booster (not refit here)
VALIDATION → calibrator fit, method selection, threshold/cost optimization
TEST → single evaluation of the frozen choice
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from ml.ulb.adapter import ULBFraudDatasetAdapter, _relpath
from ml.ulb.model import preload_openmp
from ml.ulb.calibration import (
    calibration_diagnostics,
    clip_proba,
    fit_calibrators,
    reliability_svg,
    select_calibration_method,
)
from ml.ulb.constants import (
    COST_SCENARIOS,
    DATASET_ID,
    DATASET_NAME,
    DEFAULT_COST_SCENARIO,
    MODEL_VERSION,
    TARGET_COLUMN,
    TRACK,
)
from ml.ulb.decisions import (
    CostConfig,
    expected_cost_three_way,
    pick_prototype_operating_point,
    sweep_binary_thresholds,
    three_way_decision,
)
from ml.ulb.metrics import classification_metrics
from ml.ulb.registry import MODEL_REGISTRY
from ml.ulb.split import chronological_split, split_summary
from ml.ulb.calibration_report import render_calibration_report


def _py(obj):
    if isinstance(obj, dict):
        return {k: _py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_py(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _assert_matches_saved_split(current: dict, saved: dict | None) -> None:
    if not saved:
        return
    for part in ("train", "val", "test"):
        for key in ("n", "fraud"):
            if saved.get(part, {}).get(key) != current.get(part, {}).get(key):
                raise RuntimeError(
                    f"Official {part}.{key} changed: saved={saved[part].get(key)} "
                    f"current={current[part].get(key)}. Calibration must not alter the split."
                )


def _score_pack(y, p, cutoff: float) -> dict:
    pred = (clip_proba(p) >= cutoff).astype(int)
    metrics = classification_metrics(y, p, pred)
    metrics["cutoff"] = float(cutoff)
    metrics["cutoff_note"] = (
        "Binary metrics at this cutoff. Ranking metrics (PR-AUC, ROC-AUC) are threshold-free."
    )
    return metrics


def run_ulb_calibration(
    adapter: ULBFraudDatasetAdapter | None = None,
    train_if_missing: bool = True,
) -> dict:
    adapter = adapter or ULBFraudDatasetAdapter()
    adapter.eval_dir = Path(adapter.eval_dir)
    adapter.model_dir = Path(adapter.model_dir)

    model_path = adapter.model_dir / "model.joblib"
    if not model_path.exists():
        if not train_if_missing:
            raise FileNotFoundError(f"ULB model missing at {model_path}")
        adapter.run_full()

    cleaned = adapter._cleaned if adapter._cleaned is not None else adapter.preprocess()
    train, val, test = chronological_split(cleaned)
    official = split_summary(train, val, test, "chronological")

    saved_metrics_path = adapter.eval_dir / "ulb_metrics.json"
    saved_split = None
    if saved_metrics_path.exists():
        saved = json.loads(saved_metrics_path.read_text())
        saved_split = saved.get("official_split")
        _assert_matches_saved_split(official, saved_split)

    preload_openmp()
    model, transformer = adapter.load_model()
    meta_path = adapter.model_dir / "metadata.json"
    booster_version = MODEL_VERSION
    if meta_path.exists():
        booster_version = json.loads(meta_path.read_text()).get("model_version") or MODEL_VERSION
    calibrated_version = f"{booster_version}-calibrated"
    y_val = val[TARGET_COLUMN].to_numpy(dtype=int)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=int)
    raw_val = clip_proba(model.predict_proba(transformer.transform(val))[:, 1])
    raw_test = clip_proba(model.predict_proba(transformer.transform(test))[:, 1])

    fitted = fit_calibrators(raw_val, y_val)
    if fitted.test_labels_used or fitted.fit_n != len(y_val):
        raise RuntimeError("Calibrator fit invariant violated")

    methods = {
        "raw": raw_val,
        "sigmoid": fitted.transform(raw_val, "sigmoid"),
        "isotonic": fitted.transform(raw_val, "isotonic"),
    }
    val_diag = {
        name: calibration_diagnostics(y_val, prob, name) for name, prob in methods.items()
    }
    selection = select_calibration_method(val_diag)
    selected = selection["selected_method"]
    cal_val = methods[selected]
    cal_test = fitted.transform(raw_test, selected)

    if not np.all((cal_val >= 0) & (cal_val <= 1)) or not np.all((cal_test >= 0) & (cal_test <= 1)):
        raise RuntimeError("Calibrated probabilities escaped [0, 1]")

    binary_table = sweep_binary_thresholds(y_val, cal_val)
    best_f1_row = max(binary_table, key=lambda r: r["f1"])

    cost_results = {
        sid: pick_prototype_operating_point(y_val, cal_val, sid) for sid in COST_SCENARIOS
    }
    prototype = cost_results[DEFAULT_COST_SCENARIO]
    t_review = float(prototype["selected"]["t_review"])
    t_block = float(prototype["selected"]["t_block"])
    if not (t_review < t_block):
        raise RuntimeError("Selected thresholds are not strictly ordered")

    test_raw_diag = calibration_diagnostics(y_test, raw_test, "raw")
    test_cal_diag = calibration_diagnostics(y_test, cal_test, selected)
    test_raw_half = _score_pack(y_test, raw_test, 0.5)
    test_at_half = _score_pack(y_test, cal_test, 0.5)
    test_at_block = _score_pack(y_test, cal_test, t_block)
    test_three_way = expected_cost_three_way(
        y_test, cal_test, t_review, t_block, CostConfig.from_scenario(DEFAULT_COST_SCENARIO)
    )
    test_decisions = three_way_decision(cal_test, t_review, t_block)

    payload = {
        "label": "PROTOTYPE CALIBRATION",
        "not_industry_standard": True,
        "track": TRACK,
        "dataset_id": DATASET_ID,
        "dataset": DATASET_NAME,
        "booster_model_version": booster_version,
        "calibrated_model_version": calibrated_version,
        "synthetic_model_untouched": True,
        "incompatible_with_product_pipeline": True,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "raw_path": _relpath(adapter.csv_path),
        "official_split": official,
        "split_matches_ulb_metrics": saved_split is not None,
        "methodology": {
            "train": "Existing ulb-xgb-v1 booster is reused; calibration does not refit it.",
            "validation": "Platt and isotonic fit on validation raw scores; method and thresholds selected here.",
            "test": "Single evaluation of the frozen calibrator and frozen thresholds.",
            "test_used_for_calibrator_fit": False,
            "test_used_for_method_selection": False,
            "test_used_for_threshold_selection": False,
            "calibrator_fit_n": fitted.fit_n,
            "calibrator_fit_n_positive": fitted.fit_n_positive,
            "validation_n": int(len(y_val)),
            "test_n": int(len(y_test)),
        },
        "signal_definitions": {
            "raw_probability": "Uncalibrated booster P(Class=1 | ULB features). Not a product risk score.",
            "calibrated_probability": "Validation-fitted map of raw_probability. Offline P(Class=1) estimate.",
            "risk_score": "calibrated_probability * 100. A scaled score, not a probability.",
            "behavior_score": "Synthetic Isolation Forest + personalized overlays. Not computed on ULB.",
            "rule_score": "Deterministic product rules. Not computed on ULB.",
            "graph_score": "Device/IP/user relationship score. Not computed on ULB.",
            "final_risk_score": "Weighted product combination. Not equal to calibrated_probability.",
        },
        "validation": {
            "raw": val_diag["raw"],
            "sigmoid": val_diag["sigmoid"],
            "isotonic": val_diag["isotonic"],
            "selection": selection,
        },
        "selected_method": selected,
        "selection_justification": selection["justification"],
        "binary_threshold_sweep_validation": binary_table,
        "best_f1_threshold_validation": best_f1_row,
        "cost_scenarios": cost_results,
        "prototype_operating_thresholds": {
            "approve_below": t_review,
            "review_from": t_review,
            "review_to": t_block,
            "block_above": t_block,
            "cost_scenario": DEFAULT_COST_SCENARIO,
            "cost_assumptions": prototype["costs"],
            "source": prototype["selected_source"],
            "notes": prototype["notes"],
            "validation_mix": {
                "approve_rate": prototype["selected"].get("approve_rate"),
                "review_rate": prototype["selected"].get("review_rate"),
                "block_rate": prototype["selected"].get("block_rate"),
                "expected_cost_per_txn": prototype["selected"].get("expected_cost_per_txn"),
                "fraud_catch_rate": prototype["selected"].get("fraud_catch_rate"),
            },
        },
        "test_evaluation": {
            "note": "Evaluated once after freezing the validation choices. Not used to pick a method.",
            "raw_probability": {
                "brier": test_raw_diag["brier"],
                "log_loss": test_raw_diag["log_loss"],
                "ece_uniform_10": test_raw_diag["ece_uniform_10"],
                "pr_auc": test_raw_half["pr_auc"],
                "roc_auc": test_raw_half["roc_auc"],
                "precision": test_raw_half["precision"],
                "recall": test_raw_half["recall"],
                "f1": test_raw_half["f1"],
                "false_positive_rate": test_raw_half["false_positive_rate"],
                "false_negative_rate": test_raw_half["false_negative_rate"],
            },
            "calibrated_probability": {
                "method": selected,
                "brier": test_cal_diag["brier"],
                "log_loss": test_cal_diag["log_loss"],
                "ece_uniform_10": test_cal_diag["ece_uniform_10"],
                "mean_predicted": test_cal_diag["mean_predicted"],
                "within_unit_interval": test_cal_diag["within_unit_interval"],
            },
            "risk_score": {
                "definition": "calibrated_probability * 100",
                "is_probability": False,
                "mean": float(cal_test.mean() * 100.0),
            },
            "metrics_at_0_5": test_at_half,
            "metrics_at_prototype_t_block": test_at_block,
            "three_way_at_prototype_thresholds": test_three_way,
            "decision_counts": {
                "APPROVE": int((test_decisions == "APPROVE").sum()),
                "REVIEW": int((test_decisions == "REVIEW").sum()),
                "BLOCK": int((test_decisions == "BLOCK").sum()),
            },
            "reliability_uniform": test_cal_diag["reliability_uniform"],
        },
        "model_registry": MODEL_REGISTRY,
        "limitations": [
            "ULB has no user/device/IP/merchant fields; calibrated P(Class=1) is not a production fraud probability.",
            "Isotonic validation ECE near zero is in-sample: the map is fit on those same labels.",
            "Validation has few fraud cases (55); isotonic can overfit and ECE is noisy.",
            "Isotonic may collapse ranking (fewer unique scores) even when Brier improves.",
            "Brier score is dominated by the negative class at ~0.12% prevalence.",
            "Cost weights are prototype configuration, not empirically estimated loss given default.",
            "A 5% non-approve cap is an operations constraint, not a statistically identified constant.",
            "Product APPROVE/REVIEW/BLOCK still uses env THRESHOLD_REVIEW/BLOCK on final_risk_score.",
            "xgb-iforest-v1-calibrated remains the live model; ulb-xgb-v1-calibrated is offline only.",
        ],
    }

    adapter.eval_dir.mkdir(parents=True, exist_ok=True)
    figures = adapter.eval_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for name, diag in val_diag.items():
        svg = reliability_svg(diag["reliability_uniform"]["bins"], f"Validation reliability — {name}")
        (figures / f"ulb_reliability_val_{name}.svg").write_text(svg)
    test_svg = reliability_svg(
        test_cal_diag["reliability_uniform"]["bins"],
        f"Test reliability — {selected} (once, not used for selection)",
    )
    (figures / "ulb_reliability_test_selected.svg").write_text(test_svg)

    clean_payload = _py(payload)
    (adapter.eval_dir / "calibration_metrics.json").write_text(json.dumps(clean_payload, indent=2))
    (adapter.eval_dir / "calibration_report.md").write_text(render_calibration_report(clean_payload))

    adapter.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "selected_method": selected,
            "sigmoid": fitted.sigmoid,
            "isotonic": fitted.isotonic,
            "fit_n": fitted.fit_n,
            "fit_n_positive": fitted.fit_n_positive,
            "test_labels_used": False,
            "t_review": t_review,
            "t_block": t_block,
            "booster_model_version": booster_version,
            "calibrated_model_version": calibrated_version,
        },
        adapter.model_dir / "probability_calibrator.joblib",
    )
    meta_path = adapter.model_dir / "calibration_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "booster_model_version": booster_version,
                "calibrated_model_version": calibrated_version,
                "selected_method": selected,
                "t_review": t_review,
                "t_block": t_block,
                "test_labels_used": False,
                "synthetic_model_untouched": True,
            },
            indent=2,
        )
    )
    return clean_payload
