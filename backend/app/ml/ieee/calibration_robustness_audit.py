"""IEEE-CIS calibration robustness audit. Train/validation scores only.

Does not retrain XGBoost. Does not fit or select on the chronological TEST split.
Does not write live, ULB, or Phase 9 IEEE result artifacts.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from app.ml.booster import preload_openmp
from app.ml.ieee.adapter import ieee_files_present, join_transaction_identity, load_tables, resolve_data_dir
from app.ml.ieee.calibration_robustness import (
    METHODS,
    bootstrap_frozen_maps,
    kfold_oof,
    nested_holdout,
    recommend,
    score_bundle,
    staircase_diagnostics,
    temporal_pretest_holdout,
    train_fit_val_eval,
)
from app.ml.ieee.calibration_robustness_report import render_ieee_robustness_report
from app.ml.ieee.constants import (
    COMBINED_VERSION,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    COST_REVIEW,
    EVAL_DIR,
    IEEE_MODEL_DIR,
    LIVE_MODEL_DIR,
    LIVE_MODEL_VERSION,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRACK,
    ULB_METRICS_PATH,
    ULB_MODEL_DIR,
)
from app.ml.ieee.errors import IeeeDatasetError
from app.ml.ieee.evaluate import clip_proba, fit_calibrators, policy_summary
from app.ml.ieee.features import add_behavioral_features, add_transaction_timing
from app.ml.ieee.graph_features import add_graph_features
from app.ml.ieee.split import chronological_split, split_summary
from app.ml.ieee.train import predict_proba

EXPECTED_SPLIT = {
    "train": {"n": 413378, "fraud": 14538},
    "validation": {"n": 88581, "fraud": 3042},
    "test": {"n": 88581, "fraud": 3083},
}

HISTORICAL_FROZEN_TEST = {
    "quoted_not_recomputed": True,
    "source": "ml/evaluation/ieee_results.json frozen_test_metrics (Phase 9)",
    "calibrator_at_the_time": "isotonic",
    "binary_threshold": 0.5,
    "metrics": {
        "pr_auc": 0.3371524181307357,
        "roc_auc": 0.8688416207249846,
        "precision": 0.39339622641509436,
        "recall": 0.2705157314304249,
        "f1": 0.3205842783009802,
        "false_positive_rate": 0.015041287515497439,
        "false_negative_rate": 0.729484268569575,
        "confusion_matrix": {"tn": 84212, "fp": 1286, "fn": 2249, "tp": 834},
    },
    "note": (
        "Historical Phase 9 chronological TEST result. Quoted only. "
        "This audit did not overwrite ieee_results.json and did not use TEST to select a calibrator."
    ),
}

EXISTING_THRESHOLDS = {
    "approve_below": 0.020004,
    "block_above": 0.511628,
    "review_from": 0.020004,
    "review_to": 0.511628,
    "source": "validation_only_phase9_isotonic",
}

PROTECTED_REPO_PATHS = (
    LIVE_MODEL_DIR / "xgb_fraud.joblib",
    LIVE_MODEL_DIR / "version.txt",
    LIVE_MODEL_DIR / "calibrator.joblib",
    LIVE_MODEL_DIR / "iforest.joblib",
    LIVE_MODEL_DIR / "metrics.json",
    ULB_METRICS_PATH,
    EVAL_DIR / "calibration_metrics.json",
    EVAL_DIR / "calibration_robustness.json",
    EVAL_DIR / "calibration_robustness_report.md",
    ULB_MODEL_DIR / "model.joblib",
    EVAL_DIR / "ieee_results.json",
    EVAL_DIR / "ieee_results.md",
    EVAL_DIR / "ieee_experiment_manifest.json",
    EVAL_DIR / "ieee_data_audit.json",
    EVAL_DIR / "ieee_leakage_report.json",
    IEEE_MODEL_DIR / f"{COMBINED_VERSION}.joblib",
    IEEE_MODEL_DIR / f"{COMBINED_VERSION}.json",
    IEEE_MODEL_DIR / "ieee-xgb-baseline-v1.joblib",
    IEEE_MODEL_DIR / "ieee-xgb-baseline-v1.json",
    IEEE_MODEL_DIR / "ieee-xgb-graph-v1.joblib",
    IEEE_MODEL_DIR / "ieee-xgb-graph-v1.json",
)


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


def _file_fingerprint(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "nbytes": len(data)}


def _policy_cost(y, p, thresholds: dict) -> float:
    y = np.asarray(y).astype(int)
    p = clip_proba(p)
    lo = float(thresholds["approve_below"])
    hi = float(thresholds["block_above"])
    approve = p < lo
    block = p >= hi
    review = ~(approve | block)
    return float(
        (approve & (y == 1)).sum() * COST_FALSE_NEGATIVE
        + (block & (y == 0)).sum() * COST_FALSE_POSITIVE
        + review.sum() * COST_REVIEW
    )


def _select_three_way_thresholds_vectorized(y, p) -> dict:
    """Same grid/cost as evaluate.select_three_way_thresholds, fully vectorized."""
    y = np.asarray(y).astype(int)
    p = clip_proba(p)
    grid = sorted(
        set(np.round(np.quantile(p, [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]), 6).tolist() + [0.2, 0.4, 0.5, 0.7, 0.9])
    )
    best = None
    for lo in grid:
        for hi in grid:
            if hi <= lo:
                continue
            approve = p < lo
            block = p >= hi
            review = ~(approve | block)
            cost = float(
                (approve & (y == 1)).sum() * COST_FALSE_NEGATIVE
                + (block & (y == 0)).sum() * COST_FALSE_POSITIVE
                + review.sum() * COST_REVIEW
            )
            rec = {
                "approve_below": float(lo),
                "block_above": float(hi),
                "review_from": float(lo),
                "review_to": float(hi),
                "val_cost": cost,
                "val_n_review": int(review.sum()),
                "source": "validation_only",
            }
            if best is None or rec["val_cost"] < best["val_cost"]:
                best = rec
    if best is None:
        best = {
            "approve_below": 0.2,
            "block_above": 0.8,
            "review_from": 0.2,
            "review_to": 0.8,
            "val_cost": None,
            "val_n_review": None,
            "source": "validation_only",
        }
    best["cost_assumptions"] = {
        "false_negative": COST_FALSE_NEGATIVE,
        "false_positive": COST_FALSE_POSITIVE,
        "review": COST_REVIEW,
        "units": "relative prototype units, not a bank loss model",
    }
    return best


def _policy_block(y, p, thresholds: dict) -> dict:
    summary = policy_summary(y, p, thresholds)
    summary["val_cost"] = _policy_cost(y, p, thresholds)
    return summary


def build_payload_from_scores(
    *,
    raw_train,
    y_train,
    raw_val,
    y_val,
    times_pre,
    official_split: dict,
    n_boot: int = 200,
    n_nested: int = 40,
    nested_eval_frac: float = 0.30,
    n_folds: int = 5,
    seed: int = 42,
    temporal_cut_fracs: tuple[float, ...] = (0.50, 0.75),
) -> dict:
    """Robustness payload from pretest scores. Signatures contain no TEST arrays."""
    raw_train = clip_proba(raw_train)
    raw_val = clip_proba(raw_val)
    y_train = np.asarray(y_train).astype(int)
    y_val = np.asarray(y_val).astype(int)
    times_pre = np.asarray(times_pre)

    fitted = fit_calibrators(raw_val, y_val)
    maps = {
        "raw": raw_val,
        "sigmoid": fitted.transform(raw_val, "sigmoid"),
        "isotonic": fitted.transform(raw_val, "isotonic"),
    }
    in_sample = {m: score_bundle(y_val, maps[m], m, include_reliability=True) for m in METHODS}
    stair = staircase_diagnostics(y_val, raw_val, maps["isotonic"])
    frozen_boot = bootstrap_frozen_maps(y_val, maps, n_boot=n_boot, seed=seed)
    nested = nested_holdout(raw_val, y_val, n_splits=n_nested, test_size=nested_eval_frac, seed=seed)
    oof = kfold_oof(raw_val, y_val, n_splits=n_folds, seed=seed)
    train_hold = train_fit_val_eval(raw_train, y_train, raw_val, y_val)

    raw_pre = np.concatenate([raw_train, raw_val])
    y_pre = np.concatenate([y_train, y_val])
    if len(times_pre) != len(raw_pre):
        raise IeeeDatasetError("pretest times must align with concatenated train+val scores")
    temporal = [temporal_pretest_holdout(raw_pre, y_pre, times_pre, cut_frac=frac, seed=seed) for frac in temporal_cut_fracs]

    payload = {
        "label": "IEEE-CIS CALIBRATION ROBUSTNESS AUDIT",
        "track": TRACK,
        "booster_model_version": COMBINED_VERSION,
        "current_phase9_selection": "isotonic",
        "live_model_unchanged": LIVE_MODEL_VERSION,
        "ieee_status": "OFFLINE CANDIDATE",
        "methodology": {
            "xgboost_refit": False,
            "test_scored": False,
            "test_used_for_fit": False,
            "test_used_for_method_selection": False,
            "test_used_for_recommendation": False,
            "ieee_max_rows_used": False,
            "fixture_used": False,
            "train_n": int(len(y_train)),
            "train_positives": int(y_train.sum()),
            "validation_n": int(len(y_val)),
            "validation_positives": int(y_val.sum()),
        },
        "official_split": official_split,
        "in_sample_validation": in_sample,
        "staircase_in_sample_validation": stair,
        "bootstrap_frozen_maps": frozen_boot,
        "nested_holdout_validation": nested,
        "kfold_oof_validation": oof,
        "train_fit_val_eval": train_hold,
        "temporal_pretest_holdouts": temporal,
        "historical_phase9_frozen_test": HISTORICAL_FROZEN_TEST,
        "n_boot": n_boot,
        "n_nested": n_nested,
        "n_folds": n_folds,
        "seed": seed,
        "audited_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["recommendation"] = recommend(payload)

    operating = "isotonic"
    decision = payload["recommendation"]["decision"]
    if decision == "PREFER_SIGMOID":
        operating = "sigmoid"
    existing_thr = dict(EXISTING_THRESHOLDS)
    reselected = _select_three_way_thresholds_vectorized(y_val, maps[operating])
    payload["threshold_review"] = {
        "existing_phase9_thresholds": existing_thr,
        "existing_on_isotonic_val": _policy_block(y_val, maps["isotonic"], existing_thr),
        "existing_on_sigmoid_val": _policy_block(y_val, maps["sigmoid"], existing_thr),
        "reselected_for_operating_method": operating,
        "reselected_operating": {**_policy_block(y_val, maps[operating], reselected), "thresholds": reselected},
        "keep_existing_if_not_prefer_sigmoid": decision != "PREFER_SIGMOID",
        "note": (
            "Phase 9 cutpoints were chosen on isotonic validation scores. "
            "This review does not search TEST. If the decision is KEEP_ISOTONIC or "
            "INCONCLUSIVE_KEEP_CURRENT, the published 0.020004 / 0.511628 cutpoints remain."
        ),
    }
    return payload


def score_frozen_combined_pretest(
    *,
    data_dir: Path | str | None = None,
    model_dir: Path | str | None = None,
    eval_dir: Path | str | None = None,
) -> dict:
    """Rebuild pretest features and score the frozen combined candidate. TEST is dropped."""
    eval_path = Path(eval_dir or EVAL_DIR)
    model_path = Path(model_dir or IEEE_MODEL_DIR)
    data_root = resolve_data_dir(data_dir)
    if not ieee_files_present(data_root):
        raise IeeeDatasetError("IEEE-CIS CSVs are required for the official robustness audit. Fixtures are not used.")

    manifest_path = eval_path / "ieee_experiment_manifest.json"
    if not manifest_path.exists():
        raise IeeeDatasetError("ieee_experiment_manifest.json is required to verify the official split.")
    manifest = json.loads(manifest_path.read_text())
    saved_split = manifest.get("split") or {}

    artifact_path = model_path / f"{COMBINED_VERSION}.joblib"
    if not artifact_path.exists():
        raise IeeeDatasetError(f"Missing frozen candidate {artifact_path}")
    artifact = joblib.load(artifact_path)
    clf = artifact["clf"]
    preprocessor = artifact["preprocessor"]

    old_max = os.environ.pop("IEEE_MAX_ROWS", None)
    try:
        txn, ident, load_meta = load_tables(data_root, max_rows=None)
    finally:
        if old_max is not None:
            os.environ["IEEE_MAX_ROWS"] = old_max
    if load_meta.get("max_rows") is not None:
        raise IeeeDatasetError("IEEE_MAX_ROWS must not be used for this audit.")
    if len(txn) != 590540:
        raise IeeeDatasetError(f"Expected 590,540 transaction rows, got {len(txn)}")

    joined, _join = join_transaction_identity(txn, ident)
    del txn, ident
    gc.collect()
    if joined["TransactionID"].duplicated().any():
        joined = joined.drop_duplicates("TransactionID", keep="first")
    joined = add_transaction_timing(joined)
    train, val, test = chronological_split(joined)
    official = split_summary(train, val, test)
    for part in ("train", "validation", "test"):
        for key in ("n", "fraud"):
            got = official[part][key]
            expected = EXPECTED_SPLIT[part][key]
            saved = (saved_split.get(part) or {}).get(key)
            if got != expected or (saved is not None and got != saved):
                raise IeeeDatasetError(
                    f"Split mismatch on {part}.{key}: got {got}, expected {expected}, manifest {saved}"
                )
    del joined, test
    gc.collect()

    pretest = pd_concat_train_val(train, val)
    del train, val
    gc.collect()
    pretest = add_behavioral_features(pretest)
    pretest = add_graph_features(pretest)
    n_train = EXPECTED_SPLIT["train"]["n"]
    train = pretest.iloc[:n_train].reset_index(drop=True)
    val = pretest.iloc[n_train:].reset_index(drop=True)
    if len(val) != EXPECTED_SPLIT["validation"]["n"]:
        raise IeeeDatasetError("Pretest re-slice did not recover the official train/validation sizes.")
    del pretest
    gc.collect()

    preload_openmp()
    y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
    y_val = val[TARGET_COLUMN].to_numpy(dtype=int)
    times_pre = np.concatenate(
        [train[TIME_COLUMN].to_numpy(), val[TIME_COLUMN].to_numpy()]
    )
    raw_train = clip_proba(predict_proba(clf, preprocessor.transform(train)))
    raw_val = clip_proba(predict_proba(clf, preprocessor.transform(val)))
    del train, val, clf, preprocessor, artifact
    gc.collect()
    return {
        "raw_train": raw_train,
        "y_train": y_train,
        "raw_val": raw_val,
        "y_val": y_val,
        "times_pre": times_pre,
        "official_split": official,
        "split_matches_phase9_manifest": True,
    }


def pd_concat_train_val(train, val):
    import pandas as pd

    return pd.concat([train, val], ignore_index=True)


def run_calibration_robustness_audit(
    *,
    data_dir: Path | str | None = None,
    eval_dir: Path | str | None = None,
    model_dir: Path | str | None = None,
    n_boot: int = 200,
    n_nested: int = 40,
    nested_eval_frac: float = 0.30,
    n_folds: int = 5,
    seed: int = 42,
    write: bool = True,
    scores: dict | None = None,
) -> dict:
    eval_path = Path(eval_dir or EVAL_DIR)
    before = {str(p): _file_fingerprint(p) for p in PROTECTED_REPO_PATHS}

    if scores is None:
        scores = score_frozen_combined_pretest(data_dir=data_dir, model_dir=model_dir, eval_dir=eval_path)

    payload = build_payload_from_scores(
        raw_train=scores["raw_train"],
        y_train=scores["y_train"],
        raw_val=scores["raw_val"],
        y_val=scores["y_val"],
        times_pre=scores["times_pre"],
        official_split=scores["official_split"],
        n_boot=n_boot,
        n_nested=n_nested,
        nested_eval_frac=nested_eval_frac,
        n_folds=n_folds,
        seed=seed,
    )
    payload["split_matches_phase9_manifest"] = bool(scores.get("split_matches_phase9_manifest", True))
    clean = _py(payload)

    if write:
        eval_path.mkdir(parents=True, exist_ok=True)
        json_path = eval_path / "ieee_calibration_robustness.json"
        md_path = eval_path / "ieee_calibration_robustness_report.md"
        json_path.write_text(json.dumps(clean, indent=2))
        md_path.write_text(render_ieee_robustness_report(clean))
        after = {str(p): _file_fingerprint(p) for p in PROTECTED_REPO_PATHS}
        for key, prev in before.items():
            if prev != after[key]:
                raise RuntimeError(f"Protected artifact changed during IEEE calibration audit: {key}")
        clean["integrity"] = {
            "protected_artifacts_unchanged": True,
            "new_calibration_artifact_written": False,
            "wrote": ["ieee_calibration_robustness.json", "ieee_calibration_robustness_report.md"],
            "protected_artifact_fingerprints": after,
        }
        json_path.write_text(json.dumps(clean, indent=2))
        md_path.write_text(render_ieee_robustness_report(clean))
    return clean
