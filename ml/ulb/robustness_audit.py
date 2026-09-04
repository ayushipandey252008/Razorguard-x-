"""Run the ULB calibration robustness audit on train/validation only."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ml.ulb.adapter import ULBFraudDatasetAdapter, _relpath
from ml.ulb.calibration import clip_proba, fit_calibrators, reliability_svg
from ml.ulb.constants import DATASET_ID, DATASET_NAME, MODEL_VERSION, TARGET_COLUMN, TRACK
from ml.ulb.model import preload_openmp
from ml.ulb.preprocess import clean_ulb_frame
from ml.ulb.robustness import (
    METHODS,
    bootstrap_frozen_maps,
    kfold_oof,
    nested_holdout,
    recommend,
    score_bundle,
    staircase_diagnostics,
    train_fit_val_eval,
)
from ml.ulb.robustness_report import render_robustness_report
from ml.ulb.split import chronological_split, split_summary


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


PROTECTED_ARTIFACTS = (
    "calibration_metrics.json",
    "calibration_report.md",
    "ulb_metrics.json",
)


def run_calibration_robustness_audit(
    adapter: ULBFraudDatasetAdapter | None = None,
    n_boot: int = 400,
    n_nested: int = 40,
    nested_eval_frac: float = 0.30,
    n_folds: int = 5,
    seed: int = 42,
    write: bool = True,
) -> dict:
    adapter = adapter or ULBFraudDatasetAdapter()
    adapter.eval_dir = Path(adapter.eval_dir)
    adapter.model_dir = Path(adapter.model_dir)

    before = {name: _file_fingerprint(adapter.eval_dir / name) for name in PROTECTED_ARTIFACTS}

    if adapter._df is None:
        adapter.load()
    cleaned, _stats = clean_ulb_frame(adapter._df)
    adapter._cleaned = cleaned
    train, val, test = chronological_split(cleaned)
    official = split_summary(train, val, test, "chronological")
    # Integrity only: confirm the official test slice is the same size. Do not score it.
    saved_split = None
    saved_metrics = adapter.eval_dir / "ulb_metrics.json"
    if saved_metrics.exists():
        saved_split = json.loads(saved_metrics.read_text()).get("official_split")
        if saved_split:
            for part in ("train", "val", "test"):
                for key in ("n", "fraud"):
                    if saved_split.get(part, {}).get(key) != official.get(part, {}).get(key):
                        raise RuntimeError("Official split changed; aborting robustness audit.")
    del test

    preload_openmp()
    model, transformer = adapter.load_model()
    y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
    y_val = val[TARGET_COLUMN].to_numpy(dtype=int)
    raw_train = clip_proba(model.predict_proba(transformer.transform(train))[:, 1])
    raw_val = clip_proba(model.predict_proba(transformer.transform(val))[:, 1])

    fitted = fit_calibrators(raw_val, y_val)
    maps = {method: fitted.transform(raw_val, method) for method in METHODS}

    in_sample = {m: score_bundle(y_val, maps[m], m) for m in METHODS}
    stair = staircase_diagnostics(y_val, raw_val, maps["isotonic"])
    frozen_boot = bootstrap_frozen_maps(y_val, maps, n_boot=n_boot, seed=seed)
    nested = nested_holdout(raw_val, y_val, n_splits=n_nested, test_size=nested_eval_frac, seed=seed)
    oof = kfold_oof(raw_val, y_val, n_splits=n_folds, seed=seed)
    train_hold = train_fit_val_eval(raw_train, y_train, raw_val, y_val)

    payload = {
        "label": "CALIBRATION ROBUSTNESS AUDIT",
        "track": TRACK,
        "dataset_id": DATASET_ID,
        "dataset": DATASET_NAME,
        "booster_model_version": MODEL_VERSION,
        "current_phase2_selection": "isotonic",
        "artifacts_left_intact_policy": True,
        "methodology": {
            "train": "Frozen ulb-xgb-v1 booster scores only; booster is not refit.",
            "validation": "Calibrator comparison, bootstrap, nested holdout, k-fold OOF.",
            "test_scored": False,
            "test_used_for_fit": False,
            "test_used_for_method_selection": False,
            "test_used_for_recommendation": False,
            "train_n": int(len(y_train)),
            "train_positives": int(y_train.sum()),
            "validation_n": int(len(y_val)),
            "validation_positives": int(y_val.sum()),
        },
        "official_split": official,
        "split_matches_ulb_metrics": saved_split is not None,
        "in_sample_validation": in_sample,
        "staircase_in_sample_validation": stair,
        "bootstrap_frozen_maps": frozen_boot,
        "nested_holdout_validation": nested,
        "kfold_oof_validation": oof,
        "train_fit_val_eval": train_hold,
        "raw_path": _relpath(adapter.csv_path),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "n_boot": n_boot,
        "n_nested": n_nested,
        "n_folds": n_folds,
        "seed": seed,
    }
    payload["recommendation"] = recommend(payload)
    payload["previous_in_sample_brier_ranking"] = "isotonic < sigmoid < raw (not used as the decision rule)"

    clean = _py(payload)

    if write:
        adapter.eval_dir.mkdir(parents=True, exist_ok=True)
        figures = adapter.eval_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        for name, bundle in in_sample.items():
            svg = reliability_svg(
                bundle["reliability_uniform"]["bins"],
                f"Val reliability (in-sample) — {name}",
            )
            (figures / f"robustness_val_insample_{name}.svg").write_text(svg)
        if oof.get("pooled_oof"):
            for name, bundle in oof["pooled_oof"].items():
                bins = (bundle.get("reliability_uniform") or {}).get("bins") or []
                svg = reliability_svg(bins, f"Val reliability (k-fold OOF) — {name}")
                (figures / f"robustness_val_oof_{name}.svg").write_text(svg)
        (adapter.eval_dir / "calibration_robustness.json").write_text(json.dumps(clean, indent=2))
        (adapter.eval_dir / "calibration_robustness_report.md").write_text(render_robustness_report(clean))

        after = {name: _file_fingerprint(adapter.eval_dir / name) for name in PROTECTED_ARTIFACTS}
        for name in PROTECTED_ARTIFACTS:
            if before[name] != after[name]:
                raise RuntimeError(f"Protected artifact changed during audit: {name}")
        clean["protected_artifacts_unchanged"] = True
        clean["protected_artifact_fingerprints"] = after
        (adapter.eval_dir / "calibration_robustness.json").write_text(json.dumps(clean, indent=2))

    return clean
