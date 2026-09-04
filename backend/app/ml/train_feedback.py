"""Offline feedback-model training. Never overwrites the live scoring artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np

from app.config import REPO_ROOT, get_settings
from app.ml.booster import make_classifier
from app.ml.features import FEATURE_COLUMNS
from app.ml.feedback_dataset import assert_no_future_leakage, temporal_split, validate_dataset
from app.ml.train import classification_metrics
from app.models.model_version import STATUS_CANDIDATE
from app.utils.ids import new_id
from app.utils.logging import get_logger

log = get_logger("ml.train_feedback")

LIVE_MODEL_VERSION = "xgb-iforest-v1-calibrated"
CANDIDATE_PREFIX = "xgb-feedback"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "ml" / "models" / "feedback"
MIN_TRAIN_ROWS = 12
RANDOM_SEED = 42


def _matrix(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[r["features"][c] for c in FEATURE_COLUMNS] for r in records], dtype=float)
    y = np.array([int(r["y"]) for r in records], dtype=int)
    return X, y


def _next_version(existing: list[str]) -> str:
    n = 1
    while f"{CANDIDATE_PREFIX}-v{n}" in existing:
        n += 1
    return f"{CANDIDATE_PREFIX}-v{n}"


def _predict_current(records: list[dict[str, Any]], predict_fn: Callable[[dict], dict] | None) -> np.ndarray:
    if predict_fn is None:
        from app.ml.predictor import model_service

        if not model_service.ready:
            model_service.load_or_train()
        predict_fn = model_service.predict
    probs = []
    for row in records:
        pred = predict_fn(row["raw"])
        probs.append(float(pred.get("ml_probability") if pred.get("ml_probability") is not None else pred.get("ml_score", 0) / 100.0))
    return np.array(probs, dtype=float)


def train_candidate(
    records: list[dict[str, Any]],
    *,
    artifact_root: Path | None = None,
    live_model_dir: Path | None = None,
    existing_versions: list[str] | None = None,
    current_predict_fn: Callable[[dict], dict] | None = None,
    min_rows: int | None = None,
) -> dict[str, Any]:
    """Train a CANDIDATE model. Does not write into the live model directory."""
    settings = get_settings()
    live_dir = Path(live_model_dir or settings.model_dir)
    artifact_root = Path(artifact_root or DEFAULT_ARTIFACT_ROOT)
    min_rows = min_rows if min_rows is not None else max(MIN_TRAIN_ROWS, int(getattr(settings, "feedback_min_train_rows", MIN_TRAIN_ROWS)))

    report = validate_dataset(records)
    if not report["ok"]:
        raise ValueError("feedback dataset failed validation: " + ", ".join(report["issues"][:5]))
    if report["n_labeled"] < min_rows:
        return {
            "ok": False,
            "skipped": True,
            "reason": f"need at least {min_rows} labeled feedback rows, have {report['n_labeled']}",
            "active_model_unchanged": True,
            "live_model_dir": str(live_dir),
            "validation": report,
        }
    if report["n_positive"] < 1 or report["n_negative"] < 1:
        return {
            "ok": False,
            "skipped": True,
            "reason": "need both fraud and legitimate confirmed labels",
            "active_model_unchanged": True,
            "validation": report,
        }

    train, eval_rows = temporal_split(records)
    assert_no_future_leakage(train, eval_rows)
    if not eval_rows:
        return {
            "ok": False,
            "skipped": True,
            "reason": "temporal split produced an empty evaluation set",
            "active_model_unchanged": True,
        }

    X_train, y_train = _matrix(train)
    X_eval, y_eval = _matrix(eval_rows)
    pos = max(int(y_train.sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    clf, family, _base = make_classifier(neg / pos, RANDOM_SEED)
    if family == "xgboost":
        clf.fit(X_train, y_train, verbose=False)
    else:
        clf.fit(X_train, y_train)

    cand_prob = clf.predict_proba(X_eval)[:, 1]
    cand_pred = (cand_prob >= 0.5).astype(int)
    candidate_metrics = classification_metrics(y_eval, cand_prob, cand_pred)

    current_prob = _predict_current(eval_rows, current_predict_fn)
    current_pred = (current_prob >= 0.5).astype(int)
    try:
        current_metrics = classification_metrics(y_eval, current_prob, current_pred)
    except Exception:
        current_metrics = {"error": "current model could not be scored on this eval slice"}

    version = _next_version(existing_versions or [])
    dest = artifact_root / version
    dest.mkdir(parents=True, exist_ok=True)
    if dest.resolve() == live_dir.resolve():
        raise RuntimeError("refusing to write a candidate into the live model directory")

    live_before = {
        p.name: p.stat().st_mtime if p.exists() else None
        for p in (live_dir / "xgb_fraud.joblib", live_dir / "version.txt", live_dir / "calibrator.joblib")
    }

    joblib.dump(clf, dest / "model.joblib")
    meta = {
        "model_id": version,
        "version": version,
        "status": STATUS_CANDIDATE,
        "dataset": "SYNTHETIC_FEEDBACK",
        "track": "SYNTHETIC_FEEDBACK",
        "not_ulb": True,
        "not_live": True,
        "feature_set": FEATURE_COLUMNS,
        "training_rows": int(len(train)),
        "positive_rows": int(y_train.sum()),
        "evaluation_rows": int(len(eval_rows)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "booster_family": family,
        "split": {
            "strategy": "temporal_by_feedback_created_at",
            "train_fraction": round(len(train) / len(records), 3),
            "eval_fraction": round(len(eval_rows) / len(records), 3),
            "max_train_ts": train[-1]["feedback_created_at"].isoformat()
            if hasattr(train[-1]["feedback_created_at"], "isoformat")
            else str(train[-1]["feedback_created_at"]),
            "min_eval_ts": eval_rows[0]["feedback_created_at"].isoformat()
            if hasattr(eval_rows[0]["feedback_created_at"], "isoformat")
            else str(eval_rows[0]["feedback_created_at"]),
        },
        "candidate_metrics": candidate_metrics,
        "current_model": {
            "version": LIVE_MODEL_VERSION,
            "metrics_on_feedback_eval": current_metrics,
            "note": "Comparison only. Live scoring was not replaced.",
        },
        "emphasis": "PR-AUC and precision/recall on analyst-confirmed labels. Not real-world fraud accuracy.",
        "id": new_id(),
        "artifact_path": str(dest),
    }
    (dest / "metrics.json").write_text(json.dumps(meta, indent=2, default=str))
    (dest / "version.txt").write_text(version)
    (dest / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))

    live_after = {
        p.name: p.stat().st_mtime if p.exists() else None
        for p in (live_dir / "xgb_fraud.joblib", live_dir / "version.txt", live_dir / "calibrator.joblib")
    }
    if live_before != live_after:
        raise RuntimeError("live model artifacts changed during candidate training")

    log.info("feedback_candidate_trained", version=version, train=len(train), eval=len(eval_rows))
    return {
        "ok": True,
        "skipped": False,
        "status": STATUS_CANDIDATE,
        "version": version,
        "model_id": version,
        "artifact_path": str(dest),
        "registry": meta,
        "active_model_unchanged": True,
        "live_model_version": LIVE_MODEL_VERSION,
        "live_model_dir": str(live_dir),
        "validation": report,
        "candidate_metrics": candidate_metrics,
        "current_metrics": current_metrics,
    }


async def persist_candidate(db, result: dict[str, Any]) -> None:
    if not result.get("ok") or not result.get("registry"):
        return
    from sqlalchemy import select

    from app.models.model_version import ModelVersion

    meta = result["registry"]
    existing = (
        await db.execute(select(ModelVersion).where(ModelVersion.version == meta["version"]))
    ).scalar_one_or_none()
    if existing:
        return
    db.add(
        ModelVersion(
            id=meta["id"],
            version=meta["version"],
            model_id=meta["model_id"],
            model_type="feedback-candidate",
            metrics=meta.get("candidate_metrics") or {},
            artifact_path=meta["artifact_path"],
            is_active=False,
            dataset=meta.get("dataset"),
            feature_set=meta.get("feature_set") or [],
            training_rows=int(meta.get("training_rows") or 0),
            positive_rows=int(meta.get("positive_rows") or 0),
            evaluation_rows=int(meta.get("evaluation_rows") or 0),
            status=STATUS_CANDIDATE,
        )
    )
    await db.commit()


async def train_candidate_from_db(db, **kwargs) -> dict[str, Any]:
    from sqlalchemy import select

    from app.ml.feedback_dataset import load_feedback_records
    from app.models.model_version import ModelVersion

    records = await load_feedback_records(db)
    existing = list((await db.execute(select(ModelVersion.version))).scalars().all())
    result = train_candidate(records, existing_versions=existing, **kwargs)
    await persist_candidate(db, result)
    return result


if __name__ == "__main__":
    import asyncio

    from app.database import SessionLocal, init_db

    async def _main():
        await init_db()
        async with SessionLocal() as db:
            result = await train_candidate_from_db(db)
        print(json.dumps({k: v for k, v in result.items() if k != "registry"}, indent=2, default=str))

    asyncio.run(_main())
