from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user, require_roles
from app.events.base import current_correlation_id
from app.events.emit import drain_after_commit, enqueue_model_drift
from app.ml.drift import compute_drift, should_emit_alert
from app.ml.offline_metrics import load_offline_calibration, load_offline_ieee, load_offline_ulb_metrics
from app.ml.predictor import model_service
from app.ml.registry import MODEL_REGISTRY
from app.ml.train_feedback import LIVE_MODEL_VERSION, train_candidate_from_db
from app.models.app_user import AppUser
from app.models.feedback import AnalystFeedback, DriftAlert
from app.models.model_version import STATUS_ACTIVE, STATUS_CANDIDATE, ModelVersion
from app.utils.ids import new_id
from app.utils.redact import redact_secrets

router = APIRouter(prefix="/ml", tags=["ml"])


@router.get("/offline-evaluation")
async def offline_evaluation(user: AppUser = Depends(get_current_user)):
    """Committed ULB holdout metrics. Not live synthetic scores."""
    return {
        "ulb": load_offline_ulb_metrics(),
        "calibration": load_offline_calibration(),
        "ieee": load_offline_ieee(),
        "model_registry": MODEL_REGISTRY,
        "synthetic_live_scores": "not included — see /api/v1/analytics for the product model",
        "label": "OFFLINE EVALUATION",
        "ieee_label": "OFFLINE PUBLIC DATASET EVALUATION",
        "active_live_model": "xgb-iforest-v1-calibrated",
    }


@router.get("/ieee-evaluation")
async def ieee_evaluation(user: AppUser = Depends(get_current_user)):
    """IEEE-CIS offline public-dataset evaluation. Not live scoring."""
    payload = load_offline_ieee()
    payload["model_registry"] = [row for row in MODEL_REGISTRY if "ieee" in str(row.get("id", "")).lower() or row.get("track") == "IEEE_CIS_OFFLINE"]
    payload["active_model"] = {
        "version": "xgb-iforest-v1-calibrated",
        "status": "ACTIVE",
        "live": True,
    }
    payload["ieee_cis"] = {
        "status": "OFFLINE CANDIDATE",
        "live": False,
    }
    return payload


async def _feedback_counts(db: AsyncSession) -> dict[str, int]:
    rows = (
        await db.execute(
            select(AnalystFeedback.analyst_decision, func.count()).group_by(AnalystFeedback.analyst_decision)
        )
    ).all()
    counts = {"CONFIRM_FRAUD": 0, "CONFIRM_LEGITIMATE": 0, "NEEDS_REVIEW": 0}
    for decision, n in rows:
        counts[str(decision)] = int(n)
    return counts


async def _maybe_alert_drift(db: AsyncSession, report: dict) -> dict:
    last = (
        await db.execute(select(DriftAlert).order_by(DriftAlert.created_at.desc()).limit(1))
    ).scalar_one_or_none()
    last_at = last.created_at if last else None
    if not should_emit_alert(report.get("status") or "stable", last_at):
        return {**report, "alert_emitted": False}
    scores = [f.get("drift_score") for f in (report.get("features") or []) if f.get("drift_score") is not None]
    psi_max = max(scores) if scores else None
    event = await enqueue_model_drift(
        db,
        correlation_id=current_correlation_id(),
        payload={
            "status": report.get("status"),
            "recommendation": report.get("recommendation"),
            "psi_max": psi_max,
        },
    )
    db.add(
        DriftAlert(
            id=new_id(),
            overall_status=str(report.get("status")),
            psi_max=psi_max,
            recommendation=str(report.get("recommendation") or ""),
            event_id=event.event_id,
        )
    )
    await db.commit()
    await drain_after_commit()
    return {**report, "alert_emitted": True, "alert_event_id": event.event_id}


@router.get("/drift")
async def drift_status(
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    report = await compute_drift(db)
    report = await _maybe_alert_drift(db, report)
    return redact_secrets(report)


@router.get("/model-status")
async def model_status(
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    if not model_service.ready:
        model_service.load_or_train()
    active_row = (
        await db.execute(select(ModelVersion).where(ModelVersion.is_active.is_(True)))
    ).scalar_one_or_none()
    candidate_row = (
        await db.execute(
            select(ModelVersion)
            .where(ModelVersion.status == STATUS_CANDIDATE)
            .order_by(ModelVersion.trained_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    drift = await compute_drift(db)
    metrics = model_service.metrics or {}
    return redact_secrets(
        {
            "active_model": {
                "version": model_service.version,
                "model_id": (active_row.model_id if active_row and active_row.model_id else model_service.version),
                "dataset": (active_row.dataset if active_row and active_row.dataset else "SYNTHETIC_DATASET"),
                "training_rows": (active_row.training_rows if active_row else metrics.get("n_samples")),
                "last_evaluation": {
                    "pr_auc": metrics.get("pr_auc"),
                    "roc_auc": metrics.get("roc_auc"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                },
                "status": STATUS_ACTIVE,
                "artifact_path": str(model_service.model_dir),
                "live": True,
            },
            "candidate_model": None
            if candidate_row is None
            else {
                "version": candidate_row.version,
                "model_id": candidate_row.model_id or candidate_row.version,
                "dataset": candidate_row.dataset,
                "metrics": candidate_row.metrics,
                "status": candidate_row.status,
                "training_rows": candidate_row.training_rows,
                "evaluation_rows": candidate_row.evaluation_rows,
                "artifact_path": candidate_row.artifact_path,
                "live": False,
            },
            "training_dataset": "SYNTHETIC_DATASET for live; SYNTHETIC_FEEDBACK for candidates",
            "last_evaluation": metrics,
            "drift_status": {
                "status": drift.get("status"),
                "recommendation": drift.get("recommendation"),
                "checked_at": drift.get("checked_at"),
            },
            "feedback": await _feedback_counts(db),
            "live_model_version_expected": LIVE_MODEL_VERSION,
            "ieee_cis": {
                "status": "OFFLINE CANDIDATE",
                "live": False,
                "auto_activated": False,
                "versions": ["ieee-xgb-baseline-v1", "ieee-xgb-combined-v1", "ieee-xgb-graph-v1"],
                "label": "OFFLINE PUBLIC DATASET EVALUATION",
            },
            "auto_activation": False,
            "note": "Candidate models stay offline until explicitly selected. This prototype has no activation endpoint.",
        }
    )


@router.post("/train-feedback")
async def train_feedback(
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(require_roles("ADMIN")),
):
    """Offline candidate training. Does not replace the live model."""
    result = await train_candidate_from_db(db)
    if not result.get("ok") and not result.get("skipped"):
        raise HTTPException(status_code=400, detail=str(result.get("reason") or "training failed"))
    return redact_secrets(result)
