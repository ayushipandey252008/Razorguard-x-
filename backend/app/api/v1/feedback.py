"""Analyst feedback observations. Does not rewrite historical risk rows."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user, require_roles
from app.events.base import current_correlation_id
from app.events.emit import drain_after_commit, enqueue_analyst_feedback
from app.models.app_user import AppUser
from app.models.feedback import ANALYST_DECISIONS, AnalystFeedback, outcome_for_decision
from app.models.investigation import Investigation
from app.models.outbox import OutboxEvent
from app.models.risk import RiskAssessment
from app.schemas.common import FeedbackCreate
from app.security.rbac import can_decide
from app.services.audit import write_audit
from app.utils.ids import new_id
from app.utils.redact import redact_secrets

router = APIRouter(prefix="/feedback", tags=["feedback"])


def _to_out(row: AnalystFeedback) -> dict:
    return {
        "feedback_id": row.feedback_id,
        "investigation_id": row.investigation_id,
        "transaction_id": row.transaction_id,
        "analyst_decision": row.analyst_decision,
        "actual_outcome": row.actual_outcome,
        "reason": row.reason,
        "analyst_id": row.analyst_id,
        "created_at": row.created_at,
        "model_version": row.model_version,
        "risk_score": row.risk_score,
        "ml_probability": row.ml_probability,
        "decision_at_prediction_time": row.decision_at_prediction_time,
    }


@router.post("")
async def create_feedback(
    body: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(require_roles("ADMIN", "RISK_ANALYST")),
):
    if not can_decide(user.role):
        raise HTTPException(status_code=403, detail="Only RISK_ANALYST or ADMIN can record feedback")
    if body.decision not in ANALYST_DECISIONS:
        raise HTTPException(status_code=422, detail="Invalid analyst decision")
    inv = (
        await db.execute(select(Investigation).where(Investigation.id == body.investigation_id))
    ).scalar_one_or_none()
    if inv is None:
        raise HTTPException(status_code=404, detail="Investigation not found")

    existing = (
        await db.execute(select(AnalystFeedback).where(AnalystFeedback.investigation_id == inv.id))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail="Feedback already recorded for this investigation",
            headers={"X-Existing-Feedback": existing.feedback_id},
        )

    risk = (
        await db.execute(select(RiskAssessment).where(RiskAssessment.transaction_id == inv.transaction_id))
    ).scalar_one_or_none()
    snapshot_decision = risk.decision if risk else None
    snapshot_score = risk.final_risk_score if risk else None
    snapshot_prob = risk.ml_probability if risk else None
    snapshot_version = risk.model_version if risk else None

    row = AnalystFeedback(
        feedback_id=new_id(),
        investigation_id=inv.id,
        transaction_id=inv.transaction_id,
        analyst_decision=body.decision,
        actual_outcome=outcome_for_decision(body.decision),
        reason=body.reason,
        analyst_id=user.id,
        model_version=snapshot_version,
        risk_score=snapshot_score,
        ml_probability=snapshot_prob,
        decision_at_prediction_time=snapshot_decision,
    )
    db.add(row)
    await write_audit(
        db,
        user.email,
        "analyst_feedback",
        "investigation",
        inv.id,
        {"decision": body.decision, "transaction_id": inv.transaction_id},
    )
    event = await enqueue_analyst_feedback(
        db,
        transaction_id=inv.transaction_id,
        correlation_id=current_correlation_id(),
        payload={
            "feedback_id": row.feedback_id,
            "investigation_id": inv.id,
            "transaction_id": inv.transaction_id,
            "decision": body.decision,
            "actual_outcome": row.actual_outcome,
            "model_version": snapshot_version,
            "decision_at_prediction_time": snapshot_decision,
        },
    )
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Feedback already recorded for this investigation") from exc
    await db.refresh(row)
    if risk is not None:
        await db.refresh(risk)
        if risk.decision != snapshot_decision or risk.final_risk_score != snapshot_score:
            raise HTTPException(status_code=500, detail="historical risk row was mutated; aborting")
    await drain_after_commit()
    return redact_secrets(
        {
            **_to_out(row),
            "event_id": event.event_id,
            "correlation_id": event.correlation_id,
            "historical_risk_unchanged": True,
        }
    )


@router.get("")
async def list_feedback(
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(get_current_user),
    transaction_id: str | None = Query(default=None),
    investigation_id: str | None = Query(default=None),
    decision: str | None = Query(default=None),
    model_version: str | None = Query(default=None),
):
    stmt = select(AnalystFeedback).order_by(AnalystFeedback.created_at.desc()).limit(200)
    if transaction_id:
        stmt = stmt.where(AnalystFeedback.transaction_id == transaction_id)
    if investigation_id:
        stmt = stmt.where(AnalystFeedback.investigation_id == investigation_id)
    if decision:
        if decision not in ANALYST_DECISIONS:
            raise HTTPException(status_code=422, detail="Invalid analyst decision filter")
        stmt = stmt.where(AnalystFeedback.analyst_decision == decision)
    if model_version:
        stmt = stmt.where(AnalystFeedback.model_version == model_version)
    rows = list((await db.execute(stmt)).scalars().all())
    counts = {
        "CONFIRM_FRAUD": 0,
        "CONFIRM_LEGITIMATE": 0,
        "NEEDS_REVIEW": 0,
    }
    all_rows = list((await db.execute(select(AnalystFeedback))).scalars().all())
    for r in all_rows:
        if r.analyst_decision in counts:
            counts[r.analyst_decision] += 1
    return {
        "items": [_to_out(r) for r in rows],
        "counts": counts,
        "n": len(rows),
    }


@router.get("/{feedback_id}/outbox")
async def feedback_outbox(
    feedback_id: str,
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(require_roles("ADMIN", "RISK_ANALYST")),
):
    row = (
        await db.execute(select(AnalystFeedback).where(AnalystFeedback.feedback_id == feedback_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Feedback not found")
    events = list(
        (
            await db.execute(
                select(OutboxEvent).where(
                    OutboxEvent.event_type == "analyst-feedback-recorded",
                    OutboxEvent.aggregate_id.in_([row.investigation_id, row.transaction_id]),
                )
            )
        ).scalars().all()
    )
    return {
        "feedback_id": row.feedback_id,
        "outbox": [
            {
                "event_id": e.event_id,
                "status": e.status,
                "correlation_id": e.correlation_id,
                "event_type": e.event_type,
            }
            for e in events
        ],
    }
