from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.investigator import run_investigation
from app.database import get_db
from app.deps import get_current_user
from app.events.base import current_correlation_id
from app.events.emit import (
    drain_after_commit,
    enqueue_analyst_feedback,
    enqueue_investigation_completed,
    enqueue_investigation_created,
)
from app.graph.factory import graph_status
from app.models.app_user import AppUser
from app.models.investigation import AnalystDecision, Investigation, InvestigationToolCall
from app.models.risk import RiskAssessment
from app.models.transaction import Transaction
from app.schemas.common import DecisionRequest
from app.security.rbac import can_decide, can_investigate
from app.services.audit import write_audit
from app.services.events import TXN_CHANNEL, event_bus
from app.utils.ids import new_id, utcnow
from app.utils.logging import get_logger
from app.utils.redact import redact_secrets

log = get_logger("investigations")

router = APIRouter(prefix="/investigations", tags=["investigations"])


@router.get("")
async def list_investigations(
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    rows = (
        (
            await db.execute(
                select(Investigation, RiskAssessment)
                .join(RiskAssessment, RiskAssessment.transaction_id == Investigation.transaction_id, isouter=True)
                .order_by(Investigation.created_at.desc())
                .limit(100)
            )
        )
        .all()
    )
    out = []
    for inv, risk in rows:
        out.append(
            {
                "id": inv.id,
                "transaction_id": inv.transaction_id,
                "status": inv.status,
                "severity": inv.severity,
                "recommended_action": inv.recommended_action,
                "confidence": inv.confidence,
                "agent_provider": inv.agent_provider,
                "created_at": inv.created_at,
                "final_risk_score": risk.final_risk_score if risk else None,
                "decision": risk.decision if risk else None,
            }
        )
    return out


async def _load_investigation(db: AsyncSession, investigation_id: str) -> Investigation | None:
    return (
        await db.execute(select(Investigation).where(Investigation.id == investigation_id))
    ).scalar_one_or_none()


async def _resolve_or_create_investigation(db: AsyncSession, lookup_id: str) -> tuple[Investigation, bool]:
    """Accept an investigation id or a transaction id without a second conflicting route."""
    inv = await _load_investigation(db, lookup_id)
    if inv is not None:
        return inv, False
    txn = (
        await db.execute(select(Transaction).where(Transaction.transaction_id == lookup_id))
    ).scalar_one_or_none()
    if txn is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    existing = (
        await db.execute(
            select(Investigation)
            .where(Investigation.transaction_id == txn.transaction_id)
            .order_by(Investigation.created_at.desc())
        )
    ).scalars().first()
    if existing is not None:
        return existing, False
    risk = (
        await db.execute(select(RiskAssessment).where(RiskAssessment.transaction_id == txn.transaction_id))
    ).scalar_one_or_none()
    decision = risk.decision if risk else "REVIEW"
    inv = Investigation(
        id=new_id(),
        transaction_id=txn.transaction_id,
        status="OPEN",
        severity="high" if decision == "BLOCK" else "medium" if decision == "REVIEW" else "low",
    )
    db.add(inv)
    await db.flush()
    return inv, True


def _trace_payload(items: list) -> list[dict]:
    out = []
    for item in items:
        out.append(
            redact_secrets(
                {
                    "tool": item.get("tool"),
                    "arguments": item.get("arguments") or {},
                    "status": item.get("status"),
                    "duration_ms": item.get("duration_ms"),
                    "result_summary": item.get("result_summary"),
                    "result": item.get("result") or {},
                }
            )
        )
    return out


async def _replace_tool_calls(db: AsyncSession, investigation_id: str, trace: list[dict]) -> None:
    await db.execute(
        delete(InvestigationToolCall).where(InvestigationToolCall.investigation_id == investigation_id)
    )
    for item in _trace_payload(trace):
        db.add(
            InvestigationToolCall(
                id=new_id(),
                investigation_id=investigation_id,
                tool=item.get("tool") or "unknown",
                arguments=item.get("arguments"),
                status=item.get("status") or "success",
                duration_ms=item.get("duration_ms"),
                result_summary=item.get("result_summary"),
                result=item.get("result"),
            )
        )


def _run_response(inv: Investigation, report: dict, trace: list) -> dict:
    return redact_secrets(
        {
            "id": inv.id,
            "investigation_id": inv.id,
            "transaction_id": inv.transaction_id,
            "status": inv.status,
            "provider": inv.agent_provider,
            "recommendation": report.get("recommendation") or report.get("recommended_action"),
            "risk_level": report.get("risk_level"),
            "confidence": report.get("confidence"),
            "confidence_qualitative": report.get("confidence_qualitative"),
            "summary": report.get("summary"),
            "evidence": report.get("evidence"),
            "tool_trace": trace,
            "limitations": report.get("limitations"),
            "report": report,
        }
    )


@router.get("/{investigation_id}")
async def get_investigation(
    investigation_id: str,
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    inv = await _load_investigation(db, investigation_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    decisions = (
        (
            await db.execute(
                select(AnalystDecision)
                .where(AnalystDecision.investigation_id == investigation_id)
                .order_by(AnalystDecision.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    risk = (
        await db.execute(select(RiskAssessment).where(RiskAssessment.transaction_id == inv.transaction_id))
    ).scalar_one_or_none()
    from app.models.audit import AuditLog

    audits = (
        (
            await db.execute(
                select(AuditLog)
                .where(
                    (AuditLog.entity_id == inv.id) | (AuditLog.entity_id == inv.transaction_id)
                )
                .order_by(AuditLog.timestamp.desc())
                .limit(50)
            )
        )
        .scalars()
        .all()
    )
    report = redact_secrets(inv.ai_report) if inv.ai_report else None
    return {
        "id": inv.id,
        "transaction_id": inv.transaction_id,
        "status": inv.status,
        "severity": inv.severity,
        "ai_report": report,
        "recommended_action": inv.recommended_action,
        "confidence": inv.confidence,
        "agent_provider": inv.agent_provider,
        "provider": inv.agent_provider,
        "created_at": inv.created_at,
        "risk": {
            "final_risk_score": risk.final_risk_score,
            "decision": risk.decision,
            "explanation": risk.explanation,
            "graph_evidence": risk.graph_evidence,
            "ml_probability": risk.ml_probability,
            "model_version": risk.model_version,
        }
        if risk
        else None,
        "analyst_decisions": [
            {
                "id": d.id,
                "actor_email": d.actor_email,
                "decision": d.decision,
                "reason": d.reason,
                "previous_ai_recommendation": d.previous_ai_recommendation,
                "created_at": d.created_at,
            }
            for d in decisions
        ],
        "audit_history": [
            {
                "timestamp": a.timestamp,
                "actor": a.actor,
                "action": a.action,
                "entity_type": a.entity_type,
                "entity_id": a.entity_id,
                "metadata": a.extra,
            }
            for a in audits
        ],
    }


@router.get("/{investigation_id}/trace")
async def get_investigation_trace(
    investigation_id: str,
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    inv = await _load_investigation(db, investigation_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    rows = (
        (
            await db.execute(
                select(InvestigationToolCall)
                .where(InvestigationToolCall.investigation_id == investigation_id)
                .order_by(InvestigationToolCall.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    if rows:
        trace = [
            redact_secrets(
                {
                    "tool": row.tool,
                    "arguments": row.arguments or {},
                    "status": row.status,
                    "duration_ms": row.duration_ms,
                    "result_summary": row.result_summary,
                    "result": row.result or {},
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
            )
            for row in rows
        ]
    else:
        trace = _trace_payload((inv.ai_report or {}).get("tool_trace") or [])
    return {
        "investigation_id": inv.id,
        "transaction_id": inv.transaction_id,
        "provider": inv.agent_provider,
        "tool_trace": trace,
    }


@router.post("/{investigation_id}/run")
async def run_agent(
    investigation_id: str,
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    if not can_investigate(user.role):
        raise HTTPException(status_code=403, detail="Insufficient role")
    inv, created = await _resolve_or_create_investigation(db, investigation_id)
    inv.status = "RUNNING"
    if created:
        await enqueue_investigation_created(
            db,
            transaction_id=inv.transaction_id,
            correlation_id=current_correlation_id(),
            payload={
                "investigation_id": inv.id,
                "transaction_id": inv.transaction_id,
                "status": "OPEN",
                "severity": inv.severity,
            },
        )
    await db.commit()
    await db.refresh(inv)
    if created:
        await drain_after_commit()
    result = await run_investigation(db, inv.transaction_id, investigation_id=inv.id)
    report = result.get("report") or {}
    report["investigation_id"] = inv.id
    trace = _trace_payload(result.get("tool_trace") or report.get("tool_trace") or [])
    report["tool_trace"] = trace
    log.info(
        "agent_investigation",
        investigation_id=inv.id,
        transaction_id=inv.transaction_id,
        provider=result.get("provider"),
        model=result.get("model"),
        latency_ms=result.get("latency_ms"),
        tool_calls=len(trace),
        tool_failures=sum(1 for t in trace if t.get("status") in {"error", "unavailable"}),
        fallback=bool(result.get("fallback_reason")),
        identified_cluster=bool((report.get("potential_fraud_ring") or {}).get("identified")),
        correlation_id=current_correlation_id(),
    )
    stored = redact_secrets({**report, "tool_trace": trace, "model": result.get("model")})
    inv.ai_report = stored
    inv.recommended_action = report.get("recommended_action") or report.get("recommendation")
    conf = report.get("confidence")
    inv.confidence = float(conf) if isinstance(conf, (int, float)) else None
    inv.agent_provider = result.get("provider")
    inv.status = "COMPLETED"
    inv.updated_at = utcnow()
    await _replace_tool_calls(db, inv.id, trace)
    await write_audit(
        db,
        user.email,
        "ai_recommendation",
        "investigation",
        inv.id,
        {
            "recommended_action": inv.recommended_action,
            "provider": inv.agent_provider,
            "model": result.get("model"),
            "latency_ms": result.get("latency_ms"),
        },
    )
    graph = (report.get("graph_evidence") or {}) if isinstance(report, dict) else {}
    evidence = report.get("evidence") if isinstance(report, dict) else None
    await enqueue_investigation_completed(
        db,
        transaction_id=inv.transaction_id,
        correlation_id=current_correlation_id(),
        payload={
            "investigation_id": inv.id,
            "transaction_id": inv.transaction_id,
            "recommendation": inv.recommended_action,
            "risk_level": report.get("risk_level") if isinstance(report, dict) else None,
            "provider": result.get("provider"),
            "graph_backend": graph.get("graph_backend") or graph_status().get("graph_backend"),
            "evidence_count": len(evidence) if isinstance(evidence, list) else len(trace),
            "timestamp": utcnow().isoformat(),
        },
    )
    await db.commit()
    await event_bus.publish(
        TXN_CHANNEL,
        {
            "type": "investigation_created",
            "investigation_id": inv.id,
            "transaction_id": inv.transaction_id,
            "recommended_action": inv.recommended_action,
            "correlation_id": current_correlation_id(),
        },
    )
    await drain_after_commit()
    return _run_response(inv, stored, trace)


@router.post("/{investigation_id}/decision")
async def decide(
    investigation_id: str,
    body: DecisionRequest,
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    if not can_decide(user.role):
        raise HTTPException(status_code=403, detail="Only RISK_ANALYST or ADMIN can record decisions")
    inv = await _load_investigation(db, investigation_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    decision = AnalystDecision(
        id=new_id(),
        investigation_id=inv.id,
        actor_id=user.id,
        actor_email=user.email,
        decision=body.decision,
        reason=body.reason,
        previous_ai_recommendation=inv.recommended_action,
    )
    db.add(decision)
    inv.status = "CLOSED" if body.decision != "ESCALATE" else "ESCALATED"
    await write_audit(
        db,
        user.email,
        "analyst_decision",
        "investigation",
        inv.id,
        {"decision": body.decision, "reason": body.reason},
    )
    await write_audit(
        db,
        user.email,
        body.decision.lower() + "_action",
        "transaction",
        inv.transaction_id,
        {"investigation_id": inv.id},
    )
    await enqueue_analyst_feedback(
        db,
        transaction_id=inv.transaction_id,
        correlation_id=current_correlation_id(),
        payload={
            "investigation_id": inv.id,
            "transaction_id": inv.transaction_id,
            "decision": body.decision,
            "status": inv.status,
        },
    )
    await db.commit()
    await drain_after_commit()
    return {"ok": True, "status": inv.status, "decision": body.decision}
