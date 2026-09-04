from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user, require_roles
from app.models.app_user import AppUser
from app.models.risk import RiskAssessment, TriggeredRule
from app.models.transaction import Transaction
from app.schemas.common import ProcessedTransactionOut, RiskAssessmentOut, TransactionCreate, TransactionOut
from app.services.pipeline import process_transaction

router = APIRouter(prefix="/transactions", tags=["transactions"])


def _txn_out(txn: Transaction, risk: RiskAssessment | None = None) -> TransactionOut:
    return TransactionOut(
        transaction_id=txn.transaction_id,
        user_id=txn.user_id,
        merchant_id=txn.merchant_id,
        amount=txn.amount,
        currency=txn.currency,
        timestamp=txn.timestamp,
        device_id=txn.device_id,
        ip_address=txn.ip_address,
        location=txn.location,
        payment_method=txn.payment_method,
        merchant_category=txn.merchant_category,
        account_age_days=txn.account_age_days,
        failed_attempts=txn.failed_attempts,
        transaction_velocity=txn.transaction_velocity,
        previous_transaction_count=txn.previous_transaction_count,
        previous_average_amount=txn.previous_average_amount,
        current_device_known=txn.current_device_known,
        current_location_known=txn.current_location_known,
        payment_identifier=txn.payment_identifier,
        scenario_tag=txn.scenario_tag,
        decision=risk.decision if risk else None,
        final_risk_score=risk.final_risk_score if risk else None,
    )


def _risk_out(risk: RiskAssessment, rules: list[TriggeredRule] | None = None) -> RiskAssessmentOut:
    weights = dict(risk.weights or {})
    return RiskAssessmentOut(
        transaction_id=risk.transaction_id,
        ml_score=risk.ml_score,
        ml_probability=risk.ml_probability,
        behavior_score=risk.behavior_score,
        rule_score=risk.rule_score,
        graph_score=risk.graph_score,
        final_risk_score=risk.final_risk_score,
        decision=risk.decision,  # type: ignore[arg-type]
        confidence=risk.confidence,
        model_version=risk.model_version,
        shap_top_features=risk.shap_top_features or [],
        anomalies=risk.anomalies or [],
        graph_evidence=risk.graph_evidence or {},
        triggered_rules=[
            {
                "rule_id": r.rule_id,
                "rule_name": r.rule_name,
                "severity": r.severity,
                "score_contribution": r.score_contribution,
                "explanation": r.explanation,
                "evidence": r.evidence or {},
            }
            for r in (rules or [])
        ],
        explanation=risk.explanation,
        weights=weights,
        probability_calibrated=bool(weights.get("probability_calibrated")),
        ml_probability_raw=weights.get("ml_probability_raw"),
    )


@router.post("", response_model=ProcessedTransactionOut)
async def create_transaction(
    body: TransactionCreate,
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(require_roles("ADMIN", "RISK_ANALYST", "INVESTIGATOR")),
):
    try:
        result = await process_transaction(db, body, actor=user.email)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    txn = result["transaction"]
    risk = result["risk"]
    return ProcessedTransactionOut(
        transaction=_txn_out(txn, risk),
        risk=_risk_out(risk, result["triggered_rules"]),
        investigation_id=result["investigation_id"],
    )


@router.get("", response_model=list[TransactionOut])
async def list_transactions(
    limit: int = Query(50, ge=1, le=200),
    decision: str | None = None,
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    stmt = (
        select(Transaction, RiskAssessment)
        .join(RiskAssessment, RiskAssessment.transaction_id == Transaction.transaction_id, isouter=True)
        .order_by(Transaction.timestamp.desc())
        .limit(limit)
    )
    if decision:
        stmt = stmt.where(RiskAssessment.decision == decision)
    rows = (await db.execute(stmt)).all()
    return [_txn_out(t, r) for t, r in rows]


@router.get("/{transaction_id}")
async def get_transaction(
    transaction_id: str,
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    txn = (
        await db.execute(select(Transaction).where(Transaction.transaction_id == transaction_id))
    ).scalar_one_or_none()
    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    risk = (
        await db.execute(select(RiskAssessment).where(RiskAssessment.transaction_id == transaction_id))
    ).scalar_one_or_none()
    rules = []
    if risk:
        rules = (
            (await db.execute(select(TriggeredRule).where(TriggeredRule.risk_assessment_id == risk.id)))
            .scalars()
            .all()
        )
    from app.models.investigation import Investigation
    from app.models.audit import AuditLog
    from app.models.payment_user import PaymentUser

    inv = (
        (
            await db.execute(
                select(Investigation)
                .where(Investigation.transaction_id == transaction_id)
                .order_by(Investigation.created_at.desc())
            )
        )
        .scalars()
        .first()
    )
    payment_user = (
        await db.execute(select(PaymentUser).where(PaymentUser.user_id == txn.user_id))
    ).scalar_one_or_none()
    typical = payment_user.typical_amount if payment_user else None
    user_baseline = None
    if payment_user:
        user_baseline = {
            "user_id": payment_user.user_id,
            "typical_amount": payment_user.typical_amount,
            "typical_hour": payment_user.typical_hour,
            "home_location": payment_user.home_location,
            "known_devices": payment_user.known_devices or [],
            "known_locations": payment_user.known_locations or [],
            "account_age_days": payment_user.account_age_days,
            "current_amount": txn.amount,
            "amount_vs_typical": (
                round(txn.amount / typical, 2) if typical and typical > 0 else None
            ),
        }
    audits = (
        (
            await db.execute(
                select(AuditLog)
                .where(AuditLog.entity_id == transaction_id)
                .order_by(AuditLog.timestamp.desc())
                .limit(40)
            )
        )
        .scalars()
        .all()
    )
    return {
        "transaction": _txn_out(txn, risk),
        "risk": _risk_out(risk, rules) if risk else None,
        "investigation": (
            {
                "id": inv.id,
                "status": inv.status,
                "severity": inv.severity,
                "ai_report": inv.ai_report,
                "recommended_action": inv.recommended_action,
                "confidence": inv.confidence,
                "agent_provider": inv.agent_provider,
            }
            if inv
            else None
        ),
        "audit_trail": [
            {
                "id": a.id,
                "timestamp": a.timestamp,
                "actor": a.actor,
                "action": a.action,
                "entity_type": a.entity_type,
                "entity_id": a.entity_id,
                "metadata": a.extra,
            }
            for a in audits
        ],
        "user_baseline": user_baseline,
    }
