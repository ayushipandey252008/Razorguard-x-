from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user, require_roles
from app.models.app_user import AppUser
from app.models.risk import RiskAssessment, TriggeredRule
from app.schemas.common import RiskAssessmentOut, TransactionCreate
from app.services.pipeline import process_transaction
from app.api.v1.transactions import _risk_out

router = APIRouter(prefix="/risk", tags=["risk"])


@router.post("/evaluate")
async def evaluate(
    body: TransactionCreate,
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(require_roles("ADMIN", "RISK_ANALYST", "INVESTIGATOR")),
):
    try:
        result = await process_transaction(db, body, actor=user.email)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "transaction_id": result["transaction"].transaction_id,
        "risk": _risk_out(result["risk"], result["triggered_rules"]),
        "investigation_id": result["investigation_id"],
    }


@router.get("/{transaction_id}", response_model=RiskAssessmentOut)
async def get_risk(
    transaction_id: str,
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    risk = (
        await db.execute(select(RiskAssessment).where(RiskAssessment.transaction_id == transaction_id))
    ).scalar_one_or_none()
    if risk is None:
        raise HTTPException(status_code=404, detail="Risk assessment not found")
    rules = (
        (await db.execute(select(TriggeredRule).where(TriggeredRule.risk_assessment_id == risk.id)))
        .scalars()
        .all()
    )
    return _risk_out(risk, rules)
