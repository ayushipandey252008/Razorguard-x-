from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user
from app.graph.rings import detect_potential_rings, prototype_graph_thresholds
from app.config import get_settings
from app.ml.predictor import model_service
from app.models.app_user import AppUser
from app.models.investigation import AnalystDecision, Investigation
from app.models.risk import RiskAssessment
from app.models.transaction import Transaction

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("")
async def analytics(
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    total = (await db.execute(select(func.count()).select_from(Transaction))).scalar() or 0
    decisions = (
        await db.execute(
            select(RiskAssessment.decision, func.count()).group_by(RiskAssessment.decision)
        )
    ).all()
    decision_map = {d: c for d, c in decisions}
    reviewed = (
        await db.execute(select(func.count()).select_from(Investigation).where(Investigation.status != "OPEN"))
    ).scalar() or 0
    open_inv = (
        await db.execute(select(func.count()).select_from(Investigation).where(Investigation.status.in_(["OPEN", "RUNNING", "COMPLETED", "ESCALATED"])))
    ).scalar() or 0
    active_inv = (
        await db.execute(
            select(func.count()).select_from(Investigation).where(Investigation.status.in_(["OPEN", "RUNNING", "COMPLETED"]))
        )
    ).scalar() or 0
    high_risk = decision_map.get("BLOCK", 0) + decision_map.get("REVIEW", 0)
    blocked = decision_map.get("BLOCK", 0)
    rings = detect_potential_rings(min_users=3)

    # volume by day
    since = datetime.now(timezone.utc) - timedelta(days=14)
    rows = (
        await db.execute(select(Transaction.timestamp, RiskAssessment.decision, RiskAssessment.final_risk_score)
        .join(RiskAssessment, RiskAssessment.transaction_id == Transaction.transaction_id, isouter=True)
        .where(Transaction.timestamp >= since))
    ).all()
    volume = defaultdict(int)
    fraudish = defaultdict(int)
    for ts, decision, _score in rows:
        if ts is None:
            continue
        day = ts.date().isoformat()
        volume[day] += 1
        if decision in {"BLOCK", "REVIEW"}:
            fraudish[day] += 1
    volume_series = [{"date": d, "count": volume[d], "flagged": fraudish[d]} for d in sorted(volume)]

    # risk distribution buckets
    scores = (await db.execute(select(RiskAssessment.final_risk_score))).scalars().all()
    review_t = get_settings().threshold_review
    block_t = get_settings().threshold_block
    buckets = {
        f"0-{int(review_t)}": 0,
        f"{int(review_t)}-{int(block_t)}": 0,
        f"{int(block_t)}-100": 0,
    }
    for s in scores:
        if s < review_t:
            buckets[f"0-{int(review_t)}"] += 1
        elif s < block_t:
            buckets[f"{int(review_t)}-{int(block_t)}"] += 1
        else:
            buckets[f"{int(block_t)}-100"] += 1

    cats = (
        await db.execute(
            select(Transaction.merchant_category, RiskAssessment.decision, func.count())
            .join(RiskAssessment, RiskAssessment.transaction_id == Transaction.transaction_id)
            .group_by(Transaction.merchant_category, RiskAssessment.decision)
        )
    ).all()
    by_category = defaultdict(lambda: {"APPROVE": 0, "REVIEW": 0, "BLOCK": 0})
    for cat, dec, c in cats:
        by_category[cat][dec] = c

    locs = (
        await db.execute(
            select(Transaction.location, func.count())
            .join(RiskAssessment, RiskAssessment.transaction_id == Transaction.transaction_id)
            .where(RiskAssessment.decision.in_(["REVIEW", "BLOCK"]))
            .group_by(Transaction.location)
        )
    ).all()

    outcomes = (
        await db.execute(select(AnalystDecision.decision, func.count()).group_by(AnalystDecision.decision))
    ).all()

    anomalies = (
        await db.execute(select(RiskAssessment.anomalies, RiskAssessment.created_at).limit(500))
    ).all()
    anomaly_trend = defaultdict(int)
    for anoms, created in anomalies:
        if created is None:
            continue
        day = created.date().isoformat()
        anomaly_trend[day] += len(anoms or [])

    return {
        "totals": {
            "transactions": total,
            "reviewed": reviewed,
            "high_risk": high_risk,
            "blocked": blocked,
            "fraud_rate": (high_risk / total) if total else 0.0,
            "potential_fraud_rings": len(rings),
            "active_investigations": active_inv,
        },
        "decisions": decision_map,
        "volume": volume_series,
        "risk_distribution": [{"bucket": k, "count": v} for k, v in buckets.items()],
        "by_category": [{"category": k, **v} for k, v in by_category.items()],
        "by_location": [{"location": loc, "flagged": c} for loc, c in locs],
        "anomaly_trend": [{"date": d, "signals": anomaly_trend[d]} for d in sorted(anomaly_trend)],
        "investigation_outcomes": [{"decision": d, "count": c} for d, c in outcomes],
        "model": {
            "version": model_service.version,
            "metrics": model_service.metrics,
            "ready": model_service.ready,
            "probability_calibrated": model_service.probability_calibrated,
            "evaluation_track": (model_service.metrics or {}).get("track", "SYNTHETIC_DATASET"),
            "graph_cluster_thresholds": prototype_graph_thresholds(),
        },
        "prototype": True,
        "disclaimer": "Prototype / Synthetic Data. Metrics are not production fraud performance.",
    }
