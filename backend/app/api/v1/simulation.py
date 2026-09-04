from __future__ import annotations

import random

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import require_roles
from app.graph.rings import detect_potential_rings
from app.models.app_user import AppUser
from app.ml.scenarios.generators import SCENARIO_NAMES
from app.ml.scenarios.runner import run_scenario_evaluation
from app.schemas.common import ScenarioEvaluateRequest, SimulationRequest, TransactionCreate
from app.services.pipeline import process_transaction
from app.services.synthetic import scenario_transactions

router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("/run")
async def run_simulation(
    body: SimulationRequest,
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(require_roles("ADMIN", "RISK_ANALYST")),
):
    rng = random.Random()
    generated = scenario_transactions(body.scenario, body.count, rng)
    results = []
    for row in generated:
        payload = TransactionCreate(
            user_id=row["user_id"],
            merchant_id=row["merchant_id"],
            amount=row["amount"],
            currency=row["currency"],
            timestamp=row["timestamp"],
            device_id=row["device_id"],
            ip_address=row["ip_address"],
            location=row["location"],
            payment_method=row["payment_method"],
            merchant_category=row["merchant_category"],
            account_age_days=row["account_age_days"],
            failed_attempts=row["failed_attempts"],
            transaction_velocity=row["transaction_velocity"],
            previous_transaction_count=row["previous_transaction_count"],
            previous_average_amount=row["previous_average_amount"],
            current_device_known=row["current_device_known"],
            current_location_known=row["current_location_known"],
            payment_identifier=row["payment_identifier"],
            scenario_tag=row["scenario_tag"],
        )
        processed = await process_transaction(db, payload, actor=user.email)
        risk = processed["risk"]
        results.append(
            {
                "transaction_id": processed["transaction"].transaction_id,
                "user_id": processed["transaction"].user_id,
                "amount": processed["transaction"].amount,
                "scenario_tag": processed["transaction"].scenario_tag,
                "final_risk_score": risk.final_risk_score,
                "decision": risk.decision,
                "ml_score": risk.ml_score,
                "behavior_score": risk.behavior_score,
                "rule_score": risk.rule_score,
                "graph_score": risk.graph_score,
                "investigation_id": processed["investigation_id"],
                "injected_label": row.get("is_fraud"),
            }
        )

    decisions = {"APPROVE": 0, "REVIEW": 0, "BLOCK": 0}
    for r in results:
        decisions[r["decision"]] += 1
    labels = [r["injected_label"] for r in results]
    preds_block = [1 if r["decision"] == "BLOCK" else 0 for r in results]
    # False positives only meaningful vs injected labels, not real-world fraud.
    fp = sum(1 for y, p in zip(labels, preds_block) if y == 0 and p == 1)
    detected = sum(1 for r in results if r["decision"] != "APPROVE")
    clusters = detect_potential_rings(min_users=3)
    return {
        "scenario": body.scenario,
        "count": len(results),
        "transactions": results,
        "suspicious_transactions": [r for r in results if r["decision"] != "APPROVE"],
        "detected_flagged": detected,
        "decision_counts": decisions,
        "false_positives_vs_injected_label": fp,
        "detected_clusters": clusters,
        "note": (
            "Results come from the live risk pipeline. Injected labels describe the "
            "synthetic generator pattern, not confirmed real-world fraud. "
            "False positives are vs those injected labels only."
        ),
    }


@router.post("/evaluate")
async def evaluate_scenarios(
    body: ScenarioEvaluateRequest,
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(require_roles("ADMIN", "RISK_ANALYST")),
):
    counts = dict(body.counts or {})
    if not counts:
        for name in body.scenarios:
            counts[name] = body.count_per_scenario
    unknown = [k for k in counts if k not in SCENARIO_NAMES]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown scenarios: {unknown}")
    result = await run_scenario_evaluation(
        db,
        counts=counts,
        seed=body.seed,
        run_investigations=body.run_investigations,
        actor=user.email,
    )
    return result
