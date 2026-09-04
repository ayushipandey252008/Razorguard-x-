"""Run synthetic scenario evaluation through the live scoring pipeline."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.investigator import run_investigation
from app.graph.factory import graph_status
from app.ml.scenarios.generators import (
    GRAPH_SCENARIOS,
    HIGH_RISK_SCENARIOS,
    SCENARIO_NAMES,
    generate_bundle,
)
from app.ml.scenarios.graph_eval import evaluate_graph_scenario
from app.ml.scenarios.investigation_eval import evaluate_investigation_grounding
from app.ml.scenarios.metrics import overall_metrics, scenario_matrix
from app.schemas.common import TransactionCreate
from app.services.pipeline import process_transaction
from app.utils.logging import get_logger

log = get_logger("ml.scenarios.runner")


def _payload(row: dict[str, Any]) -> TransactionCreate:
    return TransactionCreate(
        user_id=row["user_id"],
        merchant_id=row["merchant_id"],
        amount=row["amount"],
        currency=row.get("currency") or "INR",
        timestamp=row.get("timestamp"),
        device_id=row["device_id"],
        ip_address=row["ip_address"],
        location=row["location"],
        payment_method=row.get("payment_method") or "UPI",
        merchant_category=row.get("merchant_category"),
        account_age_days=row.get("account_age_days"),
        failed_attempts=row.get("failed_attempts") or 0,
        transaction_velocity=row.get("transaction_velocity"),
        previous_transaction_count=row.get("previous_transaction_count"),
        previous_average_amount=row.get("previous_average_amount"),
        current_device_known=row.get("current_device_known"),
        current_location_known=row.get("current_location_known"),
        payment_identifier=row.get("payment_identifier"),
        scenario_tag=row["scenario_tag"],
    )


async def run_scenario_evaluation(
    db: AsyncSession,
    *,
    counts: dict[str, int],
    seed: int = 42,
    run_investigations: bool = False,
    max_investigations: int = 3,
    actor: str = "scenario-eval",
) -> dict[str, Any]:
    unknown = [k for k in counts if k not in SCENARIO_NAMES]
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}")
    generated = generate_bundle(counts, seed=seed)
    scored: list[dict[str, Any]] = []
    for row in generated:
        processed = await process_transaction(db, _payload(row), actor=actor)
        risk = processed["risk"]
        scored.append(
            {
                "transaction_id": processed["transaction"].transaction_id,
                "user_id": processed["transaction"].user_id,
                "device_id": processed["transaction"].device_id,
                "ip_address": processed["transaction"].ip_address,
                "scenario": row["scenario"],
                "scenario_tag": row["scenario_tag"],
                "expected_fraud": int(row["expected_fraud"]),
                "expected_outcome": "FRAUD" if row["expected_fraud"] else "LEGITIMATE",
                "decision": risk.decision,
                "score": risk.final_risk_score,
                "ml_score": risk.ml_score,
                "investigation_id": processed.get("investigation_id"),
            }
        )

    graph_eval = []
    for name in GRAPH_SCENARIOS:
        group = [r for r in scored if r["scenario"] == name]
        if group:
            graph_eval.append(evaluate_graph_scenario(name, group))

    investigations = []
    if run_investigations:
        candidates = [r for r in scored if r["scenario"] in HIGH_RISK_SCENARIOS and r["decision"] != "APPROVE"]
        for row in candidates[: max(0, int(max_investigations))]:
            inv = await run_investigation(db, row["transaction_id"], investigation_id=row.get("investigation_id"))
            investigations.append(
                {
                    "transaction_id": row["transaction_id"],
                    "scenario": row["scenario"],
                    "grounding": evaluate_investigation_grounding(inv),
                    "recommendation": (inv.get("report") or {}).get("recommended_action"),
                    "limitations": (inv.get("report") or {}).get("limitations"),
                }
            )

    return {
        "label": "SYNTHETIC SCENARIO EVALUATION",
        "not_public_dataset": True,
        "not_real_world_accuracy": True,
        "seed": seed,
        "counts_requested": counts,
        "n": len(scored),
        "transactions": scored,
        "overall": overall_metrics(scored) if scored else {},
        "scenario_matrix": scenario_matrix(scored) if scored else [],
        "graph": graph_eval,
        "graph_backend": graph_status(),
        "investigations": investigations,
        "note": (
            "Generator labels describe the synthetic pattern. "
            "They are not ULB Class labels and not confirmed real-world fraud."
        ),
    }
