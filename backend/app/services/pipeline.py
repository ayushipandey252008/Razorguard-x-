"""End-to-end scoring pipeline used by the transaction API and simulation."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.anomaly.behavior import analyze_behavior
from app.graph import service as graph_service
from app.ml.predictor import model_service
from app.models.investigation import Investigation
from app.models.risk import RiskAssessment, TriggeredRule
from app.models.transaction import Transaction
from app.rules.catalog import engine as rule_engine
from app.schemas.common import TransactionCreate
from app.events.base import bind_correlation_id, current_correlation_id
from app.events.emit import (
    drain_after_commit,
    enqueue_alert_created,
    enqueue_investigation_created,
    enqueue_risk_scored,
    enqueue_transaction_created,
    should_alert,
)
from app.events.schemas import TransactionCreated
from app.services.audit import write_audit
from app.services.enrichment import enrich_and_build
from app.services.events import TXN_CHANNEL, event_bus
from app.services.risk_engine import combine_scores, human_explanation
from app.utils.ids import new_id, utcnow
from app.utils.logging import Timer, get_logger

log = get_logger("pipeline")


async def process_transaction(
    db: AsyncSession,
    payload: TransactionCreate,
    actor: str = "system",
    *,
    drain_outbox: bool | None = None,
) -> dict:
    timer = Timer()
    correlation_id = current_correlation_id()
    bind_correlation_id(correlation_id)

    t_enrich = Timer()
    data = await enrich_and_build(db, payload)
    user_ctx = data.pop("_user")
    merchant_flag = data.pop("_merchant_watchlisted")
    enrich_ms = t_enrich.ms()

    txn = Transaction(**data)
    db.add(txn)
    await db.flush()
    created_event = TransactionCreated(
        correlation_id=correlation_id,
        transaction_id=txn.transaction_id,
        timestamp=utcnow(),
        payload={
            "user_id": txn.user_id,
            "merchant_id": txn.merchant_id,
            "amount": txn.amount,
            "currency": txn.currency,
            "payment_method": txn.payment_method,
            "merchant_category": txn.merchant_category,
            "scenario_tag": txn.scenario_tag,
        },
    )

    await write_audit(db, actor, "transaction_received", "transaction", txn.transaction_id, {"amount": txn.amount})

    t_ml = Timer()
    ml = model_service.predict(data)
    iso = model_service.isolation_score(data)
    ml_ms = t_ml.ms()

    t_beh = Timer()
    behavior = analyze_behavior(data, user_ctx, iso)
    behavior_ms = t_beh.ms()

    t_graph = Timer()
    graph_evidence = graph_service.ingest_transaction(data)
    await graph_service.persist_transaction_graph(db, data)
    graph_ms = t_graph.ms()

    t_rules = Timer()
    ctx = {
        "typical_amount": user_ctx.get("typical_amount"),
        "device_user_count": graph_evidence["device_user_count"],
        "ip_user_count": graph_evidence["ip_user_count"],
        "merchant_watchlisted": merchant_flag,
        "anomalies": behavior["detected_anomalies"],
    }
    fired = rule_engine.evaluate(data, ctx)
    rule_score = rule_engine.aggregate_score(fired)
    rules_ms = t_rules.ms()

    t_combine = Timer()
    combined = combine_scores(
        ml_score=ml["ml_score"],
        behavior_score=behavior["behavior_score"],
        rule_score=rule_score,
        graph_score=graph_evidence["graph_score"],
        triggered_count=len(fired),
    )
    explanation = human_explanation(
        combined["final_risk_score"],
        combined["decision"],
        ml["shap_top_features"],
        fired,
        behavior["detected_anomalies"],
        graph_evidence,
    )
    combine_ms = t_combine.ms()

    stage_latency_ms = {
        "enrich": enrich_ms,
        "ml": ml_ms,
        "ml_predict": ml.get("latency_ms"),
        "behavior": behavior_ms,
        "graph": graph_ms,
        "rules": rules_ms,
        "combine": combine_ms,
        "total": None,
    }

    weights = {
        **combined["weights"],
        "thresholds": combined["thresholds"],
        "probability_calibrated": bool(ml.get("probability_calibrated")),
        "ml_probability_raw": ml.get("ml_probability_raw"),
        "isolation_forest_scope": behavior.get("isolation_forest_scope"),
        "personalized_scope": behavior.get("personalized_scope"),
    }

    assessment = RiskAssessment(
        id=new_id(),
        transaction_id=txn.transaction_id,
        ml_score=ml["ml_score"],
        ml_probability=ml["ml_probability"],
        behavior_score=behavior["behavior_score"],
        rule_score=rule_score,
        graph_score=graph_evidence["graph_score"],
        final_risk_score=combined["final_risk_score"],
        decision=combined["decision"],
        confidence=combined["confidence"],
        model_version=ml["model_version"],
        shap_top_features=ml["shap_top_features"],
        anomalies=behavior["detected_anomalies"],
        graph_evidence=graph_evidence,
        explanation=explanation,
        weights=weights,
    )
    db.add(assessment)
    await db.flush()

    for rule in fired:
        db.add(
            TriggeredRule(
                id=new_id(),
                risk_assessment_id=assessment.id,
                transaction_id=txn.transaction_id,
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                severity=rule.severity,
                score_contribution=rule.score_contribution,
                explanation=rule.explanation,
                evidence=rule.evidence,
            )
        )

    investigation_id = None
    if combined["decision"] in {"REVIEW", "BLOCK"}:
        inv = Investigation(
            id=new_id(),
            transaction_id=txn.transaction_id,
            status="OPEN",
            severity="high" if combined["decision"] == "BLOCK" else "medium",
        )
        db.add(inv)
        investigation_id = inv.id
        await write_audit(
            db,
            actor,
            "investigation_created",
            "investigation",
            inv.id,
            {"transaction_id": txn.transaction_id, "decision": combined["decision"]},
        )

    await write_audit(
        db,
        actor,
        "risk_calculated",
        "transaction",
        txn.transaction_id,
        {
            "final_risk_score": combined["final_risk_score"],
            "decision": combined["decision"],
            "model_version": ml["model_version"],
            "probability_calibrated": bool(ml.get("probability_calibrated")),
            "stage_latency_ms": stage_latency_ms,
        },
    )
    t_outbox = Timer()
    await enqueue_transaction_created(db, created_event)
    await enqueue_risk_scored(
        db,
        transaction_id=txn.transaction_id,
        correlation_id=correlation_id,
        payload={
            "decision": combined["decision"],
            "final_risk_score": combined["final_risk_score"],
            "ml_score": ml["ml_score"],
            "behavior_score": behavior["behavior_score"],
            "rule_score": rule_score,
            "graph_score": graph_evidence["graph_score"],
            "model_version": ml["model_version"],
            "investigation_id": investigation_id,
            "confidence": combined["confidence"],
        },
    )
    if investigation_id:
        await enqueue_investigation_created(
            db,
            transaction_id=txn.transaction_id,
            correlation_id=correlation_id,
            payload={
                "investigation_id": investigation_id,
                "transaction_id": txn.transaction_id,
                "status": "OPEN",
                "severity": "high" if combined["decision"] == "BLOCK" else "medium",
                "decision": combined["decision"],
            },
        )
    if should_alert(combined["decision"]):
        await enqueue_alert_created(
            db,
            transaction_id=txn.transaction_id,
            correlation_id=correlation_id,
            decision=combined["decision"],
            risk_level=combined["decision"],
        )
    outbox_ms = t_outbox.ms()
    stage_latency_ms["outbox_enqueue"] = outbox_ms
    await db.commit()
    await db.refresh(txn)
    await db.refresh(assessment)

    if drain_outbox is not False:
        await drain_after_commit()

    event = {
        "type": "transaction_processed",
        "transaction_id": txn.transaction_id,
        "user_id": txn.user_id,
        "amount": txn.amount,
        "decision": combined["decision"],
        "final_risk_score": combined["final_risk_score"],
        "investigation_id": investigation_id,
        "high_risk": combined["decision"] != "APPROVE",
        "scenario_tag": txn.scenario_tag,
        "timestamp": utcnow().isoformat(),
        "correlation_id": correlation_id,
    }
    await event_bus.publish(TXN_CHANNEL, event)
    if combined["decision"] == "BLOCK":
        await event_bus.publish(TXN_CHANNEL, {**event, "type": "high_risk_alert"})

    total_ms = timer.ms()
    stage_latency_ms["total"] = total_ms
    log.info(
        "transaction_processed",
        transaction_id=txn.transaction_id,
        correlation_id=correlation_id,
        decision=combined["decision"],
        score=combined["final_risk_score"],
        latency_ms=total_ms,
        stage_latency_ms=stage_latency_ms,
        probability_calibrated=bool(ml.get("probability_calibrated")),
        graph_score=graph_evidence["graph_score"],
        cluster_id=graph_evidence.get("cluster_id"),
    )

    return {
        "transaction": txn,
        "risk": assessment,
        "triggered_rules": fired,
        "investigation_id": investigation_id,
        "behavior": behavior,
        "ml": ml,
        "stage_latency_ms": stage_latency_ms,
    }
