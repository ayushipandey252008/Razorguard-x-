"""Configurable Kafka topic names and their purpose.

This is a student prototype mapping, not a production topic taxonomy.
"""

from __future__ import annotations

from app.config import get_settings

# event_type → logical topic key
EVENT_TOPIC_KEYS = {
    "transaction-created": "transactions",
    "risk-scored": "risk-results",
    "investigation-created": "investigations",
    "investigation-completed": "investigations",
    "alert-created": "alerts",
    "analyst-feedback-recorded": "feedback",
    "model-drift-detected": "alerts",
}

TOPIC_PURPOSE = {
    "transactions": "transaction-created — a payment was accepted into the scoring pipeline",
    "risk-results": "risk-scored — synchronous ML/rules/graph decision completed",
    "investigations": "investigation-created / investigation-completed — case opened or agent finished",
    "alerts": "alert-created — BLOCK or configured high-risk REVIEW; model-drift-detected monitoring",
    "feedback": "analyst-feedback-recorded — CONFIRM_FRAUD / CONFIRM_LEGITIMATE / NEEDS_REVIEW (and legacy APPROVE / BLOCK / ESCALATE)",
    "dlq": "failed or malformed events after handler retries",
}


def topic_names() -> dict[str, str]:
    settings = get_settings()
    return {
        "transactions": settings.kafka_topic_transactions,
        "risk-results": settings.kafka_topic_risk_results,
        "investigations": settings.kafka_topic_investigations,
        "alerts": settings.kafka_topic_alerts,
        "feedback": settings.kafka_topic_feedback,
        "dlq": settings.kafka_topic_dlq,
    }


def topic_for_event(event_type: str) -> str:
    names = topic_names()
    key = EVENT_TOPIC_KEYS.get(event_type, "dlq")
    return names.get(key, names["dlq"])


def topic_documentation() -> list[dict[str, str]]:
    names = topic_names()
    return [
        {"logical": logical, "topic": names[logical], "purpose": purpose}
        for logical, purpose in TOPIC_PURPOSE.items()
        if logical in names
    ]
