"""Strongly typed domain events. Payloads exclude secrets and raw payment identifiers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.events.base import SCHEMA_VERSION, current_correlation_id, sanitize_payload
from app.utils.ids import new_id, utcnow

EventType = Literal[
    "transaction-created",
    "risk-scored",
    "investigation-created",
    "investigation-completed",
    "alert-created",
    "analyst-feedback-recorded",
    "model-drift-detected",
]


class DomainEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=new_id)
    event_type: str
    timestamp: datetime = Field(default_factory=utcnow)
    schema_version: str = SCHEMA_VERSION
    correlation_id: str = Field(default_factory=current_correlation_id)
    transaction_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload", mode="before")
    @classmethod
    def _sanitize(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("payload must be an object")
        return sanitize_payload(value)


class TransactionCreated(DomainEvent):
    event_type: Literal["transaction-created"] = "transaction-created"
    transaction_id: str


class RiskScored(DomainEvent):
    event_type: Literal["risk-scored"] = "risk-scored"
    transaction_id: str


class InvestigationCreated(DomainEvent):
    event_type: Literal["investigation-created"] = "investigation-created"
    transaction_id: str


class InvestigationCompleted(DomainEvent):
    event_type: Literal["investigation-completed"] = "investigation-completed"
    transaction_id: str


class AlertCreated(DomainEvent):
    event_type: Literal["alert-created"] = "alert-created"
    transaction_id: str


class AnalystFeedbackRecorded(DomainEvent):
    event_type: Literal["analyst-feedback-recorded"] = "analyst-feedback-recorded"
    transaction_id: str


class ModelDriftDetected(DomainEvent):
    event_type: Literal["model-drift-detected"] = "model-drift-detected"
    transaction_id: str | None = None


EVENT_MODELS: dict[str, type[DomainEvent]] = {
    "transaction-created": TransactionCreated,
    "risk-scored": RiskScored,
    "investigation-created": InvestigationCreated,
    "investigation-completed": InvestigationCompleted,
    "alert-created": AlertCreated,
    "analyst-feedback-recorded": AnalystFeedbackRecorded,
    "model-drift-detected": ModelDriftDetected,
}


def parse_event(data: dict[str, Any]) -> DomainEvent:
    if not isinstance(data, dict):
        raise ValueError("event must be an object")
    event_type = data.get("event_type")
    model = EVENT_MODELS.get(str(event_type), DomainEvent)
    return model.model_validate(data)
