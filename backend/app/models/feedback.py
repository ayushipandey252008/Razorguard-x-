"""Analyst feedback observations. Never rewrite historical risk rows."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

CONFIRM_FRAUD = "CONFIRM_FRAUD"
CONFIRM_LEGITIMATE = "CONFIRM_LEGITIMATE"
NEEDS_REVIEW = "NEEDS_REVIEW"

ANALYST_DECISIONS = (CONFIRM_FRAUD, CONFIRM_LEGITIMATE, NEEDS_REVIEW)

OUTCOME_FRAUD = "FRAUD"
OUTCOME_LEGITIMATE = "LEGITIMATE"

DEFINED_OUTCOMES = (OUTCOME_FRAUD, OUTCOME_LEGITIMATE)


def outcome_for_decision(decision: str) -> str | None:
    if decision == CONFIRM_FRAUD:
        return OUTCOME_FRAUD
    if decision == CONFIRM_LEGITIMATE:
        return OUTCOME_LEGITIMATE
    return None


class AnalystFeedback(Base):
    __tablename__ = "analyst_feedback"
    __table_args__ = (UniqueConstraint("investigation_id", name="uq_feedback_investigation"),)

    feedback_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    investigation_id: Mapped[str] = mapped_column(ForeignKey("investigations.id"), index=True)
    transaction_id: Mapped[str] = mapped_column(String(64), index=True)
    analyst_decision: Mapped[str] = mapped_column(String(32), index=True)
    actual_outcome: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    reason: Mapped[str] = mapped_column(Text)
    analyst_id: Mapped[str] = mapped_column(String(36), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ml_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision_at_prediction_time: Mapped[str | None] = mapped_column(String(16), nullable=True)


class DriftAlert(Base):
    """Cooldown log so drift alerts are not emitted on every GET."""

    __tablename__ = "drift_alerts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    overall_status: Mapped[str] = mapped_column(String(16), index=True)
    psi_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    recommendation: Mapped[str] = mapped_column(Text)
    event_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
