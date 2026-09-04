"""Typed contracts for the investigation agent.

The LLM must not invent these values. Grounding copies fraud probability
and risk scores from the risk engine tool output.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Recommendation = Literal["APPROVE", "REVIEW", "BLOCK"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
ToolStatus = Literal["success", "unavailable", "error"]
ProviderName = Literal["llm", "deterministic_fallback"]

VALID_RECOMMENDATIONS = {"APPROVE", "REVIEW", "BLOCK"}
VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


class ToolCallTrace(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: ToolStatus
    duration_ms: float = 0.0
    result_summary: str = ""
    result: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class QualitativeConfidence(BaseModel):
    kind: Literal["qualitative"] = "qualitative"
    level: Literal["low", "medium", "high"] = "medium"
    note: str = (
        "Qualitative judgment from evidence coverage. Not a statistically "
        "calibrated probability."
    )


class ModelEvidence(BaseModel):
    ml_probability: float | None = None
    ml_score: float | None = None
    ml_probability_raw: float | None = None
    probability_calibrated: bool | None = None
    model_version: str | None = None
    shap_top_features: list[Any] = Field(default_factory=list)
    decision: str | None = None
    final_risk_score: float | None = None
    explanation: str | None = None
    unavailable: bool = False
    reason: str | None = None

    model_config = {"extra": "allow"}


class BehavioralEvidence(BaseModel):
    velocity: dict[str, Any] | None = None
    baseline: dict[str, Any] | None = None
    location: dict[str, Any] | None = None
    anomalies: list[Any] = Field(default_factory=list)
    unavailable: bool = False

    model_config = {"extra": "allow"}


class RuleEvidence(BaseModel):
    triggered: list[dict[str, Any]] = Field(default_factory=list)
    note: str | None = None

    model_config = {"extra": "allow"}


class GraphEvidence(BaseModel):
    cluster_found: bool = False
    cluster_id: str | None = None
    cluster_size: int | None = None
    fraud_associated_nodes: int | None = None
    connected_users: list[str] = Field(default_factory=list)
    shared_devices: list[str] = Field(default_factory=list)
    shared_ips: list[str] = Field(default_factory=list)
    suspicious_nodes: list[Any] = Field(default_factory=list)
    relationships: list[Any] = Field(default_factory=list)
    risk_indicators: list[Any] = Field(default_factory=list)
    reason: str | None = None
    identified: bool | None = None

    model_config = {"extra": "allow"}


class InvestigationReport(BaseModel):
    investigation_id: str | None = None
    transaction_id: str
    provider: str
    summary: str
    risk_level: RiskLevel
    recommendation: Recommendation
    confidence: float | None = None
    confidence_qualitative: QualitativeConfidence = Field(default_factory=QualitativeConfidence)
    model_evidence: dict[str, Any] = Field(default_factory=dict)
    behavioral_evidence: dict[str, Any] = Field(default_factory=dict)
    rule_evidence: dict[str, Any] = Field(default_factory=dict)
    graph_evidence: dict[str, Any] = Field(default_factory=dict)
    key_findings: list[str] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    limitations: str
    generated_at: datetime | None = None
    model: str | None = None

    @field_validator("recommendation", mode="before")
    @classmethod
    def _rec(cls, value: Any) -> str:
        if isinstance(value, str) and value.upper() in VALID_RECOMMENDATIONS:
            return value.upper()
        raise ValueError("recommendation must be APPROVE, REVIEW, or BLOCK")

    @field_validator("risk_level", mode="before")
    @classmethod
    def _level(cls, value: Any) -> str:
        if isinstance(value, str) and value.upper() in VALID_RISK_LEVELS:
            return value.upper()
        raise ValueError("risk_level must be LOW, MEDIUM, HIGH, or CRITICAL")

    model_config = {"extra": "allow"}


class GetTransactionIn(BaseModel):
    transaction_id: str


class GetUserHistoryIn(BaseModel):
    user_id: str
    limit: int = Field(default=10, ge=1, le=25)


class GetUserProfileIn(BaseModel):
    user_id: str


class GetUserBaselineIn(BaseModel):
    user_id: str
    transaction_id: str | None = None


class CheckDeviceIn(BaseModel):
    device_id: str


class CheckIpIn(BaseModel):
    ip_address: str


class CheckLocationIn(BaseModel):
    user_id: str
    location: str


class CheckVelocityIn(BaseModel):
    transaction_id: str


class GetModelExplanationIn(BaseModel):
    transaction_id: str


class GetTriggeredRulesIn(BaseModel):
    transaction_id: str


class FindConnectedAccountsIn(BaseModel):
    user_id: str


class FindFraudClusterIn(BaseModel):
    transaction_id: str
    user_id: str | None = None
