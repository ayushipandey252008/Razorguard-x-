from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str
    display_name: str


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=128)


class TransactionCreate(BaseModel):
    user_id: str = Field(min_length=1, max_length=80)
    merchant_id: str = Field(min_length=1, max_length=80)
    amount: float = Field(gt=0, le=10_000_000)
    currency: str = Field(default="INR", max_length=8)
    timestamp: datetime | None = None
    device_id: str = Field(min_length=1, max_length=80)
    ip_address: str = Field(min_length=7, max_length=64)
    location: str = Field(min_length=1, max_length=80)
    payment_method: str = Field(default="UPI", max_length=32)
    merchant_category: str | None = Field(default=None, max_length=64)
    account_age_days: int | None = Field(default=None, ge=0, le=20000)
    failed_attempts: int = Field(default=0, ge=0, le=1000)
    transaction_velocity: int | None = Field(default=None, ge=0, le=10000)
    previous_transaction_count: int | None = Field(default=None, ge=0, le=1_000_000)
    previous_average_amount: float | None = Field(default=None, ge=0)
    current_device_known: bool | None = None
    current_location_known: bool | None = None
    payment_identifier: str | None = Field(default=None, max_length=80)
    scenario_tag: str | None = Field(default=None, max_length=64)

    @field_validator("ip_address")
    @classmethod
    def validate_ip(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) != 4:
            raise ValueError("Use a synthetic IPv4-style identifier")
        return v


class TransactionOut(BaseModel):
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    currency: str
    timestamp: datetime
    device_id: str
    ip_address: str
    location: str
    payment_method: str
    merchant_category: str
    account_age_days: int
    failed_attempts: int
    transaction_velocity: int
    previous_transaction_count: int
    previous_average_amount: float
    current_device_known: bool
    current_location_known: bool
    payment_identifier: str
    scenario_tag: str | None = None
    decision: str | None = None
    final_risk_score: float | None = None

    model_config = {"from_attributes": True}


class TriggeredRuleOut(BaseModel):
    rule_id: str
    rule_name: str
    severity: str
    score_contribution: float
    explanation: str
    evidence: dict[str, Any] = {}


class ShapFeature(BaseModel):
    feature: str
    contribution: float
    value: float | str | None = None


class RiskAssessmentOut(BaseModel):
    transaction_id: str
    ml_score: float
    ml_probability: float
    behavior_score: float
    rule_score: float
    graph_score: float
    final_risk_score: float
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    confidence: float
    model_version: str
    shap_top_features: list[ShapFeature] = []
    anomalies: list[dict[str, Any]] = []
    graph_evidence: dict[str, Any] = {}
    triggered_rules: list[TriggeredRuleOut] = []
    explanation: str
    weights: dict[str, Any] = {}
    probability_calibrated: bool = False
    ml_probability_raw: float | None = None


class ProcessedTransactionOut(BaseModel):
    transaction: TransactionOut
    risk: RiskAssessmentOut
    investigation_id: str | None = None


class InvestigationOut(BaseModel):
    id: str
    transaction_id: str
    status: str
    severity: str
    ai_report: dict[str, Any] | None = None
    recommended_action: str | None = None
    confidence: float | None = None
    agent_provider: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    decision: str | None = None

    model_config = {"from_attributes": True}


class DecisionRequest(BaseModel):
    decision: Literal["APPROVE", "BLOCK", "ESCALATE"]
    reason: str = Field(min_length=3, max_length=2000)


class SimulationRequest(BaseModel):
    scenario: Literal[
        "normal",
        "stolen_account",
        "card_testing",
        "account_takeover",
        "device_farm",
        "fraud_ring",
        "velocity_attack",
    ]
    count: int = Field(default=12, ge=1, le=80)


class FeedbackCreate(BaseModel):
    investigation_id: str = Field(min_length=1, max_length=64)
    decision: Literal["CONFIRM_FRAUD", "CONFIRM_LEGITIMATE", "NEEDS_REVIEW"]
    reason: str = Field(min_length=3, max_length=2000)


class ScenarioEvaluateRequest(BaseModel):
    scenarios: list[
        Literal[
            "normal_payment",
            "stolen_account",
            "card_testing",
            "high_velocity",
            "unusual_amount",
            "new_device",
            "shared_device",
            "shared_ip",
            "device_farm",
            "fraud_ring",
        ]
    ] = Field(default_factory=lambda: ["normal_payment", "stolen_account", "card_testing"])
    count_per_scenario: int = Field(default=8, ge=1, le=50)
    counts: dict[str, int] | None = None
    seed: int = Field(default=42, ge=0, le=1_000_000)
    run_investigations: bool = False


class GraphQueryOut(BaseModel):
    entity_id: str
    entity_type: str
    neighbors: list[dict[str, Any]]
    connected_users: list[str]
    graph_risk_score: float
    evidence: dict[str, Any]


class ClusterOut(BaseModel):
    cluster_id: str
    user_count: int
    shared_devices: list[str]
    shared_ips: list[str]
    merchants: list[str]
    entities: dict[str, Any]
    graph_risk_score: float
    explanation: str
    detected_at: datetime | None = None
