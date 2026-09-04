from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SECRET_PLACEHOLDERS = frozenset({"dev-secret-change-me", "change-me-to-a-long-random-string"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    app_name: str = "RazorGuard X"
    environment: str = "development"
    secret_key: str = "dev-secret-change-me"
    access_token_expire_minutes: int = 480
    algorithm: str = "HS256"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    database_url: str = "sqlite+aiosqlite:///./razorguard.db"
    redis_url: str | None = "redis://localhost:6379/0"

    weight_ml: float = 0.35
    weight_behavior: float = 0.20
    weight_rules: float = 0.25
    weight_graph: float = 0.20
    threshold_review: float = 40.0
    threshold_block: float = 70.0
    cost_false_positive: float = 10.0
    cost_false_negative: float = 100.0
    cost_review: float = 2.0

    llm_provider: str = "none"
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    openai_api_key: str | None = None

    # Prototype graph-ring heuristics. Not production-grade.
    graph_min_cluster_users: int = 3
    graph_shared_device_accounts: int = 3
    graph_shared_ip_accounts: int = 3
    graph_flagged_path_max_hops: int = 2
    graph_dense_subgraph_min_density: float = 0.6
    graph_suspicious_entity_degree: int = 3

    graph_backend: str = "networkx"
    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"
    graph_neo4j_fallback: bool = True
    graph_rebuild_on_start: bool = False
    graph_connect_timeout_seconds: int = 30

    # Domain event bus (Kafka optional). Default stays in-process so local pytest
    # and machines without Kafka keep working. EVENT_BUS=kafka is opt-in.
    event_bus: str = "inprocess"
    event_bus_fallback: bool = True
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_connect_timeout_seconds: int = 15
    kafka_publish_timeout_ms: int = 5000
    kafka_topic_transactions: str = "transactions"
    kafka_topic_risk_results: str = "risk-results"
    kafka_topic_investigations: str = "investigations"
    kafka_topic_alerts: str = "alerts"
    kafka_topic_feedback: str = "feedback"
    kafka_topic_dlq: str = "events-dlq"
    kafka_group_risk: str = "rgx-risk-results"
    kafka_group_investigations: str = "rgx-investigations"
    kafka_group_alerts: str = "rgx-alerts"
    kafka_group_feedback: str = "rgx-feedback"
    kafka_group_transactions: str = "rgx-transactions"
    event_alert_on_block: bool = True
    event_alert_on_review: bool = True

    outbox_enabled: bool = True
    outbox_poll_interval_ms: int = 250
    outbox_batch_size: int = 20
    outbox_max_attempts: int = 5
    outbox_retry_backoff_seconds: float = 1.0
    outbox_stale_processing_seconds: int = 30
    outbox_drain_after_commit: bool = True
    outbox_background_worker: bool = True
    event_consumer_in_api: bool = True

    drift_psi_low: float = 0.10
    drift_psi_high: float = 0.25
    drift_min_samples: int = 20
    drift_alert_cooldown_seconds: int = 3600
    feedback_min_train_rows: int = 12

    ieee_data_dir: Path = REPO_ROOT / "ml" / "data" / "ieee"
    ieee_max_rows: int | None = None
    ieee_train_frac: float = 0.70
    ieee_val_frac: float = 0.15
    ieee_random_seed: int = 42

    model_dir: Path = REPO_ROOT / "ml" / "models"
    seed_admin_email: str = "admin@razorguard.local"
    seed_admin_password: str = "prototype-pass"

    rate_limit_per_minute: int = 120

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def risk_weights(self) -> dict[str, float]:
        return {
            "ml": self.weight_ml,
            "behavior": self.weight_behavior,
            "rules": self.weight_rules,
            "graph": self.weight_graph,
        }

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


def assert_secure_settings(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    if not settings.is_production:
        return
    if settings.secret_key in DEFAULT_SECRET_PLACEHOLDERS or len(settings.secret_key) < 32:
        raise RuntimeError(
            "ENVIRONMENT=production requires SECRET_KEY to be a non-default string of at least 32 characters."
        )
    if settings.seed_admin_password == "prototype-pass":
        raise RuntimeError("ENVIRONMENT=production refuses the documented lab seed password.")
