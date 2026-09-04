from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

STATUS_CANDIDATE = "CANDIDATE"
STATUS_ACTIVE = "ACTIVE"
STATUS_RETIRED = "RETIRED"
MODEL_STATUSES = (STATUS_CANDIDATE, STATUS_ACTIVE, STATUS_RETIRED)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    version: Mapped[str] = mapped_column(String(64), unique=True)
    model_type: Mapped[str] = mapped_column(String(32))
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_path: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    model_id: Mapped[str] = mapped_column(String(64), default="")
    dataset: Mapped[str | None] = mapped_column(String(64), nullable=True)
    feature_set: Mapped[list] = mapped_column(JSON, default=list)
    training_rows: Mapped[int] = mapped_column(Integer, default=0)
    positive_rows: Mapped[int] = mapped_column(Integer, default=0)
    evaluation_rows: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default=STATUS_CANDIDATE)
