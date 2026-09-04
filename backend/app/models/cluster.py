from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class FraudCluster(Base):
    __tablename__ = "fraud_clusters"

    cluster_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_count: Mapped[int] = mapped_column(Integer)
    shared_devices: Mapped[list] = mapped_column(JSON, default=list)
    shared_ips: Mapped[list] = mapped_column(JSON, default=list)
    merchants: Mapped[list] = mapped_column(JSON, default=list)
    entities: Mapped[dict] = mapped_column(JSON, default=dict)
    graph_risk_score: Mapped[float] = mapped_column(Float)
    explanation: Mapped[str] = mapped_column(Text)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
