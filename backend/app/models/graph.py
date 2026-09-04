from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class GraphEntity(Base):
    __tablename__ = "graph_entities"
    __table_args__ = (UniqueConstraint("entity_type", "entity_key", name="uq_graph_entity"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(24), index=True)
    entity_key: Mapped[str] = mapped_column(String(128), index=True)
    properties: Mapped[dict] = mapped_column(JSON, default=dict)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)


class GraphRelationship(Base):
    __tablename__ = "graph_relationships"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    from_id: Mapped[str] = mapped_column(String(36), index=True)
    to_id: Mapped[str] = mapped_column(String(36), index=True)
    rel_type: Mapped[str] = mapped_column(String(32), index=True)
    transaction_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    properties: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
