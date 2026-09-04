from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("payment_users.user_id"), index=True)
    merchant_id: Mapped[str] = mapped_column(ForeignKey("merchants.merchant_id"), index=True)
    amount: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(8), default="INR")
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    device_id: Mapped[str] = mapped_column(String(64), index=True)
    ip_address: Mapped[str] = mapped_column(String(64), index=True)
    location: Mapped[str] = mapped_column(String(128), index=True)
    payment_method: Mapped[str] = mapped_column(String(32))
    merchant_category: Mapped[str] = mapped_column(String(64))
    account_age_days: Mapped[int] = mapped_column(Integer)
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    transaction_velocity: Mapped[int] = mapped_column(Integer, default=1)
    previous_transaction_count: Mapped[int] = mapped_column(Integer, default=0)
    previous_average_amount: Mapped[float] = mapped_column(Float, default=0.0)
    current_device_known: Mapped[bool] = mapped_column(Boolean, default=True)
    current_location_known: Mapped[bool] = mapped_column(Boolean, default=True)
    payment_identifier: Mapped[str] = mapped_column(String(64), index=True)
    scenario_tag: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
