from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class PaymentUser(Base):
    __tablename__ = "payment_users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_age_days: Mapped[int] = mapped_column(Integer)
    home_location: Mapped[str] = mapped_column(String(128))
    typical_amount: Mapped[float] = mapped_column(Float)
    typical_hour: Mapped[int] = mapped_column(Integer, default=14)
    known_devices: Mapped[list] = mapped_column(JSON, default=list)
    known_locations: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
