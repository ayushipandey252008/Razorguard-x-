from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.merchant import Merchant
from app.models.payment_user import PaymentUser
from app.models.transaction import Transaction
from app.schemas.common import TransactionCreate
from app.utils.ids import synthetic_txn_id, utcnow


def as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)


async def enrich_and_build(db: AsyncSession, payload: TransactionCreate) -> dict:
    user = (
        await db.execute(select(PaymentUser).where(PaymentUser.user_id == payload.user_id))
    ).scalar_one_or_none()
    merchant = (
        await db.execute(select(Merchant).where(Merchant.merchant_id == payload.merchant_id))
    ).scalar_one_or_none()
    if user is None:
        raise ValueError(f"Unknown payment user: {payload.user_id}")
    if merchant is None:
        raise ValueError(f"Unknown merchant: {payload.merchant_id}")

    ts = as_utc(payload.timestamp) or utcnow()
    window_start = ts - timedelta(hours=1)

    hist = (
        await db.execute(
            select(Transaction).where(Transaction.user_id == payload.user_id).order_by(Transaction.timestamp.desc())
        )
    ).scalars().all()

    prev_count = payload.previous_transaction_count
    if prev_count is None:
        prev_count = len(hist)
    prev_avg = payload.previous_average_amount
    if prev_avg is None:
        prev_avg = (sum(t.amount for t in hist) / len(hist)) if hist else user.typical_amount

    velocity = payload.transaction_velocity
    if velocity is None:
        velocity = 1 + sum(1 for t in hist if t.timestamp and as_utc(t.timestamp) >= window_start)

    device_known = payload.current_device_known
    if device_known is None:
        known = set(user.known_devices or [])
        known.update(t.device_id for t in hist)
        device_known = payload.device_id in known

    loc_known = payload.current_location_known
    if loc_known is None:
        known_loc = set(user.known_locations or [])
        known_loc.update(t.location for t in hist)
        loc_known = payload.location in known_loc

    txn_id = synthetic_txn_id()
    return {
        "transaction_id": txn_id,
        "user_id": payload.user_id,
        "merchant_id": payload.merchant_id,
        "amount": float(payload.amount),
        "currency": payload.currency,
        "timestamp": ts,
        "device_id": payload.device_id,
        "ip_address": payload.ip_address,
        "location": payload.location,
        "payment_method": payload.payment_method,
        "merchant_category": payload.merchant_category or merchant.category,
        "account_age_days": payload.account_age_days if payload.account_age_days is not None else user.account_age_days,
        "failed_attempts": payload.failed_attempts,
        "transaction_velocity": int(velocity),
        "previous_transaction_count": int(prev_count),
        "previous_average_amount": float(prev_avg),
        "current_device_known": bool(device_known),
        "current_location_known": bool(loc_known),
        "payment_identifier": payload.payment_identifier or f"pay_{txn_id[-10:]}",
        "scenario_tag": payload.scenario_tag,
        "_user": {
            "typical_amount": user.typical_amount,
            "typical_hour": user.typical_hour,
            "known_devices": user.known_devices,
            "known_locations": user.known_locations,
            "home_location": user.home_location,
        },
        "_merchant_watchlisted": bool(merchant.is_watchlisted),
    }
