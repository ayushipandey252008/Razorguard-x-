from __future__ import annotations

import uuid
from datetime import datetime, timezone


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def synthetic_txn_id() -> str:
    return f"txn_{uuid.uuid4().hex[:16]}"
