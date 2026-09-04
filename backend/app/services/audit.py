from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit import AuditLog
from app.utils.ids import new_id, utcnow


async def write_audit(
    db: AsyncSession,
    actor: str,
    action: str,
    entity_type: str,
    entity_id: str,
    metadata: dict | None = None,
) -> None:
    db.add(
        AuditLog(
            id=new_id(),
            timestamp=utcnow(),
            actor=actor,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            extra=metadata or {},
        )
    )
