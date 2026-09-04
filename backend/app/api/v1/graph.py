from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.deps import get_current_user
from app.graph.factory import graph_status, graph_store
from app.graph.rings import detect_potential_rings, prototype_graph_thresholds
from app.graph.service import score_entity
from app.models.app_user import AppUser
from app.models.cluster import FraudCluster
from app.utils.ids import utcnow
from app.utils.redact import redact_secrets

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/thresholds")
async def graph_thresholds(user: AppUser = Depends(get_current_user)):
    return prototype_graph_thresholds()


@router.get("/status")
async def graph_backend_status(user: AppUser = Depends(get_current_user)):
    return redact_secrets({**graph_status(), **graph_store.get_graph_metrics()})


@router.get("/metrics")
async def graph_metrics(user: AppUser = Depends(get_current_user)):
    return redact_secrets({**graph_status(), **graph_store.get_graph_metrics()})


@router.get("/snapshot")
async def snapshot(user: AppUser = Depends(get_current_user)):
    data = graph_store.snapshot()
    data.update(graph_status())
    return redact_secrets(data)


@router.get("/clusters")
async def clusters(
    persist: bool = False,
    db: AsyncSession = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    found = detect_potential_rings(min_users=3)
    if persist:
        for c in found:
            existing = (
                await db.execute(select(FraudCluster).where(FraudCluster.cluster_id == c["cluster_id"]))
            ).scalar_one_or_none()
            if existing is None:
                db.add(
                    FraudCluster(
                        cluster_id=c["cluster_id"],
                        user_count=c["user_count"],
                        shared_devices=c["shared_devices"],
                        shared_ips=c["shared_ips"],
                        merchants=c["merchants"],
                        entities=c["entities"],
                        graph_risk_score=c["graph_risk_score"],
                        explanation=c["explanation"],
                        detected_at=utcnow(),
                    )
                )
        await db.commit()
    return found


@router.post("/reset")
async def reset_graph(user: AppUser = Depends(get_current_user)):
    """Development reset: wipe the active graph backend. Does not run Cypher from the client."""
    if user.role != "ADMIN":
        raise HTTPException(status_code=403, detail="Only ADMIN can reset the graph")
    if get_settings().is_production:
        raise HTTPException(status_code=403, detail="Graph reset is disabled in production")
    graph_store.clear()
    return redact_secrets(
        {
            "ok": True,
            "graph_backend": getattr(graph_store, "name", None),
            "note": "Graph cleared. SQL transactions were not deleted. Restart or replay ingest to rebuild.",
        }
    )


@router.get("/{entity_id:path}")
async def get_entity(entity_id: str, user: AppUser = Depends(get_current_user)):
    if ":" in entity_id:
        etype, key = entity_id.split(":", 1)
    else:
        etype, key = "USER", entity_id
    return score_entity(etype.upper(), key)
