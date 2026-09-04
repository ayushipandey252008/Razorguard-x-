from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.factory import graph_store
from app.graph.rings import cluster_for_user
from app.models.graph import GraphEntity, GraphRelationship
from app.utils.ids import new_id
from app.utils.logging import Timer, get_logger

log = get_logger("graph.service")


def ingest_transaction(txn: dict) -> dict[str, Any]:
    """Update the live graph from a transaction and return graph evidence.

    Scoring formula is unchanged. Persistence backend is NetworkX or Neo4j.
    """
    user = txn["user_id"]
    device = txn["device_id"]
    ip = txn["ip_address"]
    t_up = Timer()
    graph_store.ingest_payment(txn)
    upsert_ms = graph_store.last_timings_ms.get("upsert", t_up.ms())

    t_nb = Timer()
    device_users = graph_store.users_sharing_entity("DEVICE", device)
    ip_users = graph_store.users_sharing_entity("IP", ip)
    connected = graph_store.connected_users(user, depth=2)
    neighbor_ms = graph_store.last_timings_ms.get("neighbors", t_nb.ms())
    connected_ms = graph_store.last_timings_ms.get("connected", 0.0)
    device_share = max(0, len(device_users) - 1)
    ip_share = max(0, len(ip_users) - 1)

    merchant_overlap = 0
    if device_share or ip_share:
        peer_merchants: set[str] = set()
        my_merchants = {
            nb["entity_key"]
            for nb in graph_store.neighbors("USER", user)
            if nb["entity_type"] == "MERCHANT"
        }
        for peer in set(device_users + ip_users) - {user}:
            peer_merchants.update(
                nb["entity_key"]
                for nb in graph_store.neighbors("USER", peer)
                if nb["entity_type"] == "MERCHANT"
            )
        merchant_overlap = len(my_merchants & peer_merchants)

    t_cl = Timer()
    cluster = cluster_for_user(user, min_users=3)
    cluster_ms = graph_store.last_timings_ms.get("cluster", t_cl.ms())
    cluster_component = 0.0
    if cluster:
        cluster_component = min(24.0, 0.25 * float(cluster["graph_risk_score"]))

    # Device/IP sharing and cluster membership only — merchant-only hops do not add score.
    graph_score = min(
        100.0,
        22 * min(device_share, 4) + 18 * min(ip_share, 4) + 6 * min(merchant_overlap, 3) + cluster_component,
    )

    timings = {
        "upsert": upsert_ms,
        "neighbors": neighbor_ms,
        "connected": connected_ms,
        "cluster": cluster_ms,
    }
    log.info(
        "graph_ingest",
        graph_backend=getattr(graph_store, "name", None),
        timings_ms=timings,
        transaction_id=txn.get("transaction_id"),
    )

    return {
        "connected_users": connected,
        "device_users": device_users,
        "ip_users": ip_users,
        "device_user_count": len(device_users),
        "ip_user_count": len(ip_users),
        "connected_entity_count": graph_store.entity_degree("USER", user),
        "merchant_overlap_with_shared_infra_peers": merchant_overlap,
        "cluster_id": cluster["cluster_id"] if cluster else None,
        "cluster_user_count": cluster["user_count"] if cluster else 0,
        "graph_score": round(graph_score, 2),
        "suspicious_relationship_count": int(device_share + ip_share),
        "score_basis": {
            "shared_device_users_beyond_self": device_share,
            "shared_ip_users_beyond_self": ip_share,
            "merchant_overlap": merchant_overlap,
            "cluster_component": round(cluster_component, 2),
        },
        "graph_backend": getattr(graph_store, "name", "networkx"),
        "timings_ms": timings,
    }


def score_entity(entity_type: str, entity_key: str) -> dict[str, Any]:
    """Score a single entity from shared-infra evidence, not merchant hops."""
    neighbors = graph_store.neighbors(entity_type, entity_key)
    if entity_type == "USER":
        device_share = 0
        ip_share = 0
        for nb in neighbors:
            if nb["entity_type"] == "DEVICE":
                device_share += max(0, len(graph_store.users_sharing_entity("DEVICE", nb["entity_key"])) - 1)
            elif nb["entity_type"] == "IP":
                ip_share += max(0, len(graph_store.users_sharing_entity("IP", nb["entity_key"])) - 1)
        connected_users = graph_store.connected_users(entity_key, depth=2)
        cluster = cluster_for_user(entity_key, min_users=3)
        cluster_component = min(24.0, 0.25 * float(cluster["graph_risk_score"])) if cluster else 0.0
        score = min(100.0, 22 * min(device_share, 4) + 18 * min(ip_share, 4) + cluster_component)
        return {
            "entity_id": f"{entity_type}:{entity_key}",
            "entity_type": entity_type,
            "neighbors": neighbors,
            "connected_users": connected_users,
            "graph_risk_score": round(score, 2),
            "graph_backend": getattr(graph_store, "name", "networkx"),
            "evidence": {
                "degree": graph_store.entity_degree(entity_type, entity_key),
                "shared_device_users_beyond_self": device_share,
                "shared_ip_users_beyond_self": ip_share,
                "cluster_id": cluster["cluster_id"] if cluster else None,
                "cluster_component": round(cluster_component, 2),
                "note": "Score uses shared device/IP and cluster membership, not merchant-only hops.",
            },
        }

    connected_users = graph_store.users_sharing_entity(entity_type, entity_key)
    share = max(0, len(connected_users) - 1)
    per = 22 if entity_type == "DEVICE" else 18 if entity_type == "IP" else 6
    score = min(100.0, per * min(share, 4))
    return {
        "entity_id": f"{entity_type}:{entity_key}",
        "entity_type": entity_type,
        "neighbors": neighbors,
        "connected_users": connected_users,
        "graph_risk_score": round(score, 2),
        "graph_backend": getattr(graph_store, "name", "networkx"),
        "evidence": {
            "degree": graph_store.entity_degree(entity_type, entity_key),
            "users_sharing_entity": connected_users,
            "share_beyond_one": share,
        },
    }


async def persist_transaction_graph(db: AsyncSession, txn: dict) -> None:
    """Upsert graph_entities / graph_relationships without duplicating entity keys."""

    async def get_or_create(entity_type: str, entity_key: str) -> GraphEntity:
        existing = (
            await db.execute(
                select(GraphEntity).where(
                    GraphEntity.entity_type == entity_type,
                    GraphEntity.entity_key == entity_key,
                )
            )
        ).scalar_one_or_none()
        if existing:
            return existing
        row = GraphEntity(
            id=new_id(),
            entity_type=entity_type,
            entity_key=entity_key,
            properties={},
            risk_score=0.0,
        )
        db.add(row)
        await db.flush()
        return row

    user = await get_or_create("USER", txn["user_id"])
    pairs = [
        ("DEVICE", txn["device_id"], "used_device"),
        ("IP", txn["ip_address"], "used_ip"),
        ("MERCHANT", txn["merchant_id"], "paid_merchant"),
        ("LOCATION", txn["location"], "located_at"),
        ("PAYMENT", txn["payment_identifier"], "used_payment"),
    ]
    for etype, key, rel in pairs:
        node = await get_or_create(etype, key)
        db.add(
            GraphRelationship(
                id=new_id(),
                from_id=user.id,
                to_id=node.id,
                rel_type=rel,
                transaction_id=txn["transaction_id"],
                properties={},
            )
        )
