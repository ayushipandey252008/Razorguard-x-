from __future__ import annotations

import hashlib
from typing import Any

import networkx as nx

from app.config import get_settings
from app.graph.factory import graph_store


def prototype_graph_thresholds() -> dict[str, Any]:
    """Configurable prototype ring indicators. Not production-grade."""
    settings = get_settings()
    return {
        "min_cluster_users": settings.graph_min_cluster_users,
        "shared_device_accounts": settings.graph_shared_device_accounts,
        "shared_ip_accounts": settings.graph_shared_ip_accounts,
        "flagged_path_max_hops": settings.graph_flagged_path_max_hops,
        "dense_subgraph_min_density": settings.graph_dense_subgraph_min_density,
        "suspicious_entity_degree": settings.graph_suspicious_entity_degree,
        "note": (
            "These graph thresholds are configurable prototype heuristics. "
            "They are not production-grade fraud-ring detectors."
        ),
    }


def _min_cluster_users(min_users: int | None = None) -> int:
    if min_users is not None:
        return min_users
    return get_settings().graph_min_cluster_users


def _cluster_id(users: list[str]) -> str:
    key = "|".join(sorted(users))
    return "cl_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def detect_potential_rings(min_users: int | None = None) -> list[dict]:
    """Find user clusters that share devices or IPs.

    A cluster is a *potential* fraud ring, not a confirmed one.
    """
    from app.utils.logging import Timer

    timer = Timer()
    threshold = _min_cluster_users(min_users)
    uu = graph_store.user_user_projection()
    clusters = []
    for component in nx.connected_components(uu):
        users = sorted(component)
        if len(users) < threshold:
            continue
        clusters.append(_describe_cluster(users, uu))
    clusters.sort(key=lambda c: c["graph_risk_score"], reverse=True)
    graph_store.last_timings_ms["cluster"] = timer.ms()
    return clusters


def _describe_cluster(users: list[str], uu: nx.Graph | None = None) -> dict:
    if uu is None:
        uu = graph_store.user_user_projection()
    subgraph = uu.subgraph(users)
    devices: set[str] = set()
    ips: set[str] = set()
    for _, _, data in subgraph.edges(data=True):
        for via in data.get("via") or []:
            kind, _, key = via.partition(":")
            if kind == "DEVICE":
                devices.add(key)
            elif kind == "IP":
                ips.add(key)
    merchants: set[str] = set()
    for user in users:
        for nb in graph_store.neighbors("USER", user):
            if nb["entity_type"] == "MERCHANT":
                merchants.add(nb["entity_key"])
    n = len(users)
    density = 0.0
    if n > 1:
        density = subgraph.number_of_edges() / (n * (n - 1) / 2)
    # Score from evidence only: extra users, extra shared infra, density cap.
    score = min(
        100.0,
        8 * min(n, 8) + 14 * min(len(devices), 4) + 14 * min(len(ips), 4) + 20 * min(density, 1.0),
    )
    explanation = (
        f"Potential fraud ring: {n} users share {len(devices)} device(s) and "
        f"{len(ips)} IP(s). This is a graph-structure signal, not a confirmed fraud label."
    )
    relationships = []
    for a, b, data in subgraph.edges(data=True):
        relationships.append(
            {
                "from_user": a,
                "to_user": b,
                "via": sorted(data.get("via") or []),
                "weight": data.get("weight"),
            }
        )
    thresholds = prototype_graph_thresholds()
    risk_indicators = _structural_indicators(
        users=users,
        devices=devices,
        ips=ips,
        density=density,
        thresholds=thresholds,
    )
    suspicious_nodes = [
        {"entity_type": "USER", "entity_key": u, "reason": "member_of_shared_infra_cluster"}
        for u in users
    ]
    return {
        "identified": True,
        "cluster_found": True,
        "cluster_id": _cluster_id(users),
        "connected_users": users,
        "user_count": n,
        "cluster_size": n,
        "shared_devices": sorted(devices),
        "shared_ips": sorted(ips),
        "shared_device_count": len(devices),
        "shared_ip_count": len(ips),
        "merchants": sorted(merchants),
        "relationship_counts": {
            "user_user_edges": int(subgraph.number_of_edges()),
            "shared_device_count": len(devices),
            "shared_ip_count": len(ips),
            "merchant_count": len(merchants),
            "density": round(density, 3),
        },
        "relationships": relationships,
        "suspicious_nodes": suspicious_nodes,
        "risk_indicators": risk_indicators,
        "fraud_associated_nodes": 0,
        "entities": {"users": users, "density": round(density, 3)},
        "graph_risk": round(score, 2),
        "graph_risk_score": round(score, 2),
        "explanation": explanation,
        "thresholds_used": {
            "min_cluster_users": thresholds["min_cluster_users"],
            "note": thresholds["note"],
        },
        "graph_backend": getattr(graph_store, "name", "networkx"),
    }


def cluster_for_user(user_id: str, min_users: int | None = None) -> dict | None:
    for cluster in detect_potential_rings(min_users=min_users):
        if user_id in cluster.get("connected_users", []):
            return cluster
    return None


def cluster_for_transaction(txn: dict, min_users: int | None = None) -> dict:
    """Inspect this payment's user, device, and IP for a potential ring.

    Returns identified=False with an explicit message when evidence is insufficient.
    """
    user_id = txn["user_id"]
    threshold = _min_cluster_users(min_users)
    device_users = graph_store.users_sharing_entity("DEVICE", txn["device_id"])
    ip_users = graph_store.users_sharing_entity("IP", txn["ip_address"])
    existing = cluster_for_user(user_id, min_users=threshold)

    ad_hoc_users = sorted(set(device_users) | set(ip_users) | {user_id})
    if existing is None and len(ad_hoc_users) >= threshold:
        existing = _describe_cluster(ad_hoc_users)

    base = {
        "transaction_id": txn.get("transaction_id"),
        "user_id": user_id,
        "device_id": txn.get("device_id"),
        "ip_address": txn.get("ip_address"),
        "device_users": device_users,
        "ip_users": ip_users,
        "relationship_counts": {
            "device_user_count": len(device_users),
            "ip_user_count": len(ip_users),
            "shared_infra_users": len(set(device_users) | set(ip_users)),
        },
    }
    if existing is None:
        thresholds = prototype_graph_thresholds()
        indicators = _entity_share_indicators(
            device_id=txn.get("device_id"),
            ip_address=txn.get("ip_address"),
            device_users=device_users,
            ip_users=ip_users,
            thresholds=thresholds,
        )
        return {
            **base,
            "identified": False,
            "cluster_found": False,
            "reason": "No connected suspicious cluster found",
            "message": "no suspicious cluster identified",
            "explanation": (
                "No group of three or more users sharing this device or IP was found. "
                "Shared-merchant hops are not treated as a fraud ring."
            ),
            "connected_users": graph_store.connected_users(user_id, depth=2),
            "shared_devices": [],
            "shared_ips": [],
            "merchants": [],
            "cluster_size": 0,
            "fraud_associated_nodes": 0,
            "suspicious_nodes": [],
            "relationships": [],
            "risk_indicators": indicators,
            "graph_risk": 0.0,
            "graph_risk_score": 0.0,
            "thresholds_used": {
                "min_cluster_users": thresholds["min_cluster_users"],
                "note": thresholds["note"],
            },
            "graph_backend": getattr(graph_store, "name", "networkx"),
        }
    return {**base, **existing}


def _structural_indicators(
    *,
    users: list[str],
    devices: set[str],
    ips: set[str],
    density: float,
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    indicators: list[dict[str, Any]] = []
    for device in sorted(devices):
        sharing = graph_store.users_sharing_entity("DEVICE", device)
        if len(sharing) >= thresholds["shared_device_accounts"]:
            indicators.append(
                {
                    "code": "DEVICE_SHARED_ACCOUNTS",
                    "detail": (
                        f"Device {device} is shared by {len(sharing)} accounts "
                        f"(prototype threshold {thresholds['shared_device_accounts']})."
                    ),
                    "entity_type": "DEVICE",
                    "entity_key": device,
                    "account_count": len(sharing),
                }
            )
        if graph_store.entity_degree("DEVICE", device) >= thresholds["suspicious_entity_degree"]:
            indicators.append(
                {
                    "code": "SUSPICIOUS_ENTITY_DEGREE",
                    "detail": f"Device {device} has degree {graph_store.entity_degree('DEVICE', device)}.",
                    "entity_type": "DEVICE",
                    "entity_key": device,
                }
            )
    for ip in sorted(ips):
        sharing = graph_store.users_sharing_entity("IP", ip)
        if len(sharing) >= thresholds["shared_ip_accounts"]:
            indicators.append(
                {
                    "code": "IP_SHARED_ACCOUNTS",
                    "detail": (
                        f"IP {ip} is shared by {len(sharing)} accounts "
                        f"(prototype threshold {thresholds['shared_ip_accounts']})."
                    ),
                    "entity_type": "IP",
                    "entity_key": ip,
                    "account_count": len(sharing),
                }
            )
    if density >= thresholds["dense_subgraph_min_density"] and len(users) >= thresholds["min_cluster_users"]:
        indicators.append(
            {
                "code": "DENSE_SUSPICIOUS_SUBGRAPH",
                "detail": (
                    f"User-user subgraph density {round(density, 3)} meets prototype "
                    f"threshold {thresholds['dense_subgraph_min_density']}."
                ),
                "density": round(density, 3),
            }
        )
    return indicators


def _entity_share_indicators(
    *,
    device_id: str | None,
    ip_address: str | None,
    device_users: list[str],
    ip_users: list[str],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    indicators: list[dict[str, Any]] = []
    if device_id and len(device_users) >= thresholds["shared_device_accounts"]:
        indicators.append(
            {
                "code": "DEVICE_SHARED_ACCOUNTS",
                "detail": (
                    f"Device {device_id} is shared by {len(device_users)} accounts "
                    f"(prototype threshold {thresholds['shared_device_accounts']})."
                ),
                "entity_type": "DEVICE",
                "entity_key": device_id,
                "account_count": len(device_users),
            }
        )
    if ip_address and len(ip_users) >= thresholds["shared_ip_accounts"]:
        indicators.append(
            {
                "code": "IP_SHARED_ACCOUNTS",
                "detail": (
                    f"IP {ip_address} is shared by {len(ip_users)} accounts "
                    f"(prototype threshold {thresholds['shared_ip_accounts']})."
                ),
                "entity_type": "IP",
                "entity_key": ip_address,
                "account_count": len(ip_users),
            }
        )
    return indicators


def _short_paths(flagged_users: list[str], max_hops: int) -> list[dict[str, Any]]:
    if len(flagged_users) < 2:
        return []
    uu = graph_store.user_user_projection()
    paths: list[dict[str, Any]] = []
    for i, a in enumerate(flagged_users):
        for b in flagged_users[i + 1 :]:
            if a not in uu or b not in uu:
                continue
            try:
                length = nx.shortest_path_length(uu, a, b)
            except nx.NetworkXNoPath:
                continue
            if length <= max_hops:
                paths.append(
                    {
                        "from_user": a,
                        "to_user": b,
                        "hops": length,
                        "path": nx.shortest_path(uu, a, b),
                    }
                )
    return paths


def enrich_cluster(
    cluster: dict,
    *,
    flagged_user_ids: set[str] | None = None,
    flagged_transaction_count: int = 0,
) -> dict:
    """Attach DB-backed flag counts. Does not invent users or edges."""
    flagged = set(flagged_user_ids or [])
    users = list(cluster.get("connected_users") or [])
    device_users = list(cluster.get("device_users") or [])
    thresholds = prototype_graph_thresholds()
    indicators = list(cluster.get("risk_indicators") or [])
    flagged_in_cluster = sorted(flagged & set(users or device_users))
    cluster["fraud_associated_nodes"] = len(flagged_in_cluster)
    cluster["fraud_associated_user_ids"] = flagged_in_cluster
    cluster["flagged_transaction_count"] = flagged_transaction_count

    if cluster.get("identified"):
        cluster["cluster_found"] = True
        cluster["cluster_size"] = cluster.get("cluster_size") or cluster.get("user_count") or len(users)
    else:
        cluster["cluster_found"] = False
        cluster.setdefault("reason", "No connected suspicious cluster found")

    device_id = cluster.get("device_id")
    if device_id:
        flagged_on_device = sorted(flagged & set(device_users))
        if len(flagged_on_device) >= 2:
            indicators.append(
                {
                    "code": "MULTIPLE_FLAGGED_ACCOUNTS_ON_DEVICE",
                    "detail": (
                        f"{len(flagged_on_device)} previously flagged accounts share device {device_id}."
                    ),
                    "user_ids": flagged_on_device,
                }
            )
    if flagged_transaction_count >= 2 and (cluster.get("identified") or device_users):
        indicators.append(
            {
                "code": "FLAGGED_TRANSACTIONS_CONNECTED",
                "detail": (
                    f"{flagged_transaction_count} previously flagged transactions are connected "
                    "through this entity set."
                ),
                "count": flagged_transaction_count,
            }
        )
    paths = _short_paths(flagged_in_cluster, thresholds["flagged_path_max_hops"])
    if paths:
        indicators.append(
            {
                "code": "SHORT_PATH_BETWEEN_SUSPICIOUS_ACCOUNTS",
                "detail": (
                    f"{len(paths)} short path(s) (≤{thresholds['flagged_path_max_hops']} hops) "
                    "between previously flagged accounts."
                ),
                "paths": paths[:8],
            }
        )
    suspicious = list(cluster.get("suspicious_nodes") or [])
    known = {(n.get("entity_type"), n.get("entity_key")) for n in suspicious}
    for uid in flagged_in_cluster:
        key = ("USER", uid)
        if key not in known:
            suspicious.append(
                {
                    "entity_type": "USER",
                    "entity_key": uid,
                    "reason": "previously_flagged_review_or_block",
                }
            )
    cluster["suspicious_nodes"] = suspicious
    cluster["risk_indicators"] = indicators
    return cluster
