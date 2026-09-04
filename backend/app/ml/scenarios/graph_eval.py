"""Graph-side checks for shared infrastructure scenarios."""

from __future__ import annotations

from typing import Any

from app.graph.factory import graph_status, graph_store
from app.graph.rings import detect_potential_rings


def evaluate_graph_scenario(scenario: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    status = graph_status()
    users = sorted({r["user_id"] for r in rows})
    devices = sorted({r["device_id"] for r in rows})
    ips = sorted({r["ip_address"] for r in rows})
    clusters = detect_potential_rings()
    relevant = []
    user_set = set(users)
    for cluster in clusters:
        cluster_users = set(cluster.get("connected_users") or [])
        if not cluster_users:
            ents = (cluster.get("entities") or {}).get("users") or []
            cluster_users = set(ents)
        if cluster_users & user_set:
            relevant.append(cluster)

    suspicious_nodes = 0
    connected = []
    for user in users:
        try:
            nbs = graph_store.neighbors("USER", user)
        except Exception:
            nbs = []
        other_users = [n for n in nbs if n.get("entity_type") == "USER"]
        connected.append({"user_id": user, "connected_users": [n.get("entity_key") for n in other_users]})
        if len(other_users) >= 1:
            suspicious_nodes += 1

    largest = 0
    if relevant:
        largest = max(int(c.get("user_count") or len(c.get("users") or [])) for c in relevant)

    return {
        "scenario": scenario,
        "graph_backend": status.get("graph_backend") or getattr(graph_store, "name", "networkx"),
        "graph_connected": status.get("graph_connected"),
        "cluster_found": bool(relevant),
        "cluster_count": len(relevant),
        "cluster_size": largest,
        "suspicious_nodes": suspicious_nodes,
        "connected_accounts": connected,
        "shared_devices": devices,
        "shared_ips": ips,
        "users": users,
        "note": "A cluster is a potential ring heuristic, not a confirmed fraud label.",
    }
