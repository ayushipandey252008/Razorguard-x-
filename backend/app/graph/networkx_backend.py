"""In-memory NetworkX graph. Default GRAPH_BACKEND=networkx.

Rebuild from persisted SQL transactions on process start. This is not a
durable graph store.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from app.graph.models import (
    REL_LOCATED_AT,
    REL_PAID_MERCHANT,
    REL_USED_DEVICE,
    REL_USED_IP,
    REL_USED_PAYMENT,
    node_id,
)
from app.utils.logging import Timer, get_logger

log = get_logger("graph.networkx")


class NetworkXGraphStore:
    """In-memory graph. Rebuild from persisted edges on process start."""

    name = "networkx"

    def __init__(self) -> None:
        self.g = nx.Graph()
        self.last_timings_ms: dict[str, float] = {}

    def upsert_entity(self, entity_type: str, entity_key: str, properties: dict | None = None) -> str:
        nid = node_id(entity_type, entity_key)
        if nid not in self.g:
            self.g.add_node(nid, entity_type=entity_type, entity_key=entity_key, properties=properties or {})
        elif properties:
            self.g.nodes[nid]["properties"] = {
                **(self.g.nodes[nid].get("properties") or {}),
                **properties,
            }
        return nid

    def add_relationship(
        self,
        from_type: str,
        from_key: str,
        to_type: str,
        to_key: str,
        rel_type: str,
        properties: dict | None = None,
    ) -> None:
        a = self.upsert_entity(from_type, from_key)
        b = self.upsert_entity(to_type, to_key)
        data = self.g.get_edge_data(a, b) or {}
        rels = set(data.get("rel_types") or [])
        rels.add(rel_type)
        count = int(data.get("count") or 0) + 1
        extra = {k: v for k, v in (properties or {}).items() if k not in {"rel_types", "count"}}
        self.g.add_edge(a, b, rel_types=list(rels), count=count, **extra)

    def upsert_relationship(
        self,
        from_type: str,
        from_key: str,
        to_type: str,
        to_key: str,
        rel_type: str,
        properties: dict | None = None,
    ) -> None:
        self.add_relationship(from_type, from_key, to_type, to_key, rel_type, properties)

    def ingest_payment(self, txn: dict) -> None:
        """Same USER–DEVICE/IP/MERCHANT/LOCATION/PAYMENT edges as before this phase."""
        timer = Timer()
        user = txn["user_id"]
        device = txn["device_id"]
        ip = txn["ip_address"]
        merchant = txn["merchant_id"]
        location = txn["location"]
        payment = txn["payment_identifier"]
        self.upsert_entity("USER", user, {"account_age_days": txn.get("account_age_days")})
        self.upsert_entity("DEVICE", device)
        self.upsert_entity("IP", ip)
        self.upsert_entity("MERCHANT", merchant, {"category": txn.get("merchant_category")})
        self.upsert_entity("LOCATION", location)
        self.upsert_entity("PAYMENT", payment)
        props = {"txn": txn["transaction_id"]}
        self.add_relationship("USER", user, "DEVICE", device, REL_USED_DEVICE, props)
        self.add_relationship("USER", user, "IP", ip, REL_USED_IP, props)
        self.add_relationship("USER", user, "MERCHANT", merchant, REL_PAID_MERCHANT, props)
        self.add_relationship("USER", user, "LOCATION", location, REL_LOCATED_AT, props)
        self.add_relationship("USER", user, "PAYMENT", payment, REL_USED_PAYMENT, props)
        self.last_timings_ms["upsert"] = timer.ms()

    def get_entity(self, entity_type: str, entity_key: str) -> dict[str, Any] | None:
        nid = node_id(entity_type, entity_key)
        if nid not in self.g:
            return None
        data = self.g.nodes[nid]
        return {
            "id": nid,
            "entity_type": data.get("entity_type"),
            "entity_key": data.get("entity_key"),
            "properties": data.get("properties") or {},
            "degree": int(self.g.degree(nid)),
        }

    def neighbors(self, entity_type: str, entity_key: str) -> list[dict[str, Any]]:
        timer = Timer()
        nid = node_id(entity_type, entity_key)
        if nid not in self.g:
            self.last_timings_ms["neighbors"] = timer.ms()
            return []
        out = []
        for nb in self.g.neighbors(nid):
            data = self.g.nodes[nb]
            edge = self.g.get_edge_data(nid, nb) or {}
            out.append(
                {
                    "entity_type": data.get("entity_type"),
                    "entity_key": data.get("entity_key"),
                    "rel_types": edge.get("rel_types") or [],
                    "count": edge.get("count") or 1,
                }
            )
        self.last_timings_ms["neighbors"] = timer.ms()
        return out

    def get_neighbors(self, entity_type: str, entity_key: str) -> list[dict[str, Any]]:
        return self.neighbors(entity_type, entity_key)

    def users_sharing_entity(self, entity_type: str, entity_key: str) -> list[str]:
        nid = node_id(entity_type, entity_key)
        if nid not in self.g:
            return []
        users = []
        for nb in self.g.neighbors(nid):
            data = self.g.nodes[nb]
            if data.get("entity_type") == "USER":
                users.append(data["entity_key"])
        return sorted(set(users))

    def connected_users(self, user_id: str, depth: int = 2) -> list[str]:
        """Users reachable only via DEVICE or IP (not merchant/location hops)."""
        timer = Timer()
        start = node_id("USER", user_id)
        if start not in self.g:
            self.last_timings_ms["connected"] = timer.ms()
            return []
        allowed = {"USER", "DEVICE", "IP"}
        found: set[str] = set()
        visited = {start}
        frontier = [start]
        hops = max(1, min(int(depth), 8))
        for _ in range(hops):
            nxt: list[str] = []
            for node in frontier:
                for nb in self.g.neighbors(node):
                    if nb in visited:
                        continue
                    et = self.g.nodes[nb].get("entity_type")
                    if et not in allowed:
                        continue
                    visited.add(nb)
                    nxt.append(nb)
                    key = self.g.nodes[nb].get("entity_key")
                    if et == "USER" and key != user_id:
                        found.add(key)
            frontier = nxt
        self.last_timings_ms["connected"] = timer.ms()
        return sorted(found)

    def find_connected_accounts(self, user_id: str, depth: int = 2) -> list[str]:
        return self.connected_users(user_id, depth=depth)

    def entity_degree(self, entity_type: str, entity_key: str) -> int:
        nid = node_id(entity_type, entity_key)
        if nid not in self.g:
            return 0
        return int(self.g.degree(nid))

    def user_user_projection(self) -> nx.Graph:
        """Users connected if they share a DEVICE or IP."""
        uu = nx.Graph()
        for node, data in self.g.nodes(data=True):
            if data.get("entity_type") not in {"DEVICE", "IP"}:
                continue
            users = [
                self.g.nodes[nb]["entity_key"]
                for nb in self.g.neighbors(node)
                if self.g.nodes[nb].get("entity_type") == "USER"
            ]
            for i, a in enumerate(users):
                uu.add_node(a)
                for b in users[i + 1 :]:
                    if uu.has_edge(a, b):
                        uu[a][b]["weight"] += 1
                        uu[a][b]["via"].add(node)
                    else:
                        uu.add_edge(a, b, weight=1, via={node})
        return uu

    def get_graph_metrics(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for _, data in self.g.nodes(data=True):
            et = data.get("entity_type") or "UNKNOWN"
            counts[et] = counts.get(et, 0) + 1
        rel_counts: dict[str, int] = {}
        for _, _, data in self.g.edges(data=True):
            for rel in data.get("rel_types") or []:
                rel_counts[rel] = rel_counts.get(rel, 0) + 1
        return {
            "backend": self.name,
            "node_count": self.g.number_of_nodes(),
            "edge_count": self.g.number_of_edges(),
            "entity_counts": counts,
            "relationship_counts": rel_counts,
            "timings_ms": dict(self.last_timings_ms),
        }

    def snapshot(self) -> dict[str, Any]:
        nodes = []
        for node, data in self.g.nodes(data=True):
            nodes.append(
                {
                    "id": node,
                    "entity_type": data.get("entity_type"),
                    "entity_key": data.get("entity_key"),
                    "degree": int(self.g.degree(node)),
                }
            )
        edges = []
        for a, b, data in self.g.edges(data=True):
            edges.append(
                {
                    "source": a,
                    "target": b,
                    "rel_types": data.get("rel_types") or [],
                    "count": data.get("count") or 1,
                }
            )
        return {
            "backend": self.name,
            "nodes": nodes,
            "edges": edges,
            "metrics": self.get_graph_metrics(),
        }

    def clear(self) -> None:
        self.g.clear()
        self.last_timings_ms = {}

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        return None


graph_store = NetworkXGraphStore()
