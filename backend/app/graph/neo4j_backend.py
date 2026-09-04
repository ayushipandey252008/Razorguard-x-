"""Optional Neo4j GraphBackend.

Uses parameterized Cypher only. Entity types and relationship types are
allow-listed. Untrusted transaction/user strings are bound as parameters
and treated as DATA.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from app.graph.models import (
    ALLOWED_ENTITY_TYPES,
    ALLOWED_REL_TYPES,
    NEO4J_LABELS,
    node_id,
)
from app.utils.logging import Timer, get_logger

log = get_logger("graph.neo4j")

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - driver optional for NetworkX-only installs
    GraphDatabase = None  # type: ignore[assignment]


class Neo4jUnavailable(RuntimeError):
    """Neo4j was requested but cannot be used."""


def _require_driver():
    if GraphDatabase is None:
        raise Neo4jUnavailable(
            "The neo4j Python driver is not installed. Install backend/requirements.txt "
            "or set GRAPH_BACKEND=networkx."
        )


class Neo4jGraphStore:
    name = "neo4j"

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        _require_driver()
        self.uri = uri
        self.database = database
        self.last_timings_ms: dict[str, float] = {}
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._driver.verify_connectivity()
        self._ensure_constraints()

    def _session(self):
        return self._driver.session(database=self.database)

    def _ensure_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT user_entity_key IF NOT EXISTS FOR (n:User) REQUIRE n.entity_key IS UNIQUE",
            "CREATE CONSTRAINT device_entity_key IF NOT EXISTS FOR (n:Device) REQUIRE n.entity_key IS UNIQUE",
            "CREATE CONSTRAINT ip_entity_key IF NOT EXISTS FOR (n:IP) REQUIRE n.entity_key IS UNIQUE",
            "CREATE CONSTRAINT location_entity_key IF NOT EXISTS FOR (n:Location) REQUIRE n.entity_key IS UNIQUE",
            "CREATE CONSTRAINT merchant_entity_key IF NOT EXISTS FOR (n:Merchant) REQUIRE n.entity_key IS UNIQUE",
            "CREATE CONSTRAINT txn_entity_key IF NOT EXISTS FOR (n:Transaction) REQUIRE n.entity_key IS UNIQUE",
        ]
        with self._session() as session:
            for stmt in statements:
                session.run(stmt)

    def ping(self) -> bool:
        try:
            with self._session() as session:
                session.run("RETURN 1 AS ok").single()
            return True
        except Exception:
            return False

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass

    def upsert_entity(self, entity_type: str, entity_key: str, properties: dict | None = None) -> str:
        label = NEO4J_LABELS.get((entity_type or "").upper())
        if label is None:
            raise ValueError(f"Unsupported entity type '{entity_type}'")
        props = {k: v for k, v in (properties or {}).items() if k not in {"entity_key", "entity_type"}}
        # Label is allow-listed, not user input. Properties are parameters.
        cypher = (
            f"MERGE (n:{label} {{entity_key: $key}}) "
            "SET n.entity_type = $etype "
            "SET n += $props "
            "RETURN n.entity_key AS key"
        )
        with self._session() as session:
            session.run(cypher, key=entity_key, etype=entity_type.upper(), props=props)
        return node_id(entity_type.upper(), entity_key)

    def add_relationship(
        self,
        from_type: str,
        from_key: str,
        to_type: str,
        to_key: str,
        rel_type: str,
        properties: dict | None = None,
    ) -> None:
        self.upsert_relationship(from_type, from_key, to_type, to_key, rel_type, properties)

    def upsert_relationship(
        self,
        from_type: str,
        from_key: str,
        to_type: str,
        to_key: str,
        rel_type: str,
        properties: dict | None = None,
    ) -> None:
        if rel_type not in ALLOWED_REL_TYPES:
            raise ValueError(f"Unsupported relationship type '{rel_type}'")
        from_label = NEO4J_LABELS.get((from_type or "").upper())
        to_label = NEO4J_LABELS.get((to_type or "").upper())
        if from_label is None or to_label is None:
            raise ValueError("Unsupported entity type on relationship")
        props = {k: v for k, v in (properties or {}).items() if k not in {"rel_types", "count"}}
        cypher = (
            f"MERGE (a:{from_label} {{entity_key: $from_key}}) "
            "SET a.entity_type = $from_type "
            f"MERGE (b:{to_label} {{entity_key: $to_key}}) "
            "SET b.entity_type = $to_type "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "ON CREATE SET r.count = 1 "
            "ON MATCH SET r.count = coalesce(r.count, 0) + 1 "
            "SET r += $props"
        )
        with self._session() as session:
            session.run(
                cypher,
                from_key=from_key,
                to_key=to_key,
                from_type=from_type.upper(),
                to_type=to_type.upper(),
                props=props,
            )

    def ingest_payment(self, txn: dict) -> None:
        """One parameterized write: Transaction node + user/device/IP/merchant/location."""
        timer = Timer()
        cypher = """
        MERGE (u:User {entity_key: $user_id})
          SET u.entity_type = 'USER', u.account_age_days = $account_age_days
        MERGE (d:Device {entity_key: $device_id})
          SET d.entity_type = 'DEVICE'
        MERGE (ip:IP {entity_key: $ip_address})
          SET ip.entity_type = 'IP'
        MERGE (m:Merchant {entity_key: $merchant_id})
          SET m.entity_type = 'MERCHANT', m.category = $merchant_category
        MERGE (loc:Location {entity_key: $location})
          SET loc.entity_type = 'LOCATION'
        MERGE (t:Transaction {entity_key: $transaction_id})
          SET t.entity_type = 'TRANSACTION',
              t.amount = $amount,
              t.currency = $currency,
              t.merchant_category = $merchant_category
        MERGE (u)-[made:MADE]->(t)
          ON CREATE SET made.count = 1
          ON MATCH SET made.count = coalesce(made.count, 0) + 1
        MERGE (t)-[ud:USED_DEVICE]->(d)
          ON CREATE SET ud.count = 1
          ON MATCH SET ud.count = coalesce(ud.count, 0) + 1
        MERGE (t)-[ui:USED_IP]->(ip)
          ON CREATE SET ui.count = 1
          ON MATCH SET ui.count = coalesce(ui.count, 0) + 1
        MERGE (t)-[al:AT_LOCATION]->(loc)
          ON CREATE SET al.count = 1
          ON MATCH SET al.count = coalesce(al.count, 0) + 1
        MERGE (t)-[am:AT_MERCHANT]->(m)
          ON CREATE SET am.count = 1
          ON MATCH SET am.count = coalesce(am.count, 0) + 1
        MERGE (u)-[usd:USES_DEVICE]->(d)
          ON CREATE SET usd.count = 1
          ON MATCH SET usd.count = coalesce(usd.count, 0) + 1
        MERGE (u)-[usi:USES_IP]->(ip)
          ON CREATE SET usi.count = 1
          ON MATCH SET usi.count = coalesce(usi.count, 0) + 1
        MERGE (u)-[pm:PAID_MERCHANT]->(m)
          ON CREATE SET pm.count = 1
          ON MATCH SET pm.count = coalesce(pm.count, 0) + 1
        MERGE (u)-[la:LOCATED_AT]->(loc)
          ON CREATE SET la.count = 1
          ON MATCH SET la.count = coalesce(la.count, 0) + 1
        """
        params = {
            "user_id": txn["user_id"],
            "device_id": txn["device_id"],
            "ip_address": txn["ip_address"],
            "merchant_id": txn["merchant_id"],
            "location": txn["location"],
            "transaction_id": txn["transaction_id"],
            "account_age_days": txn.get("account_age_days"),
            "merchant_category": txn.get("merchant_category"),
            "amount": txn.get("amount"),
            "currency": txn.get("currency"),
        }
        with self._session() as session:
            session.execute_write(lambda tx: tx.run(cypher, **params))
        self.last_timings_ms["upsert"] = timer.ms()
        log.info("graph_upsert", backend="neo4j", duration_ms=self.last_timings_ms["upsert"])

    def get_entity(self, entity_type: str, entity_key: str) -> dict[str, Any] | None:
        if entity_type.upper() not in ALLOWED_ENTITY_TYPES:
            return None
        cypher = (
            "MATCH (n {entity_key: $key, entity_type: $etype}) "
            "RETURN n.entity_key AS entity_key, n.entity_type AS entity_type, "
            "properties(n) AS props, COUNT { (n)--() } AS degree"
        )
        with self._session() as session:
            rec = session.run(cypher, key=entity_key, etype=entity_type.upper()).single()
        if rec is None:
            return None
        props = dict(rec["props"] or {})
        props.pop("entity_key", None)
        props.pop("entity_type", None)
        et = rec["entity_type"]
        key = rec["entity_key"]
        return {
            "id": node_id(et, key),
            "entity_type": et,
            "entity_key": key,
            "properties": props,
            "degree": int(rec["degree"] or 0),
        }

    def neighbors(self, entity_type: str, entity_key: str) -> list[dict[str, Any]]:
        timer = Timer()
        if entity_type.upper() not in ALLOWED_ENTITY_TYPES:
            self.last_timings_ms["neighbors"] = timer.ms()
            return []
        cypher = (
            "MATCH (n {entity_key: $key, entity_type: $etype})-[r]-(m) "
            "RETURN m.entity_type AS entity_type, m.entity_key AS entity_key, "
            "type(r) AS rel_type, coalesce(r.count, 1) AS count"
        )
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        with self._session() as session:
            for rec in session.run(cypher, key=entity_key, etype=entity_type.upper()):
                et, ek = rec["entity_type"], rec["entity_key"]
                item = grouped.setdefault(
                    (et, ek),
                    {"entity_type": et, "entity_key": ek, "rel_types": [], "count": 0},
                )
                rel = rec["rel_type"]
                if rel not in item["rel_types"]:
                    item["rel_types"].append(rel)
                item["count"] += int(rec["count"] or 1)
        self.last_timings_ms["neighbors"] = timer.ms()
        log.info("graph_neighbors", backend="neo4j", duration_ms=self.last_timings_ms["neighbors"])
        return list(grouped.values())

    def get_neighbors(self, entity_type: str, entity_key: str) -> list[dict[str, Any]]:
        return self.neighbors(entity_type, entity_key)

    def users_sharing_entity(self, entity_type: str, entity_key: str) -> list[str]:
        et = (entity_type or "").upper()
        if et == "DEVICE":
            cypher = (
                "MATCH (d:Device {entity_key: $key})<-[:USES_DEVICE]-(u:User) "
                "RETURN DISTINCT u.entity_key AS user_id"
            )
        elif et == "IP":
            cypher = (
                "MATCH (ip:IP {entity_key: $key})<-[:USES_IP]-(u:User) "
                "RETURN DISTINCT u.entity_key AS user_id"
            )
        else:
            cypher = (
                "MATCH (n {entity_key: $key, entity_type: $etype})--(u:User) "
                "RETURN DISTINCT u.entity_key AS user_id"
            )
        with self._session() as session:
            rows = session.run(cypher, key=entity_key, etype=et)
            return sorted({rec["user_id"] for rec in rows})

    def connected_users(self, user_id: str, depth: int = 2) -> list[str]:
        timer = Timer()
        hops = max(1, min(int(depth), 8))
        # Hop bound is a clamped integer, not an untrusted string.
        cypher = (
            "MATCH (start:User {entity_key: $uid}) "
            f"MATCH (start)-[:USES_DEVICE|USES_IP*1..{hops}]-(other:User) "
            "WHERE other.entity_key <> $uid "
            "RETURN DISTINCT other.entity_key AS user_id"
        )
        with self._session() as session:
            rows = session.run(cypher, uid=user_id)
            found = sorted({rec["user_id"] for rec in rows})
        self.last_timings_ms["connected"] = timer.ms()
        log.info("graph_connected", backend="neo4j", duration_ms=self.last_timings_ms["connected"])
        return found

    def find_connected_accounts(self, user_id: str, depth: int = 2) -> list[str]:
        return self.connected_users(user_id, depth=depth)

    def entity_degree(self, entity_type: str, entity_key: str) -> int:
        if entity_type.upper() not in ALLOWED_ENTITY_TYPES:
            return 0
        cypher = (
            "MATCH (n {entity_key: $key, entity_type: $etype}) "
            "RETURN COUNT { (n)--() } AS degree"
        )
        with self._session() as session:
            rec = session.run(cypher, key=entity_key, etype=entity_type.upper()).single()
        if rec is None:
            return 0
        return int(rec["degree"] or 0)

    def user_user_projection(self) -> nx.Graph:
        """Same semantics as NetworkX: users linked if they share a Device or IP."""
        uu = nx.Graph()
        cypher = """
        MATCH (infra)<-[r:USES_DEVICE|USES_IP]-(u:User)
        WHERE infra:Device OR infra:IP
        WITH infra, infra.entity_type AS kind, collect(DISTINCT u.entity_key) AS users
        RETURN kind, infra.entity_key AS key, users
        """
        with self._session() as session:
            for rec in session.run(cypher):
                kind = rec["kind"]
                key = rec["key"]
                users = list(rec["users"] or [])
                via = node_id(kind, key)
                for i, a in enumerate(users):
                    uu.add_node(a)
                    for b in users[i + 1 :]:
                        if uu.has_edge(a, b):
                            uu[a][b]["weight"] += 1
                            uu[a][b]["via"].add(via)
                        else:
                            uu.add_edge(a, b, weight=1, via={via})
        return uu

    def get_graph_metrics(self) -> dict[str, Any]:
        cypher_nodes = (
            "MATCH (n) WHERE n.entity_type IS NOT NULL "
            "RETURN n.entity_type AS entity_type, count(*) AS c"
        )
        cypher_rels = "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS c"
        entity_counts: dict[str, int] = {}
        rel_counts: dict[str, int] = {}
        node_count = 0
        edge_count = 0
        with self._session() as session:
            for rec in session.run(cypher_nodes):
                entity_counts[rec["entity_type"]] = int(rec["c"])
                node_count += int(rec["c"])
            for rec in session.run(cypher_rels):
                rel_counts[rec["rel"]] = int(rec["c"])
                edge_count += int(rec["c"])
        return {
            "backend": self.name,
            "node_count": node_count,
            "edge_count": edge_count,
            "entity_counts": entity_counts,
            "relationship_counts": rel_counts,
            "timings_ms": dict(self.last_timings_ms),
        }

    def snapshot(self) -> dict[str, Any]:
        nodes = []
        edges = []
        with self._session() as session:
            for rec in session.run(
                "MATCH (n) WHERE n.entity_type IS NOT NULL "
                "RETURN n.entity_type AS entity_type, n.entity_key AS entity_key, "
                "COUNT { (n)--() } AS degree"
            ):
                et, ek = rec["entity_type"], rec["entity_key"]
                nodes.append(
                    {
                        "id": node_id(et, ek),
                        "entity_type": et,
                        "entity_key": ek,
                        "degree": int(rec["degree"] or 0),
                    }
                )
            for rec in session.run(
                "MATCH (a)-[r]->(b) "
                "WHERE a.entity_type IS NOT NULL AND b.entity_type IS NOT NULL "
                "RETURN a.entity_type AS at, a.entity_key AS ak, "
                "b.entity_type AS bt, b.entity_key AS bk, "
                "type(r) AS rel, coalesce(r.count, 1) AS count"
            ):
                edges.append(
                    {
                        "source": node_id(rec["at"], rec["ak"]),
                        "target": node_id(rec["bt"], rec["bk"]),
                        "rel_types": [rec["rel"]],
                        "count": int(rec["count"] or 1),
                    }
                )
        return {
            "backend": self.name,
            "nodes": nodes,
            "edges": edges,
            "metrics": self.get_graph_metrics(),
        }

    def clear(self) -> None:
        with self._session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        self.last_timings_ms = {}

    def node_count(self) -> int:
        with self._session() as session:
            rec = session.run("MATCH (n) RETURN count(n) AS c").single()
        return int(rec["c"] if rec else 0)
