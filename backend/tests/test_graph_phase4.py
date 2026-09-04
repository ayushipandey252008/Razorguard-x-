"""Phase 4 GraphBackend: NetworkX default, optional Neo4j, security."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.graph.factory import graph_status, graph_store
from app.graph.networkx_backend import NetworkXGraphStore
from app.graph.rings import cluster_for_transaction
from app.graph.service import ingest_transaction
from app.main import app


def _login(client: TestClient):
    r = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@razorguard.local", "password": "prototype-pass"},
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _txn(i: int, user: str, device: str, ip: str, **extra):
    row = {
        "transaction_id": extra.get("transaction_id", f"p4t{i}"),
        "user_id": user,
        "device_id": device,
        "ip_address": ip,
        "merchant_id": extra.get("merchant_id", "m1"),
        "location": extra.get("location", "Dubai"),
        "payment_identifier": extra.get("payment_identifier", f"payp4{i}"),
        "account_age_days": extra.get("account_age_days", 3),
        "merchant_category": extra.get("merchant_category", "DIGITAL_GOODS"),
        "amount": extra.get("amount", 100),
        "currency": "INR",
    }
    return row


def test_networkx_ingest_payment_and_metrics():
    store = NetworkXGraphStore()
    store.ingest_payment(_txn(0, "u_a", "dev_x", "1.1.1.1"))
    assert store.get_entity("USER", "u_a")
    assert store.get_neighbors("USER", "u_a")
    assert "u_a" in store.users_sharing_entity("DEVICE", "dev_x")
    metrics = store.get_graph_metrics()
    assert metrics["backend"] == "networkx"
    assert metrics["entity_counts"]["USER"] >= 1
    assert "upsert" in metrics["timings_ms"]
    store.clear()
    assert store.get_graph_metrics()["node_count"] == 0


def test_health_exposes_graph_backend_without_credentials():
    with TestClient(app) as client:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["graph_backend"] in {"networkx", "neo4j"}
        assert "graph_connected" in body
        blob = str(body).lower()
        assert "neo4j_password" not in blob
        assert "authorization" not in body
        assert body.get("neo4j_uri") is None
        assert body.get("neo4j_password") is None


def test_graph_metrics_and_no_cypher_endpoint():
    with TestClient(app) as client:
        headers = _login(client)
        metrics = client.get("/api/v1/graph/metrics", headers=headers)
        assert metrics.status_code == 200
        data = metrics.json()
        assert "entity_counts" in data
        assert "relationship_counts" in data
        assert data["graph_backend"] in {"networkx", "neo4j"}
        cypher = client.post(
            "/api/v1/graph/cypher",
            headers=headers,
            json={"query": "MATCH (n) RETURN n"},
        )
        assert cypher.status_code in {404, 405, 422}
        snap = client.get("/api/v1/graph/snapshot", headers=headers)
        assert snap.status_code == 200
        assert "nodes" in snap.json()
        assert "password" not in str(snap.json()).lower() or "[redacted]" in str(snap.json()).lower()


def test_entity_strings_are_data_not_cypher():
    evil = "MATCH (n) DETACH DELETE n"
    ingest_transaction(_txn(91, "u_p4_safe", "dev_p4_safe", "10.0.0.9", location=evil))
    ingest_transaction(
        _txn(
            92,
            evil,
            "dev_p4_inject",
            "10.0.0.10",
            transaction_id="p4inject",
            payment_identifier="payinject",
        )
    )
    ingest_transaction(_txn(93, "u_p4_other", "dev_p4_inject", "10.0.0.10", transaction_id="p4other", payment_identifier="payother"))
    connected = graph_store.connected_users("u_p4_other", depth=2)
    assert evil in connected or graph_store.get_entity("USER", evil)
    assert graph_store.get_entity("USER", "u_p4_safe") is not None


def test_shared_device_cluster_schema_includes_backend():
    for i, user in enumerate(["u_p4c_a", "u_p4c_b", "u_p4c_c"]):
        ingest_transaction(_txn(200 + i, user, "dev_p4_cluster", "203.0.113.50", payment_identifier=f"p4c{i}"))
    cluster = cluster_for_transaction(
        {
            "transaction_id": "p4t200",
            "user_id": "u_p4c_a",
            "device_id": "dev_p4_cluster",
            "ip_address": "203.0.113.50",
        }
    )
    assert cluster["cluster_found"] is True
    assert cluster["graph_backend"] in {"networkx", "neo4j"}
    assert cluster["cluster_size"] >= 3


def _neo4j_configured() -> bool:
    return os.environ.get("RUN_NEO4J_TESTS") == "1"


@pytest.mark.skipif(not _neo4j_configured(), reason="Set RUN_NEO4J_TESTS=1 with a reachable Neo4j to run integration tests")
def test_neo4j_consistency_with_networkx():
    from app.graph.neo4j_backend import Neo4jGraphStore

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    if not password:
        pytest.skip("NEO4J_PASSWORD is required for Neo4j integration tests")
    nx_store = NetworkXGraphStore()
    neo = Neo4jGraphStore(uri, user, password, database=database)
    neo.clear()
    rows = [
        _txn(1, "nu1", "ndev", "n1.1.1.1", payment_identifier="np1"),
        _txn(2, "nu2", "ndev", "n1.1.1.1", payment_identifier="np2"),
        _txn(3, "nu3", "ndev", "n1.1.1.1", payment_identifier="np3"),
        _txn(4, "nlonely", "ndev_lonely", "n9.9.9.9", payment_identifier="npl"),
    ]
    for row in rows:
        nx_store.ingest_payment(row)
        neo.ingest_payment(row)

    assert set(nx_store.users_sharing_entity("DEVICE", "ndev")) == set(neo.users_sharing_entity("DEVICE", "ndev"))
    assert set(nx_store.connected_users("nu1")) == set(neo.connected_users("nu1"))
    assert set(nx_store.connected_users("nlonely")) == set(neo.connected_users("nlonely"))

    nx_proj = nx_store.user_user_projection()
    neo_proj = neo.user_user_projection()
    assert set(nx_proj.nodes()) == set(neo_proj.nodes())

    # Cluster via each store's projection by briefly using the store methods through rings
    # would hit the process singleton. Compare sizes from projections instead.
    import networkx as nx

    def clusters(uu):
        return [sorted(c) for c in nx.connected_components(uu) if len(c) >= 3]

    assert clusters(nx_proj) == clusters(neo_proj)
    neo.clear()
    neo.close()


def test_neo4j_cypher_is_parameterized():
    source = Path(__file__).resolve().parents[1] / "app" / "graph" / "neo4j_backend.py"
    text = source.read_text()
    assert "session.run(cypher, key=" in text or "session.run(\n                cypher" in text
    assert "$user_id" in text
    assert "f\"MATCH (n:{user_id}" not in text
    assert "execute_cypher" not in text


def test_graph_status_helper_has_no_secrets():
    status = graph_status()
    assert "password" not in status
    assert "neo4j_password" not in status
    assert status["graph_backend"] in {"networkx", "neo4j"}
