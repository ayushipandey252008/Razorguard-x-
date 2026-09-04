"""Behavioral end-to-end tests. Assert decisions and ranges, not exact scores."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.agents.tools import TOOL_SPECS
from app.config import get_settings
from app.main import app


def _login(client: TestClient):
    r = client.post("/api/v1/auth/login", json={"email": "admin@razorguard.local", "password": "prototype-pass"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_find_fraud_cluster_tool_requires_transaction_id():
    spec = next(t for t in TOOL_SPECS if t["function"]["name"] == "find_fraud_cluster")
    assert "transaction_id" in spec["function"]["parameters"]["required"]


def test_normal_transaction_approves():
    settings = get_settings()
    with TestClient(app) as client:
        headers = _login(client)
        r = client.post(
            "/api/v1/transactions",
            json={
                "user_id": "usr_ananya",
                "merchant_id": "m_groc_01",
                "amount": 620,
                "currency": "INR",
                "device_id": "dev_ananya_phone",
                "ip_address": "10.10.1.14",
                "location": "Bengaluru",
                "payment_method": "UPI",
                "failed_attempts": 0,
                "transaction_velocity": 1,
                "current_device_known": True,
                "current_location_known": True,
                "scenario_tag": "normal",
            },
            headers=headers,
        )
        assert r.status_code == 200, r.text
        risk = r.json()["risk"]
        assert 0 <= risk["final_risk_score"] <= 100
        assert risk["decision"] == "APPROVE"
        assert risk["final_risk_score"] < settings.threshold_review
        assert r.json()["investigation_id"] is None
        assert "probability_calibrated" in risk
        if risk["probability_calibrated"]:
            assert 0.0 <= risk["ml_probability"] <= 1.0
            assert risk["ml_probability_raw"] is None or 0.0 <= risk["ml_probability_raw"] <= 1.0


def test_moderate_transaction_reviews():
    settings = get_settings()
    with TestClient(app) as client:
        headers = _login(client)
        r = client.post(
            "/api/v1/transactions",
            json={
                "user_id": "usr_leila",
                "merchant_id": "m_fash_01",
                "amount": 3200,
                "currency": "INR",
                "device_id": "dev_leila_travel",
                "ip_address": "198.51.100.40",
                "location": "Singapore",
                "payment_method": "CARD_TOKEN",
                "failed_attempts": 3,
                "transaction_velocity": 5,
                "current_device_known": False,
                "current_location_known": False,
                "scenario_tag": "moderate_review",
            },
            headers=headers,
        )
        assert r.status_code == 200, r.text
        risk = r.json()["risk"]
        assert risk["decision"] == "REVIEW", (
            f"expected REVIEW, got {risk['decision']} score={risk['final_risk_score']} "
            f"components ml={risk['ml_score']} beh={risk['behavior_score']} "
            f"rules={risk['rule_score']} graph={risk['graph_score']}"
        )
        assert settings.threshold_review <= risk["final_risk_score"] < settings.threshold_block
        assert r.json()["investigation_id"]


def test_high_risk_transaction_blocks_and_investigates():
    settings = get_settings()
    with TestClient(app) as client:
        headers = _login(client)
        r = client.post(
            "/api/v1/transactions",
            json={
                "user_id": "usr_ring_a",
                "merchant_id": "m_watch_01",
                "amount": 22000,
                "currency": "INR",
                "device_id": "dev_farm_01",
                "ip_address": "203.0.113.200",
                "location": "Dubai",
                "payment_method": "UPI",
                "failed_attempts": 5,
                "transaction_velocity": 11,
                "current_device_known": True,
                "current_location_known": False,
                "scenario_tag": "fraud_ring",
            },
            headers=headers,
        )
        assert r.status_code == 200, r.text
        data = r.json()
        risk = data["risk"]
        assert risk["decision"] == "BLOCK"
        assert risk["final_risk_score"] >= settings.threshold_block
        assert risk["triggered_rules"]
        assert data["investigation_id"]
        ge = risk["graph_evidence"]
        assert ge.get("score_basis") is not None
        assert ge.get("device_user_count", 0) >= 3 or len(ge.get("device_users") or []) >= 3

        run = client.post(f"/api/v1/investigations/{data['investigation_id']}/run", headers=headers)
        assert run.status_code == 200, run.text
        report = run.json()["report"]
        tools = {e["source"] for e in report.get("evidence") or [] if isinstance(e, dict)}
        assert "find_connected_accounts" in tools
        assert "find_fraud_cluster" in tools
        cluster = report.get("potential_fraud_ring") or {}
        assert cluster.get("identified") is True
        assert cluster.get("cluster_id")
        assert len(cluster.get("connected_users") or []) >= 3
        trace_names = [t["tool"] for t in report.get("tool_trace") or []]
        assert "find_connected_accounts" in trace_names
        assert "find_fraud_cluster" in trace_names
        cluster_call = next(t for t in report["tool_trace"] if t["tool"] == "find_fraud_cluster")
        assert cluster_call["arguments"].get("transaction_id") == data["transaction"]["transaction_id"]
        assert cluster_call["result"].get("unavailable") is not True
        assert cluster_call["result"].get("identified") is True


def test_unique_device_does_not_invent_a_cluster():
    with TestClient(app) as client:
        headers = _login(client)
        r = client.post(
            "/api/v1/transactions",
            json={
                "user_id": "usr_ananya",
                "merchant_id": "m_elec_01",
                "amount": 48000,
                "currency": "INR",
                "device_id": "dev_unique_stolen_e2e",
                "ip_address": "203.0.113.77",
                "location": "Dubai",
                "payment_method": "UPI",
                "failed_attempts": 4,
                "transaction_velocity": 9,
                "current_device_known": False,
                "current_location_known": False,
                "scenario_tag": "stolen_account",
            },
            headers=headers,
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["risk"]["decision"] in {"REVIEW", "BLOCK"}
        assert data["investigation_id"]
        run = client.post(f"/api/v1/investigations/{data['investigation_id']}/run", headers=headers)
        assert run.status_code == 200, run.text
        report = run.json()["report"]
        cluster = report.get("potential_fraud_ring") or {}
        assert cluster.get("identified") is False
        assert "no suspicious cluster identified" in (cluster.get("message") or "").lower()


def test_fraud_ring_simulation_can_identify_cluster():
    with TestClient(app) as client:
        headers = _login(client)
        sim = client.post(
            "/api/v1/simulation/run",
            json={"scenario": "fraud_ring", "count": 8},
            headers=headers,
        )
        assert sim.status_code == 200, sim.text
        payload = sim.json()
        assert payload["detected_clusters"]
        flagged = [t for t in payload["transactions"] if t["investigation_id"]]
        assert flagged, "fraud_ring simulation should produce at least one REVIEW/BLOCK"
        inv_id = flagged[-1]["investigation_id"]
        run = client.post(f"/api/v1/investigations/{inv_id}/run", headers=headers)
        assert run.status_code == 200, run.text
        report = run.json()["report"]
        cluster = report.get("potential_fraud_ring") or {}
        assert cluster.get("identified") is True
        assert cluster.get("cluster_id")
        assert len(cluster.get("connected_users") or []) >= 3
        assert cluster.get("shared_devices") or cluster.get("shared_ips")
        assert cluster.get("relationship_counts")
        assert cluster.get("graph_risk") is not None or cluster.get("graph_risk_score") is not None


def test_pipeline_exposes_graph_evidence_and_calibration_fields():
    with TestClient(app) as client:
        headers = _login(client)
        r = client.post(
            "/api/v1/transactions",
            json={
                "user_id": "usr_kabir",
                "merchant_id": "m_elec_01",
                "amount": 1800,
                "device_id": "dev_kabir_laptop",
                "ip_address": "10.20.30.40",
                "location": "Mumbai",
                "scenario_tag": "normal",
            },
            headers=headers,
        )
        assert r.status_code == 200, r.text
        risk = r.json()["risk"]
        ge = risk["graph_evidence"]
        assert "graph_score" in ge
        assert "score_basis" in ge
        assert "device_users" in ge
        assert "X-Request-ID" in r.headers
