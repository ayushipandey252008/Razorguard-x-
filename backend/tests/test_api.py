from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.ml.predictor import model_service


def _login(client: TestClient, email="admin@razorguard.local", password="prototype-pass"):
    r = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health():
    with TestClient(app) as client:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_login_and_me():
    with TestClient(app) as client:
        headers = _login(client)
        me = client.get("/api/v1/auth/me", headers=headers)
        assert me.status_code == 200
        assert me.json()["role"] == "ADMIN"


def test_end_to_end_pipeline():
    with TestClient(app) as client:
        headers = _login(client)
        body = {
            "user_id": "usr_ananya",
            "merchant_id": "m_elec_01",
            "amount": 48000,
            "currency": "INR",
            "device_id": "dev_unknown_e2e",
            "ip_address": "203.0.113.77",
            "location": "Dubai",
            "payment_method": "UPI",
            "failed_attempts": 4,
            "transaction_velocity": 9,
            "current_device_known": False,
            "current_location_known": False,
            "scenario_tag": "stolen_account",
        }
        r = client.post("/api/v1/transactions", json=body, headers=headers)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["transaction"]["transaction_id"]
        risk = data["risk"]
        assert 0 <= risk["final_risk_score"] <= 100
        assert risk["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
        assert risk["model_version"]
        for key in ("ml_score", "behavior_score", "rule_score", "graph_score"):
            assert key in risk
        assert risk["explanation"]
        tid = data["transaction"]["transaction_id"]
        got = client.get(f"/api/v1/transactions/{tid}", headers=headers)
        assert got.status_code == 200
        detail = got.json()
        assert "typical_amount" in (detail.get("user_baseline") or {})
        assert not isinstance(detail["user_baseline"], list)
        analytics = client.get("/api/v1/analytics", headers=headers)
        assert analytics.status_code == 200
        assert analytics.json()["totals"]["transactions"] >= 1
        offline = client.get("/api/v1/ml/offline-evaluation", headers=headers)
        assert offline.status_code == 200
        body = offline.json()
        assert body["label"] == "OFFLINE EVALUATION"
        assert body["ulb"]["track"] == "REAL_DATASET"
        assert "ulb" in body
        assert body["calibration"]["label"] == "PROTOTYPE CALIBRATION"
        assert body["calibration"]["not_industry_standard"] is True
        ids = {row["id"] for row in body["model_registry"]}
        assert "xgb-iforest-v1-calibrated" in ids
        assert "ulb-xgb-v1" in ids
        assert "ulb-xgb-v1-calibrated" in ids
        ieee = client.get("/api/v1/ml/ieee-evaluation", headers=headers)
        assert ieee.status_code == 200, ieee.text
        ieee_body = ieee.json()
        assert ieee_body["label"] == "OFFLINE PUBLIC DATASET EVALUATION"
        assert ieee_body["ieee_cis"]["status"] == "OFFLINE CANDIDATE"
        assert ieee_body["active_model"]["version"] == "xgb-iforest-v1-calibrated"
        assert body["ieee"]["label"] == "OFFLINE PUBLIC DATASET EVALUATION"
        assert "ieee-xgb-baseline-v1" in ids


def test_simulation_and_investigation_tools():
    with TestClient(app) as client:
        headers = _login(client)
        sim = client.post(
            "/api/v1/simulation/run",
            json={"scenario": "fraud_ring", "count": 6},
            headers=headers,
        )
        assert sim.status_code == 200, sim.text
        payload = sim.json()
        assert payload["count"] == 6
        assert payload["transactions"]
        flagged = [t for t in payload["transactions"] if t["investigation_id"]]
        if flagged:
            inv_id = flagged[0]["investigation_id"]
            run = client.post(f"/api/v1/investigations/{inv_id}/run", headers=headers)
            assert run.status_code == 200, run.text
            report = run.json()["report"]
            assert "evidence" in report
            assert "recommended_action" in report
            assert report.get("limitations")


def test_viewer_cannot_simulate():
    with TestClient(app) as client:
        headers = _login(client, "viewer@razorguard.local")
        r = client.post("/api/v1/simulation/run", json={"scenario": "normal", "count": 2}, headers=headers)
        assert r.status_code == 403


def test_model_service_predicts():
    model_service.load_or_train()
    out = model_service.predict(
        {
            "amount": 120.0,
            "account_age_days": 400,
            "failed_attempts": 0,
            "transaction_velocity": 1,
            "previous_transaction_count": 20,
            "previous_average_amount": 110.0,
            "current_device_known": True,
            "current_location_known": True,
            "timestamp": datetime.now(timezone.utc),
            "payment_method": "UPI",
            "merchant_category": "GROCERY",
        }
    )
    assert 0.0 <= out["ml_probability"] <= 1.0
    assert out["model_version"]
