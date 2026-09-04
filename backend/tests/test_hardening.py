from app.config import assert_secure_settings, database_connect_args, get_settings, normalize_database_url
from app.graph.rings import cluster_for_transaction
from app.graph.service import ingest_transaction, score_entity
from app.ml.predictor import model_service
from fastapi.testclient import TestClient

from app.main import app


def test_cluster_for_transaction_identified_and_absent():
    for i, user in enumerate(["u_cl_a", "u_cl_b", "u_cl_c"]):
        ingest_transaction(
            {
                "transaction_id": f"tcl{i}",
                "user_id": user,
                "device_id": "shared_cluster_dev",
                "ip_address": "203.0.113.19",
                "merchant_id": "m1",
                "location": "Dubai",
                "payment_identifier": f"payc{i}",
                "account_age_days": 3,
                "merchant_category": "DIGITAL_GOODS",
            }
        )
    found = cluster_for_transaction(
        {
            "transaction_id": "tcl0",
            "user_id": "u_cl_a",
            "device_id": "shared_cluster_dev",
            "ip_address": "203.0.113.19",
        }
    )
    assert found["identified"] is True
    assert found["cluster_id"].startswith("cl_")
    assert len(found["connected_users"]) >= 3
    assert "shared_cluster_dev" in found["shared_devices"]
    assert found["relationship_counts"]["shared_device_count"] >= 1
    assert found["graph_risk"] > 0

    lonely = ingest_transaction(
        {
            "transaction_id": "tlonely",
            "user_id": "u_lonely",
            "device_id": "dev_lonely_only",
            "ip_address": "198.51.100.9",
            "merchant_id": "m1",
            "location": "Pune",
            "payment_identifier": "paylonely",
            "account_age_days": 40,
            "merchant_category": "GROCERY",
        }
    )
    absent = cluster_for_transaction(
        {
            "transaction_id": "tlonely",
            "user_id": "u_lonely",
            "device_id": "dev_lonely_only",
            "ip_address": "198.51.100.9",
        }
    )
    assert absent["identified"] is False
    assert absent["message"] == "no suspicious cluster identified"
    assert lonely["cluster_id"] is None


def test_merchant_only_neighbors_do_not_count_as_connected_users():
    ingest_transaction(
        {
            "transaction_id": "tm1",
            "user_id": "u_merch_a",
            "device_id": "dev_ma",
            "ip_address": "10.1.1.1",
            "merchant_id": "m_shared_only",
            "location": "Delhi",
            "payment_identifier": "pma",
            "account_age_days": 20,
            "merchant_category": "GROCERY",
        }
    )
    ingest_transaction(
        {
            "transaction_id": "tm2",
            "user_id": "u_merch_b",
            "device_id": "dev_mb",
            "ip_address": "10.1.1.2",
            "merchant_id": "m_shared_only",
            "location": "Mumbai",
            "payment_identifier": "pmb",
            "account_age_days": 20,
            "merchant_category": "GROCERY",
        }
    )
    scored = score_entity("USER", "u_merch_a")
    assert "u_merch_b" not in scored["connected_users"]


def test_normalize_database_url_for_render_postgres():
    assert normalize_database_url("sqlite+aiosqlite:///./razorguard.db").startswith("sqlite")
    assert normalize_database_url("postgres://u:p@db:5432/razorguard") == "postgresql+asyncpg://u:p@db:5432/razorguard"
    assert normalize_database_url("postgresql://u:p@db:5432/razorguard") == "postgresql+asyncpg://u:p@db:5432/razorguard"
    assert normalize_database_url("postgresql+asyncpg://u:p@db:5432/razorguard") == "postgresql+asyncpg://u:p@db:5432/razorguard"
    assert database_connect_args("sqlite+aiosqlite:///./razorguard.db") == {"check_same_thread": False}
    assert database_connect_args("postgresql+asyncpg://u:p@localhost:5432/razorguard") == {}
    assert database_connect_args("postgresql+asyncpg://u:p@dpg-example.render.com/razorguard") == {"ssl": True}


def test_production_secret_refused():
    s = get_settings()
    s.environment = "production"
    s.secret_key = "dev-secret-change-me"
    try:
        raised = False
        try:
            assert_secure_settings(s)
        except RuntimeError:
            raised = True
        assert raised
    finally:
        s.environment = "development"
        s.secret_key = "dev-secret-change-me"


def test_health_and_request_id_header():
    with TestClient(app) as client:
        r = client.get("/api/v1/health", headers={"X-Request-ID": "test-rid-1"})
        assert r.status_code == 200
        assert r.headers.get("X-Request-ID") == "test-rid-1"
        assert r.json()["prototype"] is True


def test_model_exposes_raw_and_calibrated_probability():
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
            "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            "payment_method": "UPI",
            "merchant_category": "GROCERY",
        }
    )
    assert "ml_probability_raw" in out
    assert "probability_calibrated" in out
    assert 0.0 <= out["ml_score"] <= 100.0
    assert (model_service.metrics or {}).get("track") in {None, "SYNTHETIC_DATASET"}
