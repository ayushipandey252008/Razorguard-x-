from app.graph.networkx_backend import NetworkXGraphStore
from app.graph.rings import detect_potential_rings
from app.graph.service import ingest_transaction


def test_shared_device_connects_users():
    for i, user in enumerate(["u1", "u2", "u3"]):
        ingest_transaction(
            {
                "transaction_id": f"t{i}",
                "user_id": user,
                "device_id": "shared_dev",
                "ip_address": "203.0.113.9",
                "merchant_id": "m1",
                "location": "Dubai",
                "payment_identifier": f"pay{i}",
                "account_age_days": 3,
                "merchant_category": "DIGITAL_GOODS",
            }
        )
    rings = detect_potential_rings(min_users=3)
    assert rings
    devices = {d for c in rings for d in c["shared_devices"]}
    assert "shared_dev" in devices
    assert any("potential fraud ring" in c["explanation"].lower() for c in rings)


def test_graph_backend_neighbors():
    store = NetworkXGraphStore()
    store.add_relationship("USER", "a", "DEVICE", "d", "used_device")
    nbs = store.neighbors("USER", "a")
    assert nbs[0]["entity_key"] == "d"
