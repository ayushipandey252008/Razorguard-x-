"""Shared graph vocabulary.

NetworkX keeps the historical undirected USER–DEVICE/IP/MERCHANT/LOCATION/PAYMENT
edges used by scoring. Neo4j stores the richer labeled model (Transaction nodes
and MADE / USED_* / AT_* / USES_* relationships) while exposing the same
query methods.
"""

from __future__ import annotations

ENTITY_TYPES = ("USER", "DEVICE", "IP", "LOCATION", "MERCHANT", "TRANSACTION", "PAYMENT")

NEO4J_LABELS = {
    "USER": "User",
    "DEVICE": "Device",
    "IP": "IP",
    "LOCATION": "Location",
    "MERCHANT": "Merchant",
    "TRANSACTION": "Transaction",
    "PAYMENT": "Payment",
}

LABEL_TO_TYPE = {v: k for k, v in NEO4J_LABELS.items()}

# NetworkX / legacy edge names (scoring and existing tests).
REL_USED_DEVICE = "used_device"
REL_USED_IP = "used_ip"
REL_PAID_MERCHANT = "paid_merchant"
REL_LOCATED_AT = "located_at"
REL_USED_PAYMENT = "used_payment"

# Neo4j relationship types requested for the persistent model.
REL_MADE = "MADE"
REL_TXN_USED_DEVICE = "USED_DEVICE"
REL_TXN_USED_IP = "USED_IP"
REL_AT_LOCATION = "AT_LOCATION"
REL_AT_MERCHANT = "AT_MERCHANT"
REL_USES_DEVICE = "USES_DEVICE"
REL_USES_IP = "USES_IP"

ALLOWED_ENTITY_TYPES = frozenset(ENTITY_TYPES)
ALLOWED_REL_TYPES = frozenset(
    {
        REL_USED_DEVICE,
        REL_USED_IP,
        REL_PAID_MERCHANT,
        REL_LOCATED_AT,
        REL_USED_PAYMENT,
        REL_MADE,
        REL_TXN_USED_DEVICE,
        REL_TXN_USED_IP,
        REL_AT_LOCATION,
        REL_AT_MERCHANT,
        REL_USES_DEVICE,
        REL_USES_IP,
    }
)


def node_id(entity_type: str, entity_key: str) -> str:
    return f"{entity_type}:{entity_key}"
