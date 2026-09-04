"""Select NetworkX or Neo4j at runtime.

GRAPH_BACKEND=networkx is the default and does not need Docker or Neo4j.
GRAPH_BACKEND=neo4j requires NEO4J_* settings. If Neo4j cannot be reached and
GRAPH_NEO4J_FALLBACK=true, NetworkX is used and health reports the failure.
"""

from __future__ import annotations

import time
from typing import Any

from app.config import get_settings
from app.graph.networkx_backend import graph_store as _networkx_store
from app.utils.logging import get_logger
from app.utils.redact import redact_secrets

log = get_logger("graph.factory")

_store = None
_status: dict[str, Any] = {
    "graph_backend": "networkx",
    "graph_connected": True,
    "graph_backend_configured": "networkx",
    "reason": None,
}


class GraphBackendError(RuntimeError):
    pass


def graph_status() -> dict[str, Any]:
    return redact_secrets(dict(_status))


def _try_neo4j(settings):
    from app.graph.neo4j_backend import Neo4jGraphStore

    uri = (settings.neo4j_uri or "").strip()
    user = (settings.neo4j_username or "").strip()
    password = settings.neo4j_password
    database = settings.neo4j_database or "neo4j"
    if not uri or not user or not password:
        raise GraphBackendError(
            "GRAPH_BACKEND=neo4j requires NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD. "
            "Credentials are not hardcoded."
        )
    timeout = max(1, int(settings.graph_connect_timeout_seconds))
    last_error = None
    deadline = time.monotonic() + timeout
    while True:
        try:
            store = Neo4jGraphStore(uri, user, password, database=database)
            if not store.ping():
                raise GraphBackendError("Neo4j ping failed")
            return store
        except Exception as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(1.5)
    raise GraphBackendError(f"Neo4j connection unavailable: {type(last_error).__name__}") from last_error


def _build():
    global _status
    settings = get_settings()
    configured = (settings.graph_backend or "networkx").strip().lower()
    if configured in {"", "networkx", "nx"}:
        _status = {
            "graph_backend": "networkx",
            "graph_connected": True,
            "graph_backend_configured": "networkx",
            "reason": None,
        }
        log.info("graph_backend_selected", graph_backend="networkx")
        return _networkx_store

    if configured != "neo4j":
        raise GraphBackendError(f"Unknown GRAPH_BACKEND '{configured}'. Use networkx or neo4j.")

    try:
        store = _try_neo4j(settings)
        _status = {
            "graph_backend": "neo4j",
            "graph_connected": True,
            "graph_backend_configured": "neo4j",
            "reason": None,
        }
        log.info("graph_backend_selected", graph_backend="neo4j")
        return store
    except Exception as exc:
        if settings.graph_neo4j_fallback:
            log.warning(
                "neo4j_unavailable_fallback_networkx",
                reason="connection unavailable",
                error=type(exc).__name__,
            )
            _status = {
                "graph_backend": "networkx",
                "graph_connected": False,
                "graph_backend_configured": "neo4j",
                "reason": "connection unavailable",
                "active_backend": "networkx",
                "fallback": True,
            }
            return _networkx_store
        _status = {
            "graph_backend": "neo4j",
            "graph_connected": False,
            "graph_backend_configured": "neo4j",
            "reason": "connection unavailable",
        }
        raise GraphBackendError(
            "GRAPH_BACKEND=neo4j but Neo4j is unavailable. "
            "Set GRAPH_NEO4J_FALLBACK=true for local NetworkX fallback, "
            "or start Neo4j (see docs/graph-backend.md)."
        ) from exc


def get_graph_store():
    global _store
    if _store is None:
        _store = _build()
    return _store


def reset_graph_store_for_tests() -> None:
    """Drop the singleton so the next get_graph_store() rebuilds it."""
    global _store
    if _store is not None and getattr(_store, "name", None) == "neo4j":
        try:
            _store.close()
        except Exception:
            pass
    _store = None


def close_graph_store() -> None:
    global _store
    if _store is not None:
        try:
            _store.close()
        except Exception:
            pass
        _store = None


class _StoreProxy:
    """Late-bound proxy so `from app.graph.factory import graph_store` stays valid."""

    def __getattr__(self, item: str):
        return getattr(get_graph_store(), item)

    def __setattr__(self, key: str, value) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            setattr(get_graph_store(), key, value)


graph_store = _StoreProxy()
