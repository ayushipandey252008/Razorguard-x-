"""Graph store abstraction.

NetworkX is the first implementation. A Neo4j backend can implement the same
`GraphBackend` protocol without changing callers.
"""

from __future__ import annotations

from app.graph.backend import GraphBackend
from app.graph.factory import get_graph_store, graph_status, graph_store

__all__ = ["GraphBackend", "get_graph_store", "graph_status", "graph_store"]
