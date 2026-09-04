"""GraphBackend protocol.

NetworkX is the default in-process store. Neo4j is an optional persistent
implementation of the same interface. Callers must not depend on a vendor.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphBackend(Protocol):
    name: str

    def upsert_entity(self, entity_type: str, entity_key: str, properties: dict | None = None) -> str: ...

    def add_relationship(
        self,
        from_type: str,
        from_key: str,
        to_type: str,
        to_key: str,
        rel_type: str,
        properties: dict | None = None,
    ) -> None: ...

    def upsert_relationship(
        self,
        from_type: str,
        from_key: str,
        to_type: str,
        to_key: str,
        rel_type: str,
        properties: dict | None = None,
    ) -> None: ...

    def ingest_payment(self, txn: dict) -> None: ...

    def get_entity(self, entity_type: str, entity_key: str) -> dict[str, Any] | None: ...

    def neighbors(self, entity_type: str, entity_key: str) -> list[dict[str, Any]]: ...

    def get_neighbors(self, entity_type: str, entity_key: str) -> list[dict[str, Any]]: ...

    def users_sharing_entity(self, entity_type: str, entity_key: str) -> list[str]: ...

    def connected_users(self, user_id: str, depth: int = 2) -> list[str]: ...

    def find_connected_accounts(self, user_id: str, depth: int = 2) -> list[str]: ...

    def entity_degree(self, entity_type: str, entity_key: str) -> int: ...

    def user_user_projection(self) -> Any: ...

    def get_graph_metrics(self) -> dict[str, Any]: ...

    def snapshot(self) -> dict[str, Any]: ...

    def clear(self) -> None: ...

    def ping(self) -> bool: ...

    def close(self) -> None: ...
