# Graph backends

This is an independent student prototype and is not an official Razorpay product.

The live graph is accessed only through `GraphBackend`. NetworkX is the default local/dev store. Neo4j is an optional persistent implementation of the same interface. Scoring heuristics do not change when the store changes.

## GraphBackend abstraction

```
backend/app/graph/
  backend.py          protocol
  models.py           entity/relationship names
  factory.py          GRAPH_BACKEND selection
  networkx_backend.py default in-memory store
  neo4j_backend.py    optional persistent store
  rings.py            fraud-cluster heuristics (store-agnostic)
  service.py          ingest + graph score (formula unchanged)
```

Callers (`ingest_transaction`, investigation tools, `/api/v1/graph/*`) use `app.graph.factory.graph_store`. They must not import Neo4j drivers or run Cypher.

Supported operations: `upsert_entity`, `upsert_relationship` / `add_relationship`, `ingest_payment`, `get_entity`, `get_neighbors`, `users_sharing_entity`, `connected_users` / `find_connected_accounts`, `entity_degree`, `user_user_projection`, `get_graph_metrics`, `snapshot`, `clear`, `ping`.

Fraud-cluster detection stays in `rings.py`. Both backends build the same user–user projection (users linked by a shared device or IP). The agent does not know which store produced the evidence.

## NetworkX implementation

Default: `GRAPH_BACKEND=networkx`.

Undirected USER–DEVICE / IP / MERCHANT / LOCATION / PAYMENT edges (`used_device`, `used_ip`, …). No Neo4j process is required. On API start the in-memory graph is cleared and rebuilt from SQL transactions.

Existing fraud-ring heuristics and tests target this store.

## Neo4j implementation

`GRAPH_BACKEND=neo4j` plus:

- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE` (default `neo4j`)

Credentials are never hardcoded. Cypher is parameterized. Labels and relationship types are allow-listed. Entity strings are DATA.

If Neo4j cannot be reached:

- `GRAPH_NEO4J_FALLBACK=true` (development default): use NetworkX and report `graph_connected: false`, `reason: "connection unavailable"`.
- `GRAPH_NEO4J_FALLBACK=false`: fail clearly at startup. The API does not pretend Neo4j is persisting.

Health (no secrets):

```json
{ "graph_backend": "neo4j", "graph_connected": true }
```

or

```json
{ "graph_backend": "neo4j", "graph_connected": false, "reason": "connection unavailable" }
```

When fallback is active, `graph_backend` is `networkx` and `graph_backend_configured` remains `neo4j`.

## Data model (Neo4j)

Nodes (stable `entity_key`, `entity_type`): User, Device, IP, Location, Merchant, Transaction.

Relationships:

- User `-[:MADE]->` Transaction
- Transaction `-[:USED_DEVICE]->` Device
- Transaction `-[:USED_IP]->` IP
- Transaction `-[:AT_LOCATION]->` Location
- Transaction `-[:AT_MERCHANT]->` Merchant
- User `-[:USES_DEVICE]->` Device
- User `-[:USES_IP]->` IP
- User `-[:PAID_MERCHANT]->` Merchant (query parity with NetworkX)
- User `-[:LOCATED_AT]->` Location (query parity)

MERGE/upsert; no duplicate logical entities. Payment tokens / PANs are not stored on Neo4j nodes.

## Fraud-ring detection

Unchanged prototype heuristics (not production-grade): shared device/IP, ≥3 users, density, short flagged paths. `find_fraud_cluster` returns the same schema from either backend.

## Configuration

See `.env.example`. Local default is NetworkX.

## Docker

`docker compose up` starts Neo4j (`7474` browser, `7687` bolt) with a volume `neo4j_data`. Default compose `GRAPH_BACKEND` is still `networkx` so existing flows keep working.

To use Neo4j from the backend container:

```bash
GRAPH_BACKEND=neo4j GRAPH_NEO4J_FALLBACK=false docker compose up --build
```

The process retries Bolt until `GRAPH_CONNECT_TIMEOUT_SECONDS` (default 30).

## Persistence

Neo4j keeps graph state across backend and container restarts while the volume remains. NetworkX does not.

On boot:

- NetworkX: always rebuild from SQL.
- Neo4j: skip full rebuild when nodes already exist, unless `GRAPH_REBUILD_ON_START=true`. New payments still MERGE through the pipeline.

Development reset (ADMIN, not production): `POST /api/v1/graph/reset` clears the active backend. SQL rows are not deleted. Replay ingest or restart (NetworkX) to refill.

## Fallback behavior

NetworkX mode never needs Neo4j. Neo4j mode either connects, falls back if configured, or fails loudly.

## Investigation agent

`find_connected_accounts` and `find_fraud_cluster` call GraphBackend. Reports may include `graph_backend`. Raw Cypher is not exposed.

## Limitations

Prototype only. Neo4j community in Compose is not a production cluster. Heuristics are not real-world fraud accuracy. Optional Kafka is documented in `docs/event-driven-architecture.md`. GNNs, Kubernetes, and a feature store are not implemented.

## Integration tests

```bash
export RUN_NEO4J_TESTS=1
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=...
cd backend && PYTHONPATH=. python -m pytest tests/test_graph_phase4.py -q
```

These tests wipe the Neo4j database they connect to. Do not point them at a graph you care about. The default unit suite does not start Neo4j.
