# Architecture

RazorGuard X is a modular monolith: one FastAPI process owns scoring, one Next.js app owns the operator UI.

## Request path

```
TransactionCreate
  → enrichment (history, known device/location)
  → persist Transaction
  → XGBoost probability + SHAP
  → Isolation Forest + behavioral flags
  → GraphBackend ingest + graph score (NetworkX default, optional Neo4j)
  → rule catalog
  → weighted combiner → APPROVE | REVIEW | BLOCK
  → investigation row if not APPROVE
  → outbox events in the same DB transaction
  → commit
  → outbox worker → EventBus (in-process or Kafka)
  → WebSocket live-wire event
```

Simulation and the public transaction API share `services/pipeline.py`. The UI never computes risk.

## Packages

| Package | Responsibility |
| --- | --- |
| `app/ml` | Features, train, predict, SHAP |
| `app/ml/ieee` | Isolated IEEE-CIS offline adapter (candidates only) |
| `app/anomaly` | Isolation Forest mapping + interpretable deviations |
| `app/rules` | Registerable rules, evidence payloads |
| `app/graph` | `GraphBackend` protocol, NetworkX default, optional Neo4j, ring detector |
| `app/services/risk_engine` | Weights, thresholds, narrative |
| `app/agents` | LLM provider + tool box + fallback investigator |
| `app/security` | JWT, password hashing, RBAC |
| `app/events` | Typed `EventBus`, transactional outbox, optional Kafka |
| `app/api/v1` | HTTP / WebSocket only |

## Persistence

PostgreSQL (Compose) or SQLite (local). Tables cover app users, payment users, merchants, transactions, risk assessments, triggered rules, investigations, analyst decisions, analyst feedback, graph entities/relationships, clusters, audit logs, model versions, processed events, alerts, failed events, drift alerts, and the transactional outbox.

Graph **query** path is GraphBackend (NetworkX rebuilt from SQL on boot, or Neo4j when configured). SQL `graph_entities` remain an auxiliary write path.

## Realtime

`EventBus` in `app/services/events.py` publishes WebSocket live-wire messages to Redis when `REDIS_URL` is reachable, and always fans out to in-process queues.

Typed domain events (`transaction-created`, `risk-scored`, …) use `app.events.EventBus`. Default `EVENT_BUS=inprocess`. `EVENT_BUS=kafka` is optional and falls back when the broker is down (`docs/event-driven-architecture.md`).

## Observability (implemented)

Structured JSON logs with `request_id` and `correlation_id` (`X-Request-ID` / `X-Correlation-ID`). Pipeline stages log latency: enrich, ML, behavior, graph, rules, combine. Agent investigations log tool-loop latency. Domain event publish/consume samples appear on `GET /api/v1/events/status` as prototype measurements.

## Extension points (not implemented)

- **GNNs** — consume the same entity/edge snapshot
- **Per-user Isolation Forest** — forest is global; personalized checks are explicit overlays in `anomaly/behavior.py`
- **Feature store / online learning** — not implemented. Prototype PSI drift + offline feedback candidates: `docs/feedback-and-model-monitoring.md`
- **Certified thresholds** — `THRESHOLD_*` env vars plus a synthetic cost experiment, not an industry standard

## Frontend

App Router pages talk to `/api/v1` with a bearer token. No business logic in the browser beyond presentation and human-in-the-loop forms.
