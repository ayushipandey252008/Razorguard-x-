# Event-driven architecture (Phase 5)

This is an independent student prototype and is not an official Razorpay product.

Kafka is an **optional event transport**. The payment API still scores synchronously and returns APPROVE / REVIEW / BLOCK in the HTTP response. If Kafka is down, the application continues using an in-process bus.

## Architecture

```
Client  ──POST /transactions──►  FastAPI pipeline (unchanged scoring)
                                      │
                                      │  1. persist + enrich + ML + behavior + rules + graph
                                      │  2. HTTP response = risk decision (sync)
                                      │
                                      ▼
                          insert outbox (same DB txn)
                                      │
                                   commit
                                      ▼
                               outbox worker
                                      ▼
                               EventBus.publish
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
              EVENT_BUS=inprocess                 EVENT_BUS=kafka
              (default)                           optional broker
                    │                                   │
                    ▼                                   ▼
            in-memory handlers                  Kafka topics + consumers
            (local / pytest)                    + in-process fallback
```

Callers depend on `EventBus`, not `KafkaProducer` / `KafkaConsumer`.

```
backend/app/events/
  base.py            correlation ID + payload sanitization
  schemas.py         Pydantic envelopes
  bus.py             EventBus protocol
  inprocess_bus.py   default local bus
  kafka_bus.py       optional Kafka transport
  factory.py         EVENT_BUS selection + fallback status
  consumers.py       risk / investigation / alert handlers
```

The existing Redis/in-process bus in `app/services/events.py` still fans out **WebSocket** live-wire messages. Domain events are a separate, typed stream.

## Event schemas

Every event has: `event_id`, `event_type`, `timestamp`, `schema_version`, `correlation_id`, `transaction_id` (where applicable), `payload`.

| Type | When | Payload (no secrets / no PAN) |
| --- | --- | --- |
| `transaction-created` | After the transaction row is committed | user/merchant ids, amount, currency, method, category, scenario |
| `risk-scored` | After the existing pipeline decision | decision, component scores, model version, investigation_id |
| `investigation-created` | REVIEW/BLOCK case opened | investigation_id, status, severity |
| `investigation-completed` | Agent run finished | recommendation, risk_level, provider, graph_backend, evidence_count |
| `alert-created` | BLOCK, or REVIEW if `EVENT_ALERT_ON_REVIEW=true` | alert_id, decision, risk_level, kind |
| `analyst-feedback-recorded` | Human decision | investigation_id, decision, status |

Payloads are sanitized: `payment_identifier`, PAN/CVV-like keys, API keys, passwords, and bearer tokens are dropped or redacted.

## Topics

Configurable via `KAFKA_TOPIC_*`. Defaults:

| Topic | Events | Purpose |
| --- | --- | --- |
| `transactions` | transaction-created | Payment entered the scoring pipeline |
| `risk-results` | risk-scored | Synchronous risk decision completed |
| `investigations` | investigation-created, investigation-completed | Case lifecycle (not a command to create rows) |
| `alerts` | alert-created | BLOCK / high-risk REVIEW notifications |
| `feedback` | analyst-feedback-recorded | Analyst APPROVE / BLOCK / ESCALATE |
| `events-dlq` | failed / malformed | Lightweight dead-letter after handler retries |

## Producer / consumer flow

1. `POST /transactions` runs the **existing** scoring pipeline.
2. After DB commit of the **transactional outbox**, a worker publishes through EventBus. Kafka is never called inside the domain transaction (`docs/transactional-outbox.md`).
3. Kafka produce uses a short timeout (`KAFKA_PUBLISH_TIMEOUT_MS`, default 5000ms) and never fails the HTTP risk decision.
4. Consumers (transactions, risk-results, investigations, alerts, feedback) validate schema, log `correlation_id`, and apply idempotent side effects.
5. Consumers do **not** re-score, do **not** ingest the graph, and do **not** insert duplicate investigation rows. Investigations remain created by the pipeline.

## Fallback

| `EVENT_BUS` | Kafka reachable | `EVENT_BUS_FALLBACK` | Active bus | Health |
| --- | --- | --- | --- | --- |
| `inprocess` (default) | n/a | n/a | inprocess | `configured=inprocess`, `active=inprocess` |
| `kafka` | yes | n/a | kafka | `kafka_connected=true` |
| `kafka` | no | `true` (default) | inprocess | `configured=kafka`, `active=inprocess`, `fallback=true`, `reason=connection unavailable` |
| `kafka` | no | `false` | startup error | failure is not hidden |

Fallback is visible in `GET /health` (`event_bus`) and `GET /api/v1/events/status`.

## Idempotency

Duplicate delivery must not create duplicate alerts, duplicate investigations, or extra graph edges.

1. `event_id` is stored in `processed_events` (unique primary key). A second delivery is skipped.
2. Alerts also have unique `(source_event_id)` and `(transaction_id, kind)`.
3. Investigation and graph writes stay in the synchronous pipeline, which already upserts graph entities. Event consumers never call `ingest_transaction`.

## Correlation IDs

`X-Correlation-ID` (or `X-Request-ID` if the correlation header is absent) is bound for the request. Every domain event in that transaction flow carries the same `correlation_id`. Structured logs include it. The response repeats `X-Correlation-ID`.

## Kafka setup (local)

Kafka is in Docker Compose but **the app does not require it**.

```bash
# Default stack — scoring works with EVENT_BUS=inprocess
docker compose up --build

# Optional: run Kafka and point a host API at it
docker compose up kafka
EVENT_BUS=kafka EVENT_BUS_FALLBACK=true KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

The Compose image is `bitnamilegacy/kafka:3.9.0` (Bitnami moved versioned `bitnami/kafka` tags off Docker Hub).

Listeners:

- Host processes: `localhost:9092`
- Backend container: `kafka:19092`

Healthcheck uses `kafka-broker-api-versions.sh`. This is a single-node KRaft broker for lab use, not a production cluster.

## Local development without Docker Kafka

```bash
cd backend
PYTHONPATH=. uvicorn app.main:app --reload --port 8000
```

Default `EVENT_BUS=inprocess`. pytest does not need Kafka.

Optional integration tests:

```bash
RUN_KAFKA_TESTS=1 PYTHONPATH=. pytest tests/test_kafka_integration.py tests/test_events_phase5.py -q
```

## Limitations

- Prototype only. Not production-ready. Not a throughput or durability claim.
- In-process events are lost when the API process exits.
- Kafka consumers are a small set of asyncio tasks in the API process, not a worker fleet.
- Dead-letter handling is a SQL row plus an optional `events-dlq` topic after two handler attempts.
- WebSockets still use the older Redis/memory bus; they are not Kafka consumers.
- GNNs, Kubernetes, multi-agent orchestration, and a feature store are out of scope.

Live broker results (PASS / FAIL / NOT VERIFIED) are in `docs/kafka-verification.md`.
