# Kafka verification report

Status of the live broker path: **PASS** (host Python process against Compose Kafka on `localhost:9092`).

This is not a production durability or throughput claim. Kafka remains optional (`EVENT_BUS=inprocess` by default).

## Root cause(s)

These were observed, not hypothetical:

1. **Compose image was unpullable.** `bitnami/kafka:3.9` is gone from Docker Hub. The broker never started until the image was switched to `bitnamilegacy/kafka:3.9.0`.
2. **No consumer on `transactions`.** `KafkaEventBus.start_consumers` subscribed to risk-results, investigations, alerts, and feedback only. `transaction-created` was produced and never consumed.
3. **Producer timeouts were too aggressive.** `request_timeout_ms` tracked the 800ms publish timeout, which is too short for metadata / first produce / topic auto-create.
4. **Topic creation raced auto-create.** The first produce depended on broker auto-create with no admin ensure step.
5. **Admin client never closed.** `AIOKafkaAdminClient` exposes `close()`, not `stop()`. `await admin.stop()` always failed and was swallowed, leaking the admin connection.
6. **Compose services did not wait on Kafka health** even when `EVENT_BUS=kafka` (now `depends_on` with `condition: service_healthy` and `required: false` so Kafka stays optional).

## Files changed

- `docker-compose.yml` — `bitnamilegacy/kafka:3.9.0`; optional healthy Kafka dependency for backend/event-worker
- `backend/app/config.py` — `kafka_group_transactions`, connect timeout 15s, publish timeout 5000ms
- `backend/app/events/kafka_bus.py` — transactions consumer, topic ensure, producer request timeout floor 15s, admin `close()`, DLQ raw preview
- `backend/app/workers/event_consumer.py` — docstring matches actual topics
- `.env.example` — group names, connect/publish timeouts
- `docs/event-driven-architecture.md` — image, consumers, test command
- `backend/tests/test_events_phase5.py` — regression tests (transactions consumer, pullable image, admin close)
- `backend/tests/test_kafka_integration.py` — opt-in real-broker tests (`RUN_KAFKA_TESTS=1`)

No LLM, ML artifacts, ULB/IEEE evaluation files, Prisma, Kubernetes, or extra brokers were modified.

## Kafka version / configuration

| Item | Value |
| --- | --- |
| Image | `bitnamilegacy/kafka:3.9.0` (linux/arm64, id `55df55bfc7ed`) |
| Mode | single-node KRaft (`PROCESS_ROLES=controller,broker`) |
| Host bootstrap | `localhost:9092` (EXTERNAL) |
| Docker bootstrap | `kafka:19092` (INTERNAL) |
| Client | `aiokafka==0.12.0` |
| Auto-create | enabled, plus explicit `create_topics` on producer connect |
| Default app mode | `EVENT_BUS=inprocess`, `EVENT_BUS_FALLBACK=true` |

## Broker status

**PASS.** Container `razorgaurdx-kafka-1` was `Up (healthy)`. Healthcheck `kafka-broker-api-versions.sh --bootstrap-server 127.0.0.1:9092` succeeded. Broker API versions responded on `localhost:9092`.

## Topics verified

**PASS** against the running broker (`kafka-topics.sh --list` / `--describe` after tests):

- `transactions`
- `risk-results`
- `investigations`
- `alerts`
- `feedback`
- `events-dlq`

Each has 1 partition, replication factor 1, leader 0 in ISR.

## Producer verification

**PASS.** `AIOKafkaProducer.send_and_wait` with `acks=1`. Publish results returned `ok=True` and `event_bus=kafka`. Prototype produce latencies after topics existed: **0.72–1.96 ms** per event type (not a benchmark).

Closed/unconnected producer returns `ok=False` and does not report success.

## Consumer verification

**PASS.** Five consumers start (`transactions`, `risk-results`, `investigations`, `alerts`, `feedback`). Independent readers confirmed `event_id` and `correlation_id` on every produced type. Application consumers wrote `processed_events` for all six handler event types.

## Outbox verification

**PASS.**

| Guarantee | Result |
| --- | --- |
| Successful DB transaction creates the outbox row | PASS (pipeline + outbox tests) |
| Rolled-back transaction leaves no outbox row | PASS (`test_rollback_removes_outbox_event`, SQLite, no broker required) |
| Kafka down leaves the row PENDING / retryable | PASS (unit: status kafka_connected=false; broker: producer closed → PENDING with `attempts>=1` and `last_error`) |
| Successful Kafka publish marks PUBLISHED | PASS (`drain_outbox_batch` after live produce) |
| Kafka failure is not treated as published | PASS (`_transport_accepted` rejects fallback; closed producer → not PUBLISHED) |

## Idempotency verification

**PASS.** Same `alert-created` `event_id` published twice; `process_event` returned `duplicate=True`; one `alerts` row for that transaction.

## Retry / DLQ verification

**PASS.**

- Handler raised `controlled_handler_failure`: **2** in-process attempts (`MAX_HANDLER_ATTEMPTS`), then `failed_events.retry_count==2`, DLQ message contained `event_id`, offset committed (no infinite loop).
- Malformed JSON on `alerts`: `failed_events` persisted with preview, DLQ contained the unique marker, consumer continued.

## Failure recovery verification

**PASS.**

- `EVENT_BUS=kafka` + `EVENT_BUS_FALLBACK=true` + `KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:1`: HTTP `POST /api/v1/transactions` returned 200 with a risk decision; `/health` showed `configured=kafka`, `fallback=true`, `kafka_connected=false`.
- In-process mode is unchanged when `EVENT_BUS` is unset/inprocess.

## Latency

Prototype only, local broker, after topics existed:

- Per-topic produce: 1.96, 0.74, 0.72, 0.80, 1.13, 0.82 ms (transaction, risk, investigation-created, investigation-completed, alert, feedback)
- Optional roundtrip publish: 1.51 ms

End-to-end pipeline+outbox+consume was not timed as a SLA; it completed inside the pytest function (seconds, dominated by scoring/DB).

## Exact commands used

```bash
docker compose up -d kafka
docker compose ps kafka
docker exec razorgaurdx-kafka-1 kafka-broker-api-versions.sh --bootstrap-server 127.0.0.1:9092
docker exec razorgaurdx-kafka-1 kafka-topics.sh --bootstrap-server 127.0.0.1:9092 --list
docker exec razorgaurdx-kafka-1 kafka-topics.sh --bootstrap-server 127.0.0.1:9092 --describe

cd backend
PYTHONPATH=. ./.venv/bin/pytest -q --tb=line
RUN_KAFKA_TESTS=1 PYTHONPATH=. ./.venv/bin/pytest tests/test_kafka_integration.py tests/test_events_phase5.py::test_optional_kafka_broker_roundtrip -q --tb=short
RUN_KAFKA_TESTS=1 PYTHONPATH=. ./.venv/bin/pytest -q --tb=line
```

## Tests passed / skipped

| Run | Result |
| --- | --- |
| Default suite (no `RUN_KAFKA_TESTS`) | **162 passed, 12 skipped** (2026-09-05; Kafka + Neo4j integrations skipped) |
| `RUN_KAFKA_TESTS=1` Kafka file | **11 passed** against the live broker (recorded during Kafka verification) |
| `RUN_KAFKA_TESTS=1` full suite | **167 passed, 1 skipped** (`RUN_NEO4J_TESTS` not set; recorded during Kafka verification) |

No existing tests were deleted or weakened.

## Remaining limitations

- **NOT VERIFIED:** Dockerized `backend` / `event-worker` containers with `EVENT_BUS=kafka`. Verification used the same Python modules on the host against Compose Kafka. Those Compose services default to `EVENT_BUS=inprocess`.
- **NOT VERIFIED:** Multi-broker / SASL / TLS / exactly-once producer semantics.
- Kafka is still optional. Local pytest and `EVENT_BUS=inprocess` do not need the broker.
- Consumers are asyncio tasks in the API or `python -m app.workers.event_consumer`, not a consumer fleet.
- Dead-letter is a SQL `failed_events` row plus a compact `events-dlq` envelope (reason, event_id, raw preview), not a replay product.
- WebSocket live-wire still uses the older Redis/memory bus.
- Single-node KRaft lab broker only.
