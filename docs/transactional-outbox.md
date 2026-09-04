# Transactional outbox (Phase 6)

This is an independent student prototype and is not an official Razorpay product.

Durable event delivery is **at-least-once**, not exactly-once. Consumers must stay idempotent. This is not a production message bus.

## Problem

Phase 5 published domain events *after* the risk row was committed:

```
DB commit  →  EventBus.publish  →  Kafka / in-process
```

If the process died between commit and publish, the payment decision existed in SQL and the event did not. Kafka inside the same DB transaction would be worse: a broker timeout could roll back a valid risk decision.

## Phase 5 vs Phase 6

```
Phase 5:
  DB commit → publish

Phase 6:
  DB transaction
    persist transaction + risk (+ investigation)
    insert outbox event(s)
  commit
  → worker
  → EventBus
  → Kafka or in-process
```

Kafka is never called while the domain transaction is open. The database is the source of event intent.

## Failure window closed

| After a successful commit | Outbox row |
| --- | --- |
| API crash before publish | `PENDING` — worker publishes later |
| Kafka down | `PENDING` / retry — API still returns the risk decision |
| Worker crash while `PROCESSING` | stale claim released back to `PENDING` |
| Malformed envelope | `FAILED` + failed-event/DLQ, no endless retry |
| Retry budget exhausted | `FAILED` with `last_error` (ADMIN can retry) |

If the DB transaction rolls back, risk state and outbox rows both disappear.

## Components

```
backend/app/models/outbox.py      OutboxEvent table
backend/app/events/outbox.py       enqueue / claim / status
backend/app/events/outbox_worker.py poll + publish through EventBus
backend/app/workers/event_consumer.py  standalone consumer + outbox loop
backend/app/workers/outbox.py      standalone outbox publisher
```

Statuses: `PENDING` → `PROCESSING` → `PUBLISHED`, or `FAILED`.

`event_id` is unique. Claiming:

- PostgreSQL: `SELECT … FOR UPDATE SKIP LOCKED`
- SQLite: `UPDATE … WHERE status='PENDING'` and keep the row only when `rowcount=1`

Two workers must not publish the same row at the same time. Duplicate *delivery* after a crash is still possible (at-least-once). Handlers key on `event_id`.

## Worker

`OUTBOX_ENABLED=true` (default).

The worker:

1. Releases stale `PROCESSING` rows
2. Claims a batch (`OUTBOX_BATCH_SIZE`)
3. **Commits the claim** (so Kafka is not inside the domain txn)
4. Publishes each row through `EventBus`
5. Marks `PUBLISHED`, or schedules backoff, or `FAILED`

Backoff: `OUTBOX_RETRY_BACKOFF_SECONDS * 2^(attempts-1)`, capped at 60s. After `OUTBOX_MAX_ATTEMPTS`, the row is `FAILED`.

Transport failures retry. Malformed payloads fail permanently and go to the Phase 5 failed-event/DLQ path.

When `EVENT_BUS=kafka` and the broker is down, the worker **does not** treat in-process fallback as success. Rows stay `PENDING`.

When `EVENT_BUS=inprocess`, the same outbox path is used: DB → outbox → worker → in-process bus. Kafka is not required.

`OUTBOX_DRAIN_AFTER_COMMIT=true` (default) runs one worker batch after the HTTP handler commits so local tests and the UI see events quickly. That drain still happens **after** commit. Crash tests skip it.

## Standalone processes

API (default):

```bash
cd backend && PYTHONPATH=. uvicorn app.main:app --reload
```

Outbox + Kafka consumers without FastAPI:

```bash
cd backend && PYTHONPATH=. python -m app.workers.event_consumer
```

This process starts Kafka consumers when `EVENT_BUS=kafka` even if the API has `EVENT_CONSUMER_IN_API=false`. Delivery remains at-least-once.

Outbox only:

```bash
cd backend && PYTHONPATH=. python -m app.workers.outbox
```

Optional Compose profile (does not start unless requested):

```bash
docker compose --profile workers up --build
```

Do not run two consumer processes against the same Kafka group unless you accept at-least-once duplicate deliveries (handlers are idempotent).

## API

| Method | Path | Who |
| --- | --- | --- |
| GET | `/api/v1/events/outbox/status` | authenticated |
| POST | `/api/v1/events/outbox/drain` | ADMIN |
| POST | `/api/v1/events/outbox/{event_id}/retry` | ADMIN |

Status returns counts only: pending, processing, published, failed, oldest pending age, last success/failure. No payloads, no secrets.

## Delivery semantics

| Layer | Role |
| --- | --- |
| Database | durable source of risk state **and** event intent |
| Outbox | durable pending publish |
| EventBus | transport (in-process or Kafka) |
| Consumers | idempotent side effects |

This is **at-least-once**. It is not exactly-once. It is not production Kafka.

## Limitations

- Prototype only. SQLite claiming is good enough for one local process, not a multi-region outbox.
- In-process events still die with the process *after* publish; durability is the outbox **until** publish succeeds.
- Background worker is disabled in pytest (`OUTBOX_BACKGROUND_WORKER=false`) so crash tests are deterministic.
- No Kubernetes, GNN, feature store, or multi-agent orchestration.
