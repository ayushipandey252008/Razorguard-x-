# RazorGuard X — Audit (hardening phase)

**Historical snapshot** of the working MVP **before** the hardening pass. Current architecture, Kafka/outbox, Neo4j, feedback, and evaluation tracks are documented in `README.md` and `docs/`. Prototype / synthetic data. Not a production fraud system. Not affiliated with Razorpay.

Later phases in this effort address agent/graph tools, ML evaluation tracks, calibration, behavior overlays, e2e tests, security, and docs. This file stays as the pre-hardening record.

## 1. Current architecture

Single FastAPI process scores payments; Next.js is a presentation layer.

`TransactionCreate` → history enrichment → persist → XGBoost (+ optional SHAP) → Isolation Forest + interpretable behavior flags → NetworkX ingest → rule catalog → weighted combiner → APPROVE | REVIEW | BLOCK → investigation row if not APPROVE → audit + WebSocket.

Simulation and the transaction API share `services/pipeline.py`. The agent may only call named tools (no arbitrary SQL). Graph queries are in-process NetworkX, rebuilt from transactions on boot. SQL `graph_entities` / `graph_relationships` are an upsert write path.

## 2. Implemented components

| Area | Status |
| --- | --- |
| Ingest, enrichment, SQLite/Postgres | Implemented |
| XGBoost (libomp / HGB fallback), SHAP, Isolation Forest | Implemented |
| Rule engine with evidence payloads | Implemented |
| NetworkX graph + ring detector | Implemented (see §8) |
| Weighted risk combiner + JWT/RBAC | Implemented |
| Deterministic investigation agent + optional LLM | Implemented (see §7) |
| Simulation, analytics, live WebSocket | Implemented |
| Dashboard pages (command, wire, cases, graph, telemetry, range) | Implemented |
| Pytest API/unit suite | Implemented (gaps in §9) |

## 3. Known limitations

- Labels and features are synthetic; metrics do not transfer to live payments.
- Thresholds 40 / 70 were prototype constants (not cost-calibrated).
- Default `SECRET_KEY` is a development placeholder.
- Redis optional; events otherwise in-memory (lost across processes).
- LLM unused unless `LLM_API_KEY` is set.
- Graph not Neo4j; no Kafka, GNN, feature store, or real acquirer integration.

## 4. Technical debt

- `find_fraud_cluster` keyed only by `user_id` and returned `unavailable` when no cluster existed (sounds like a missing tool).
- `connected_users` used full ego-graph (including merchants), inflating “connections” vs the device/IP projection used for rings.
- `cluster_id` regenerated with UUID on every detect call (unstable).
- `ml_probability` displayed as a percentage without stating calibration.
- `ModelVersion` / graph SQL were late add-ons; query path still NetworkX.
- `python-jose` uses naive `utcnow` internally.
- Alembic initial migration is `create_all`, not a diff.

## 5. Security concerns

- Default JWT secret if env is unset; production must refuse it.
- Seed password `prototype-pass` documented for lab use only.
- CORS allow-list is env-driven but credentials are enabled.
- 500s could leak traces via Starlette debug defaults.
- Login not separately tighter-rate-limited than the global limiter.
- WebSocket accepts a missing token (hello still sent) — lab convenience, not production.
- No PAN/CVV by design; payment identifiers are synthetic tokens.

## 6. ML / data-science concerns

- Target is the generator’s own `is_fraud` flag (pattern recovery, not hidden fraud).
- ~12% injected fraud + 3% label flips; still not real prevalence (~0.1%).
- Features (new device, velocity, amount ratio) are the same signals used to *create* labels → leakage by construction.
- Stratified random split, not a time split (timestamps are random-ish, not a true stream).
- Isolation Forest is global; user baselines are rule-like overlays.
- No calibration curve was stored; raw `predict_proba` treated as P(fraud).
- Synthetic and any public dataset must not share a metrics table.

## 7. Agent limitations

- Default path is a fixed tool list, not an LLM planner.
- `find_fraud_cluster` did not take `transaction_id` and used `unavailable` for “no cluster”.
- Reports can recommend BLOCK because the risk engine already did; the agent does not independently relitigate.
- Tool traces are stored on the investigation JSON (needed for evidence, noisy for UI).

## 8. Graph limitations

- In-memory; rebuilt at startup.
- Ring score mixed user count, device/IP counts, and density with large coefficients.
- Graph risk added `4 * connected_users` including merchant- hop neighbors.
- Two users sharing a device is sharing, not automatically a “ring” (need ≥3 for that language).

## 9. Testing gaps

- No explicit APPROVE / REVIEW / BLOCK behavioral cases (only “one of the three”).
- Ring simulation did not assert cluster identification in the agent report.
- No public-dataset evaluation test (optional skip if CSV absent).
- Frontend tests cover formatters only.
- No Playwright.

## 10. Production-scaling gaps

- One process, one graph, one model file.
- No model registry, drift jobs, or online learning.
- SQLite in local/dev; Compose Postgres unproven in this environment (daemon was down).
- No horizontal WebSocket fan-out without Redis.
- Thresholds and weights are process env, not a change-audit UI.
