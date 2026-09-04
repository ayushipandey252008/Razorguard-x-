# LLM investigation agent

This is an independent student prototype and is not an official Razorpay product.

The investigation agent is an **evidence-based explanation layer**. It does not score payments, does not own graph topology, and does not query the database except through a fixed tool registry.

## Architecture

```
Investigation Request
        ↓
Investigator Agent
        ↓
Tool Registry (typed, closed set)
        ↓
Controlled investigation tools
        ↓
Structured evidence
        ↓
LLM synthesis  or  deterministic fallback
        ↓
Grounding (copy risk-engine probability, drop fabricated facts)
        ↓
Structured Investigation Report
```

Packages:

| File | Role |
| --- | --- |
| `backend/app/agents/investigator.py` | Orchestration, grounding, traces |
| `backend/app/agents/registry.py` | Closed typed tool list |
| `backend/app/agents/tools.py` | Tool implementations |
| `backend/app/agents/provider.py` | Provider factory |
| `backend/app/agents/openai_provider.py` | OpenAI-compatible Chat Completions + tool calling |
| `backend/app/agents/fallback_provider.py` | Marker used when no LLM is configured |
| `backend/app/agents/prompts.py` | System prompt and untrusted-data wrapping |
| `backend/app/agents/schemas.py` | Report and tool input contracts |

The live risk path (`services/pipeline.py`, XGBoost, rules, NetworkX ingest) is unchanged. ULB evaluation artifacts are unchanged.

## Tool registry

The agent can only call:

`get_transaction`, `get_user_history`, `get_user_profile`, `get_user_baseline`, `check_device`, `check_ip`, `check_location`, `check_transaction_velocity`, `get_model_explanation`, `get_triggered_rules`, `find_connected_accounts`, `find_fraud_cluster`.

There is no SQL tool and no generic database handle. Unknown names are rejected. Arguments are validated against Pydantic input models. Missing lookups return `{ "unavailable": true, "reason": "..." }` and are never filled with invented values.

`get_transaction` does not return `payment_identifier`.

## Agent flow

1. Load the transaction via `get_transaction`.
2. If no LLM is configured, run the deterministic investigator (fixed relevant tool sequence, report assembled only from results).
3. If an LLM is configured, send the investigation objective, frozen risk-engine snapshot, tool definitions, and wrapped seed data. The model may call tools over several rounds.
4. Tool outputs are wrapped as untrusted DATA, not instructions.
5. Grounding always copies `ml_probability`, `ml_score`, `model_version`, and the engine decision from `get_model_explanation`. Invalid `recommendation` values are rejected. Fabricated clusters are dropped unless `find_fraud_cluster` actually found one. Transaction facts are taken from the tool, not from free-form LLM text.
6. Persist the report and each tool call. Never persist API keys.

The LLM does not have to call every tool. The fallback investigator calls a broader fixed set so a demo still has a complete trace when no key is present.

## Provider abstraction

Environment:

- `LLM_PROVIDER` — `none` / `deterministic` → fallback; `openai` / `llm` → OpenAI-compatible if a key exists
- `LLM_MODEL`
- `OPENAI_API_KEY` or `LLM_API_KEY`
- `LLM_BASE_URL` (optional; default OpenAI)

If no valid configuration exists, `provider = "deterministic_fallback"`. If an LLM call succeeds, `provider = "llm"`. If the LLM errors, the agent falls back and still marks `deterministic_fallback`. Keys are never hardcoded and never returned by the API.

Future vendors implement `complete_with_tools` without changing the investigator.

## Grounding strategy

- Transaction / user / device / IP / location strings are DATA. Prompt-injection text in those fields cannot authorize SQL, cannot change the fraud probability, and cannot mint a cluster.
- Tool JSON is wrapped in `<untrusted_data>` delimiters.
- Recommendation is copied from the risk engine (`APPROVE` | `REVIEW` | `BLOCK`). The LLM may explain that decision; it may not publish its own probability.
- Confidence is labeled **qualitative** unless a real calibrator is cited. The numeric `confidence` field is the existing risk-engine value, not a new calibrated agent score.
- Logs run through a redaction processor (`api_key`, bearer tokens, `sk-…` prefixes).

## Fraud-cluster logic

`find_fraud_cluster` uses the existing NetworkX graph (no Neo4j). A cluster is a connected component of users who share a device or IP, with a configurable minimum size (default 3). Merchant-only hops are not a ring.

When a cluster exists the tool returns `cluster_found=true` plus size, shared devices/IPs, relationships, suspicious nodes, and risk indicators. When none exists it returns `cluster_found=false` and `reason: "No connected suspicious cluster found"` (legacy `identified=false` / `message: "no suspicious cluster identified"` remain for older tests). Clusters are not fabricated.

Prototype graph indicators (configurable, **not production-grade**):

- device shared by ≥ N accounts
- IP shared by ≥ N accounts
- multiple previously flagged accounts on one device
- flagged transactions connected through an entity
- dense user–user subgraph
- short paths between flagged accounts

Thresholds are exposed on `GET /api/v1/health`, `GET /api/v1/graph/thresholds`, and analytics model information.

## Fallback behavior

No key, `LLM_PROVIDER=none`, or an LLM HTTP/transport failure → deterministic investigator, no crash, provider clearly marked `deterministic_fallback`.

## API

- `POST /api/v1/investigations/{investigation_id}/run` — also accepts a **transaction id** (creates a case if needed) so there is no second conflicting route
- `GET /api/v1/investigations/{investigation_id}`
- `GET /api/v1/investigations/{investigation_id}/trace`

Responses include investigation id, recommendation, risk level, confidence, summary, evidence, tool trace, provider, and limitations.

## Security model

- Closed tool registry
- No arbitrary SQL
- Schema-validated arguments
- Unavailable for missing evidence
- Engine probability cannot be overridden
- Graph links cannot be invented
- Invalid recommendations rejected
- Provider failure falls back
- Secrets redacted from logs, traces, and responses
- Prompt injection in payment fields treated as data

## Limitations

This is a student prototype. It is not production-ready, not a real-world fraud accuracy claim, and not an official Razorpay product. Graph thresholds are heuristics. The LLM, when used, can still misunderstand evidence; grounding reduces but does not eliminate that risk. Optional Neo4j and Kafka are lab transports with in-process fallbacks. GNNs, Kubernetes, a feature store, multi-agent swarms, RAG, and voice are not implemented.
