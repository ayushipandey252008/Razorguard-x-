# Investigation agent

The agent cannot run arbitrary SQL. It may only call named tools in `app/agents/tools.py` via the typed registry in `app/agents/registry.py`.

Full Phase 3 write-up: [llm-investigation-agent.md](./llm-investigation-agent.md).

This is an independent student prototype and is not an official Razorpay product.

## Tools

`get_transaction`, `get_user_history`, `get_user_profile`, `get_user_baseline`, `check_device`, `check_ip`, `check_location`, `check_transaction_velocity`, `find_connected_accounts`, `find_fraud_cluster`, `get_model_explanation`, `get_triggered_rules`

`find_fraud_cluster` requires `transaction_id`. It returns structured cluster evidence (`cluster_found`, `cluster_id`, connected users, shared devices/IPs, merchants, relationship counts, graph risk, explanation) or `cluster_found=false` / `identified=false` with message **no suspicious cluster identified**. Missing lookup data is `{ "unavailable": true, "reason": "..." }` — that is not used for “no ring”.

## Providers

`LLM_PROVIDER` + `OPENAI_API_KEY` or `LLM_API_KEY` + `LLM_MODEL`. When an LLM is used, reports mark `provider=llm`. If the key is missing or the provider errors, `deterministic_fallback` runs registered tools and writes the JSON report **only** from those payloads.

## Report shape

Investigation id, transaction id, provider, summary, risk level, recommendation (`APPROVE|REVIEW|BLOCK`), qualitative confidence, model/behavior/rule/graph evidence, key findings, tool trace, limitations.

Legacy keys remain: transaction summary, risk assessment, evidence (including unavailable), suspicious signals, connected entities, potential fraud-ring (`identified` true/false).

Recommended action follows the risk engine decision already stored. The agent copies `ml_probability` from the risk engine and does not independently relitigate scores.
