# Local LLM investigation verification

Independent student prototype. Not an official Razorpay product. Not a production
fraud-accuracy claim. Synthetic RazorGuard data only.

**Status: `LOCAL_LLM_VERIFIED`**

A real Llama 3.1 8B investigation completed through the existing
`OpenAIProvider` against local Ollama. No paid API and no external LLM provider
were used.

## Runtime

| Item | Value |
| --- | --- |
| Ollama | 0.33.2 (Homebrew) |
| Model | `llama3.1:8b` (Q4_K_M, 4.9 GB) |
| Provider class | `OpenAIProvider` |
| Provider name | `llm` |
| `LLM_PROVIDER` | `ollama` |
| Local endpoint | `http://127.0.0.1:11434/v1/chat/completions` |
| API key | dummy `ollama` (ignored by Ollama; required by the existing config gate) |
| Hardware | Apple M3, 24 GB unified memory, Metal iGPU |

`.env` remains gitignored. `.env.example` was not given a dummy key.

## Preflight

| Check | Result |
| --- | --- |
| Ollama reachable at `127.0.0.1:11434` | yes |
| `llama3.1:8b` present locally | yes |
| `llm_is_configured()` | `True` |
| Resolved class | `OpenAIProvider` |
| `LLM_BASE_URL` is loopback | yes |
| OpenAI / OpenRouter / Groq / A4F host | not used |
| Deterministic fallback still available | yes (used in the failure test) |

## Controlled local investigation

Synthetic high-risk `stolen_account` transaction only. No real customer or
payment data.

| Field | Value |
| --- | --- |
| provider | `llm` |
| model | `llama3.1:8b` |
| investigation ID | `55eda86a-99f9-4b36-9b9d-9b50f2d5e522` |
| transaction ID | `txn_c3af9777843c4f39` |
| engine decision | `BLOCK` |
| engine `ml_probability` | 0.9800000190734863 |
| engine `final_risk_score` | 70.13 |
| engine model | `xgb-iforest-v1-calibrated` |
| LLM-initiated tool calls | 2 (`get_transaction`, `get_model_explanation`) |
| evidence items | 2 |
| recommendation | `BLOCK` |
| fallback | no |
| wall latency | 16025 ms |

Llama used native OpenAI-style `tool_calls` over two rounds, then returned a
report. `_ground_report` copied the risk-engine probability, score, and
`BLOCK` decision. Grounding matched the engine on all three. No cluster was
fabricated.

## Tool-calling compatibility

Ollama OpenAI-compat supports `tools`, `tool_calls`, `tool_call_id`, `role=tool`,
and multi-round calling. `tool_choice` is accepted and ignored.

A one-tool diagnostic returned structured `tool_calls` in ~3 s. The full
RazorGuard prompt initially caused Llama 3.1 8B to print tool calls as ordinary
JSON text and generate until HTTP timeout (45 s, then 180 s). Isolated fixes
that made the existing loop succeed:

1. Local HTTP timeout 180 s (remote stays 45 s).
2. Local `max_tokens=768` so a text dump cannot hang the round.
3. System prompt: use native `tool_calls` first; write the JSON report only
   after tool results exist.

The `name` field on `role=tool` messages did **not** cause a 400. It was not
changed.

## Agent boundaries

Closed registry only: `get_transaction`, `get_user_history`, `get_user_profile`,
`get_user_baseline`, `check_device`, `check_ip`, `check_location`,
`check_transaction_velocity`, `get_model_explanation`, `get_triggered_rules`,
`find_connected_accounts`, `find_fraud_cluster`.

No SQL tool, no filesystem tool, no shell tool, no arbitrary HTTP tool. Backend
`ToolBox` executed the two requested tools and returned JSON to the model.

## Risk-decision separation

`_ground_report` remained authoritative. On the live Llama run:

- report `ml_probability` = engine `ml_probability`
- report `final_risk_score` = engine `final_risk_score`
- recommendation = engine `BLOCK`

Llama cannot change ML probability, anomaly/rule/graph component scores, final
risk score, or APPROVE/REVIEW/BLOCK.

## Failure / fallback test

Ollama was stopped (`ConnectError` to `127.0.0.1:11434`). A second synthetic
`stolen_account` investigation (`txn_d918db12d60e48d9` /
`37fe66ba-a61d-4f89-8f7a-a6963b213eb7`) completed as `deterministic_fallback`
in 79 ms with 12 tool-sourced evidence items. Engine decision stayed `BLOCK`
and probabilities were unchanged. Ollama was restored afterward;
`llama3.1:8b` is still local.

## Security / redaction

No API secrets, Bearer tokens, `sk-` keys, or `payment_identifier` appeared in
the investigation payload. Logs use `app.utils.redact`. The dummy local key is
not a credential.

## Integrity

- Live model artifacts unchanged: `xgb_fraud.joblib` sha256
  `79b2b82ae1c9722178223881953cb92f6d47188280dc083c5f897c551d30ee5c`,
  version `xgb-iforest-v1-calibrated`.
- No ULB or IEEE training or evaluation overwrite.
- No keys in this markdown or the companion JSON.

## Tests

- `tests/test_agent_phase3.py`: 30 passed (registry, grounding, redaction,
  fallback, config gate, local timeout).
- Full backend suite: 153 passed, 2 skipped.
- IEEE/ULB training was not rerun.

Pytest forces `LLM_PROVIDER=none` so the suite does not call Ollama.

## Limitations

- Llama 3.1 8B called 2 of 12 registered tools, then stopped. Coverage is
  narrower than the deterministic investigator.
- Ollama ignores `tool_choice`.
- Local 8B tool rounds need a token cap; unbounded generation timed out.
- Dummy `LLM_API_KEY=ollama` is still required by `llm_is_configured()`.
- Qualitative confidence is not a calibrated probability.
- Prototype only; not production fraud accuracy.

## Final status

`LOCAL_LLM_VERIFIED`
