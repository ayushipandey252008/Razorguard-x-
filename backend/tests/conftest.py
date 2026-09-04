"""Backend tests that disable process-wide rate limits (login is 10/minute in lab)."""

import os
from pathlib import Path

# Avoid a background outbox poller racing crash-recovery assertions.
os.environ.setdefault("OUTBOX_BACKGROUND_WORKER", "false")
# Tests must stay on the deterministic investigator even if local .env points at Ollama.
os.environ["LLM_PROVIDER"] = "none"
os.environ["LLM_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
# Isolate the default suite from a developer Neo4j .env (persistent rings leak into
# "unique device" cases). Opt in with RUN_NEO4J_TESTS=1.
if os.environ.get("RUN_NEO4J_TESTS") != "1":
    os.environ["GRAPH_BACKEND"] = "networkx"
    os.environ["GRAPH_NEO4J_FALLBACK"] = "true"
# Do not reuse the operator SQLite file (seed + simulations pollute graph tests).
_test_db = Path(__file__).resolve().parent.parent / "razorguard_pytest.db"
if _test_db.exists():
    _test_db.unlink()
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_test_db}"

from app.config import get_settings
from app.rate_limit import limiter

get_settings.cache_clear()
limiter.enabled = False
