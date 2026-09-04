# Security

Lab prototype. Not a production control plane.

## Currently implemented

- JWT access tokens (`SECRET_KEY`, HS256). `ENVIRONMENT=production` refuses placeholder secrets shorter than 32 characters and the documented lab seed password.
- Password hashing via bcrypt (`app/security/passwords.py`)
- Roles: ADMIN, RISK_ANALYST, INVESTIGATOR, VIEWER
- Simulation and transaction ingest require analyst-level roles; decisions require RISK_ANALYST/ADMIN
- Pydantic validation on write payloads (length/range limits on transaction fields)
- SlowAPI default rate limit (`RATE_LIMIT_PER_MINUTE`); login is additionally limited to 10/minute
- CORS allow-list from `CORS_ORIGINS` (credentials enabled for the dashboard origin)
- Audit log for ingest, scoring, investigations, analyst decisions
- Generic 500 responses (no traceback in the body)
- SQLAlchemy bound parameters (no string-built SQL)
- Request correlation via `X-Request-ID`
- `.env` is gitignored; use `.env.example`
- No real card data by design — payment identifiers are synthetic tokens

## Baseline / lab concessions

- Default `SECRET_KEY` and seed password `prototype-pass` exist for local demo only
- WebSocket may omit a token **outside** production so the live wire works in the lab
- Rate limits are in-process (not shared across workers)

## Not implemented

HTTPS termination, secret rotation, WAF, per-tenant isolation, field-level encryption at rest, production-grade audit SIEM.
