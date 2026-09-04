# Deployment

Cloud deployment (Render / Vercel / managed Postgres) is **not configured in this repository yet**. That is a separate phase.

This file documents **local** runtimes only. Do not treat `localhost` URLs as production endpoints.

## Local processes

See the root `README.md` for exact commands. Summary:

- Backend: FastAPI on `http://127.0.0.1:8000` (`/api/v1/health`, `/docs`)
- Frontend: Next.js on `http://localhost:3000`
- Optional Compose: Postgres, Redis, Neo4j, Kafka

Copy `.env.example` to `.env`. Never commit `.env`.

Frontend browser variables are only:

- `NEXT_PUBLIC_API_URL`
- `NEXT_PUBLIC_WS_URL`

Do not put JWT secrets, database passwords, Neo4j/Kafka credentials, or LLM keys in `NEXT_PUBLIC_*`.

## Compose

```bash
docker compose up --build
```

Optional Kafka workers:

```bash
docker compose --profile workers up --build
```

## What is not ready

- No `render.yaml` / Vercel project config in this repo
- No production secret manager
- Default JWT secret and seed password are lab-only
- Kafka and Neo4j were verified locally; they are not a production cluster

Next step after the public GitHub repository exists: audit a cloud deployment plan (separate phase).
