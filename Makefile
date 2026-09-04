.PHONY: train test backend frontend up
train:
	PYTHONPATH=backend python ml/training/train_baseline.py
test:
	cd backend && PYTHONPATH=. pytest -q
backend:
	cd backend && PYTHONPATH=. uvicorn app.main:app --reload --port 8000
frontend:
	cd frontend && npm run dev
up:
	docker compose up --build
