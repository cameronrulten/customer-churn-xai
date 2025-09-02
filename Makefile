PY:=uv run
export PYTHONWARNINGS=ignore

.PHONY: env install lint format test train serve dashboard docker-up docker-down docker-build

env:
	uv sync

install: env

lint:
	uv run ruff check .
	uv run black --check .

format:
	uv run black .
	uv run ruff check . --fix

test:
	$(PY) pytest -q

train:
	$(PY) python -m src.models.train --config configs/model.yaml

serve:
	$(PY) uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(PY) streamlit run src/viz/dashboard.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
