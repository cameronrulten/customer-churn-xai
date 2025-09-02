# Customer Churn XAI (End-to-End)

**Goal:** Predict customer churn and **explain** decisions. Full pipeline:
ingestion → feature engineering → training → evaluation → **FastAPI** service → **Streamlit** dashboard.
Includes SQL storage and Docker orchestration.

## Data
Default: IBM Telco Customer Churn (public). See `configs/data.yaml` for paths.
You can switch to Olist (delivery ETA forecasting) or NYC TLC with minor config edits.

## Quickstart (uv + Docker)
```bash
# create env & install
uv sync
# run tests & lint
make test lint
# train model
make train
# launch API & dashboard (docker-compose)
make docker-up
