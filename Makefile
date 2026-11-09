PY := .venv/bin/python
BIN := .venv/bin
REPORT_DIR ?= reports/reproduction
ARTIFACT_DIR ?= artifacts/real
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib

.PHONY: install acquire eda train fixture test lint serve
install:
	uv sync --frozen --extra dev
acquire:
	$(PY) scripts/acquire.py
eda:
	$(BIN)/fraud eda --csv data/raw/creditcard.csv --output reports/eda
train:
	$(BIN)/fraud train --csv data/raw/creditcard.csv --kind real --output $(REPORT_DIR) --artifacts $(ARTIFACT_DIR)
fixture:
	$(BIN)/fraud fixture
	$(BIN)/fraud train --csv data/synthetic/fixture.csv --kind synthetic --output reports/synthetic --artifacts artifacts/synthetic
lint:
	$(BIN)/ruff check src tests scripts
	$(BIN)/ruff format --check src tests scripts
test:
	$(BIN)/pytest -q
serve:
	$(BIN)/uvicorn fraud_detection.api.app:app --host 127.0.0.1 --port 8000
