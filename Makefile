.PHONY: download-data run-tier1 run-tier2 run-evaluation run-shap run-plots run-all lint test

download-data:
	python scripts/download_data.py

run-tier1:
	python scripts/run_tier1.py

run-tier2:
	python scripts/run_tier2.py

run-evaluation:
	python scripts/run_evaluation.py

run-shap:
	python scripts/run_shap.py

run-plots:
	python scripts/run_plots.py

run-all: download-data run-tier1 run-tier2 run-evaluation run-shap run-plots

lint:
	ruff check .
	mypy src/

test:
	pytest tests/ -x --ignore=tests/test_integration.py -m "not network"

test-all:
	pytest tests/ -x

test-integration:
	pytest tests/test_integration.py -x
