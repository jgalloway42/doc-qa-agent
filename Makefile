.PHONY: install lint format type-check test test-cov ingest ui mlflow clean docker-build docker-up

install:
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

type-check:
	mypy doc_qa/ --ignore-missing-imports

test:
	pytest

test-cov:
	pytest --cov=doc_qa --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

ingest:
	python -m cli.ingest_cli ingest docs/

ui:
	streamlit run ui/app.py

mlflow:
	mlflow ui --backend-store-uri ./mlruns --port 5000

clean:
	rm -rf chroma_db/ mlruns/ test_mlruns/ test_chroma_db/ .ingested_hashes.json __pycache__/ .pytest_cache/ htmlcov/ coverage.xml

docker-build:
	docker build -t doc-qa-agent .

docker-up:
	docker-compose up
