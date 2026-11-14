.PHONY: help install test check lint run-ui generate-fixtures clean

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
UVICORN := .venv/bin/uvicorn

help:
	@echo "NeuroAxonNet - Phase 01 Engineering Environment"
	@echo "-----------------------------------------------"
	@echo "make install           - Install project in editable mode"
	@echo "make generate-fixtures - Generate synthetic test microscopy patches and manifests"
	@echo "make test              - Run all unit, golden, leakage, and integration tests"
	@echo "make check             - Validate environment, dependencies, and run full test suite"
	@echo "make run-ui            - Launch the FastAPI research web server and UI on port 8000"
	@echo "make clean             - Clean build and cache files"

install:
	$(PYTHON) -m pip install -e .

generate-fixtures:
	$(PYTHON) -m neuroaxonnet.data.fixtures

test:
	$(PYTEST) tests/ -v

check:
	@echo "=== NeuroAxonNet System & Environment Audit ==="
	$(PYTHON) -c "import neuroaxonnet; print('NeuroAxonNet Version:', neuroaxonnet.__version__)"
	$(PYTHON) -c "import numpy, scipy, PIL, pydantic; print('Core Scientific Stack: OK')"
	@echo "=== Running Full Test Suite ==="
	$(PYTEST) tests/ -v --tb=short

run-ui:
	$(PYTHON) -m uvicorn neuroaxonnet.api.server:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
