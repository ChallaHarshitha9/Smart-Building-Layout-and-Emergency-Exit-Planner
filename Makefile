.PHONY: install test lint typecheck schemas check
install:
	pip install -e ".[dev]"
test:
	python -m pytest
lint:
	ruff check src tests scripts
typecheck:
	mypy
schemas:
	python -m smartbuilding.core.schemas --out schemas
check: lint typecheck test
