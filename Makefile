.PHONY: help clean check autoformat test
.DEFAULT_GOAL := help

help:
	@echo "make check       Run black and ruff without modifying files"
	@echo "make autoformat  Format with black and apply safe ruff fixes"
	@echo "make test        Run the CPU-only unit tests"
	@echo "make clean       Remove Python caches"

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

check:
	black --check .
	ruff check --output-format=full .

autoformat:
	black .
	ruff check --fix-only --show-fixes .

test:
	pytest -q tests
