.PHONY: help install test test-cov lint format ruff typecheck check dev run run-dev run-prod docker-up docker-down setup-model clean all

PYTHON ?= python3
UV ?= uv

help:
	@echo "================================================================="
	@echo "  NIQ Object Counter & Prediction Platform - Task Runner"
	@echo "================================================================="
	@echo "Quick Run Commands:"
	@echo "  make dev          - Quickly start local API server (in-memory mode)"
	@echo "  make run          - Alias for 'make dev'"
	@echo "  make check        - Run all quality checks (lint + typecheck + tests)"
	@echo "  make test         - Run pytest test suite (53 tests)"
	@echo "  make typecheck    - Run mypy static type checking"
	@echo ""
	@echo "Code Quality & Formatting Commands:"
	@echo "  make lint         - Run ruff linter check"
	@echo "  make format       - Auto-format code and fix import sorting with ruff"
	@echo "  make ruff         - Run ruff formatter and linter auto-fix"
	@echo "  make clean        - Remove all cache files (__pycache__, .pytest_cache, .ruff_cache, .mypy_cache, .coverage)"
	@echo ""
	@echo "Other Commands:"
	@echo "  make install      - Install all dependencies using uv"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make run-prod     - Run Flask app pointing to production services"
	@echo "  make docker-up    - Build and start full stack via Docker Compose"
	@echo "  make docker-down  - Stop all Docker Compose services"
	@echo "  make setup-model  - Download and extract SSD MobileNet v2 model"
	@echo "  make all          - Install dependencies, check types, and run tests"

# Quick aliases
dev: run-dev
run: run-dev

check: lint typecheck test

all: install check

install:
	@command -v $(UV) >/dev/null 2>&1 && $(UV) sync --all-extras || pip install -r requirements.txt

test:
	@command -v $(UV) >/dev/null 2>&1 && $(UV) run pytest || pytest

test-cov:
	@command -v $(UV) >/dev/null 2>&1 && $(UV) run pytest --cov=counter tests/ || pytest --cov=counter tests/

lint:
	@command -v $(UV) >/dev/null 2>&1 && $(UV) run ruff check counter tests || ruff check counter tests

typecheck:
	@command -v $(UV) >/dev/null 2>&1 && $(UV) run mypy counter || mypy counter

format:
	@command -v $(UV) >/dev/null 2>&1 && $(UV) run ruff format counter tests && $(UV) run ruff check --fix counter tests || (ruff format counter tests && ruff check --fix counter tests)

ruff: format lint

run-dev:
	@command -v $(UV) >/dev/null 2>&1 && $(UV) run python -m counter.entrypoints.webapp || python -m counter.entrypoints.webapp

run-prod:
	ENV=prod DB_TYPE=postgres DETECTOR_TYPE=tfs @command -v $(UV) >/dev/null 2>&1 && $(UV) run python -m counter.entrypoints.webapp || ENV=prod python -m counter.entrypoints.webapp

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v

setup-model:
	mkdir -p tmp/model/ssd_mobilenet_v2/1
	curl -L -o tmp/model.tar.gz http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
	tar -xzvf tmp/model.tar.gz -C tmp/model
	mv tmp/model/ssd_mobilenet_v2_coco_2018_03_29/saved_model/saved_model.pb tmp/model/ssd_mobilenet_v2/1/
	rm -rf tmp/model.tar.gz tmp/model/ssd_mobilenet_v2_coco_2018_03_29
	chmod -R 777 tmp/model || true

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage .coverage.* coverage.xml tmp/debug/*.jpg *.egg-info build dist
	find counter tests -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find counter tests -type f -name "*.pyc" -delete 2>/dev/null || true
	find counter tests -type f -name "*.pyo" -delete 2>/dev/null || true
