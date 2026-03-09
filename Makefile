# =============================================================================
# MLOps-Sentinel — Makefile
# Targets: install | train | serve | test | lint | format | docker-build |
#          docker-up | docker-down | clean
# =============================================================================

.DEFAULT_GOAL := help
SHELL         := /bin/bash

# ---------- Configuration ----------------------------------------------------
PYTHON        ?= python
PIP           ?= pip
IMAGE_NAME    ?= mlops-sentinel
IMAGE_TAG     ?= latest
PORT          ?= 8000
N_SAMPLES     ?= 5000
MODEL_DIR     ?= models

# ---------- Colours (no-op on Windows without ANSI support) ------------------
BOLD  := \033[1m
RESET := \033[0m
GREEN := \033[32m
CYAN  := \033[36m

# =============================================================================
.PHONY: help install train serve test lint format \
        docker-build docker-up docker-down clean

help:  ## Show this help message
	@echo ""
	@echo "$(BOLD)MLOps-Sentinel$(RESET) — available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
install:  ## Install Python dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(RESET)"

# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
train:  ## Train the model and save artefact to $(MODEL_DIR)/
	$(PYTHON) -m src.training.train \
		--output-dir $(MODEL_DIR) \
		--n-samples $(N_SAMPLES)
	@echo "$(GREEN)✓ Model trained and saved to $(MODEL_DIR)/$(RESET)"

# ---------------------------------------------------------------------------
# API server (local development)
# ---------------------------------------------------------------------------
serve:  ## Start the FastAPI server locally (hot-reload enabled)
	uvicorn src.api.main:app \
		--host 0.0.0.0 \
		--port $(PORT) \
		--reload \
		--log-level info

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test:  ## Run the full Pytest suite
	pytest tests/ \
		--tb=short \
		--asyncio-mode=auto \
		-v

test-coverage:  ## Run tests with HTML coverage report
	pytest tests/ \
		--tb=short \
		--asyncio-mode=auto \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		-v
	@echo "$(GREEN)✓ Coverage report → htmlcov/index.html$(RESET)"

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
lint:  ## Run flake8 linter
	flake8 src/ tests/ \
		--max-line-length=100 \
		--extend-ignore=E203,W503 \
		--count \
		--statistics

format:  ## Auto-format code with black
	black src/ tests/

format-check:  ## Check formatting without modifying files
	black --check --diff src/ tests/

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build:  ## Build the Docker image
	docker build \
		--target runtime \
		--tag $(IMAGE_NAME):$(IMAGE_TAG) \
		.
	@echo "$(GREEN)✓ Image $(IMAGE_NAME):$(IMAGE_TAG) built$(RESET)"

docker-up:  ## Start the full stack (API + Prometheus) via docker-compose
	docker compose up --build -d
	@echo "$(GREEN)✓ Stack running — API: http://localhost:$(PORT) | Prometheus: http://localhost:9090$(RESET)"

docker-down:  ## Tear down the docker-compose stack
	docker compose down --remove-orphans
	@echo "$(GREEN)✓ Stack stopped$(RESET)"

docker-logs:  ## Tail logs from the API container
	docker compose logs -f api

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
clean:  ## Remove compiled artefacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
	@echo "$(GREEN)✓ Cleaned$(RESET)"
