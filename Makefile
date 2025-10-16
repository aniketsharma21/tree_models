.PHONY: help setup install install-dev test test-unit test-integration test-performance test-coverage lint format clean build docs serve-docs

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
PROJECT_NAME := tree_models
VERSION := $(shell grep '^version' pyproject.toml | cut -d'"' -f2)

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "$(GREEN)Tree Models Makefile$(NC)"
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Setup and Installation
setup: ## Setup development environment
	@echo "$(GREEN)Setting up development environment...$(NC)"
	$(PYTHON) -m venv venv
	@echo "$(YELLOW)Activate virtual environment with: source venv/bin/activate$(NC)"
	@echo "$(YELLOW)Then run: make install-dev$(NC)"

install: ## Install package in production mode
	@echo "$(GREEN)Installing $(PROJECT_NAME) in production mode...$(NC)"
	$(PIP) install -e .

install-dev: ## Install package in development mode with all dependencies
	@echo "$(GREEN)Installing $(PROJECT_NAME) in development mode...$(NC)"
	$(PIP) install -e ".[dev,test,docs,ml]"
	$(PIP) install pre-commit
	pre-commit install

install-minimal: ## Install package with minimal dependencies
	@echo "$(GREEN)Installing $(PROJECT_NAME) with minimal dependencies...$(NC)"
	$(PIP) install -e ".[minimal]"

# Testing
test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(NC)"
	$(PYTEST) tests/ -v --tb=short

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTEST) tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(PYTEST) tests/integration/ -v

test-performance: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(NC)"
	$(PYTEST) tests/performance/ -v -s --tb=short

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(PYTEST) tests/ --cov=$(PROJECT_NAME) --cov-report=html --cov-report=term-missing --cov-fail-under=80
	@echo "$(YELLOW)Coverage report generated in htmlcov/index.html$(NC)"

test-fast: ## Run tests excluding slow ones
	@echo "$(GREEN)Running fast tests only...$(NC)"
	$(PYTEST) tests/ -v -m "not slow" --tb=short

test-ml: ## Run tests requiring ML libraries (XGBoost, LightGBM, etc.)
	@echo "$(GREEN)Running ML library tests...$(NC)"
	$(PYTEST) tests/ -v -m "requires_ml_libs"

# Code Quality
lint: ## Run linting checks
	@echo "$(GREEN)Running linting checks...$(NC)"
	flake8 $(PROJECT_NAME)/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 $(PROJECT_NAME)/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics
	mypy $(PROJECT_NAME)/ --ignore-missing-imports

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	black $(PROJECT_NAME)/ tests/ --line-length=120
	isort $(PROJECT_NAME)/ tests/ --profile black --line-length=120

format-check: ## Check code formatting without making changes
	@echo "$(GREEN)Checking code formatting...$(NC)"
	black $(PROJECT_NAME)/ tests/ --check --line-length=120
	isort $(PROJECT_NAME)/ tests/ --check-only --profile black --line-length=120

# Security and Safety
security: ## Run security checks
	@echo "$(GREEN)Running security checks...$(NC)"
	safety check
	bandit -r $(PROJECT_NAME)/ -f json -o security-report.json || true
	@echo "$(YELLOW)Security report generated in security-report.json$(NC)"

# Documentation
docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	cd docs && make html
	@echo "$(YELLOW)Documentation built in docs/_build/html/$(NC)"

docs-clean: ## Clean documentation build
	@echo "$(GREEN)Cleaning documentation...$(NC)"
	cd docs && make clean

serve-docs: docs ## Build and serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8000$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# Building and Distribution
build: clean ## Build source and wheel distributions
	@echo "$(GREEN)Building $(PROJECT_NAME) v$(VERSION)...$(NC)"
	$(PYTHON) -m build
	@echo "$(YELLOW)Built distributions in dist/$(NC)"

build-wheel: ## Build wheel distribution only
	@echo "$(GREEN)Building wheel for $(PROJECT_NAME)...$(NC)"
	$(PYTHON) -m build --wheel

build-sdist: ## Build source distribution only
	@echo "$(GREEN)Building source distribution for $(PROJECT_NAME)...$(NC)"
	$(PYTHON) -m build --sdist

# Publishing
publish-test: build ## Publish to Test PyPI
	@echo "$(GREEN)Publishing to Test PyPI...$(NC)"
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
	@echo "$(YELLOW)Uploaded to https://test.pypi.org/project/$(PROJECT_NAME)/$(NC)"

publish: build ## Publish to PyPI
	@echo "$(RED)Publishing to PyPI...$(NC)"
	@read -p "Are you sure you want to publish v$(VERSION) to PyPI? [y/N]: " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		twine upload dist/*; \
		echo "$(GREEN)Published $(PROJECT_NAME) v$(VERSION) to PyPI$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled publication$(NC)"; \
	fi

# Development Utilities
clean: ## Clean build artifacts and temporary files
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean ## Clean everything including virtual environment
	@echo "$(GREEN)Cleaning everything...$(NC)"
	rm -rf venv/
	rm -rf .venv/

setup-hooks: ## Setup pre-commit hooks
	@echo "$(GREEN)Setting up pre-commit hooks...$(NC)"
	pre-commit install
	pre-commit install --hook-type commit-msg

update-deps: ## Update all dependencies
	@echo "$(GREEN)Updating dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev,test,docs,ml]"

# Docker support
docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t $(PROJECT_NAME):$(VERSION) .
	docker build -t $(PROJECT_NAME):latest .

docker-test: ## Run tests in Docker
	@echo "$(GREEN)Running tests in Docker...$(NC)"
	docker run --rm $(PROJECT_NAME):latest make test

docker-run: ## Run interactive shell in Docker
	@echo "$(GREEN)Starting Docker container...$(NC)"
	docker run --rm -it $(PROJECT_NAME):latest /bin/bash

# Performance and Profiling
profile: ## Run performance profiling
	@echo "$(GREEN)Running performance profiling...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats -m pytest tests/performance/
	@echo "$(YELLOW)Profile saved to profile.stats$(NC)"

benchmark: ## Run benchmarks and save results
	@echo "$(GREEN)Running benchmarks...$(NC)"
	$(PYTEST) tests/performance/ --benchmark-only --benchmark-json=benchmark.json
	@echo "$(YELLOW)Benchmark results saved to benchmark.json$(NC)"

# Examples and Demos
examples: ## Run example notebooks/scripts
	@echo "$(GREEN)Running examples...$(NC)"
	$(PYTHON) examples/quickstart/basic_usage.py
	$(PYTHON) examples/advanced/complete_pipeline.py

demo: ## Run demo with sample data
	@echo "$(GREEN)Running demo...$(NC)"
	$(PYTHON) -c "from tree_models.examples import run_demo; run_demo()"

# Release Management
version: ## Show current version
	@echo "$(GREEN)Current version: $(VERSION)$(NC)"

bump-patch: ## Bump patch version (1.0.0 -> 1.0.1)
	@echo "$(GREEN)Bumping patch version...$(NC)"
	bump2version patch

bump-minor: ## Bump minor version (1.0.0 -> 1.1.0)
	@echo "$(GREEN)Bumping minor version...$(NC)"
	bump2version minor

bump-major: ## Bump major version (1.0.0 -> 2.0.0)
	@echo "$(GREEN)Bumping major version...$(NC)"
	bump2version major

# Environment Management
env-info: ## Show environment information
	@echo "$(GREEN)Environment Information:$(NC)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Project: $(PROJECT_NAME) v$(VERSION)"
	@echo "Current directory: $(shell pwd)"
	@echo "Virtual env: $(VIRTUAL_ENV)"

deps-info: ## Show dependency information
	@echo "$(GREEN)Dependency Information:$(NC)"
	$(PIP) list --format=columns

# CI/CD Helpers
ci-install: ## Install dependencies for CI/CD
	@echo "$(GREEN)Installing CI dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[test,dev]"

ci-test: ## Run tests in CI environment
	@echo "$(GREEN)Running CI tests...$(NC)"
	$(PYTEST) tests/ --cov=$(PROJECT_NAME) --cov-report=xml --junitxml=junit.xml

ci-lint: ## Run linting in CI environment
	@echo "$(GREEN)Running CI linting...$(NC)"
	flake8 $(PROJECT_NAME)/ tests/ --format=junit-xml --output-file=flake8.xml
	mypy $(PROJECT_NAME)/ --junit-xml mypy.xml

# Quick Development Commands
quick-test: ## Quick test run (fast tests only with minimal output)
	@$(PYTEST) tests/unit/ -q --tb=no

quick-check: ## Quick code quality check
	@echo "$(GREEN)Quick check...$(NC)"
	@flake8 $(PROJECT_NAME)/ --count --select=E9,F63,F7,F82 --show-source --statistics
	@$(PYTEST) tests/unit/ -q --tb=no
	@echo "$(GREEN)✓ Quick check passed$(NC)"

dev: ## Start development mode (install + quick test)
	@echo "$(GREEN)Starting development mode...$(NC)"
	@$(MAKE) install-dev
	@$(MAKE) quick-check
	@echo "$(GREEN)✓ Development environment ready$(NC)"

# All-in-one commands
full-check: ## Run complete code quality and testing suite
	@echo "$(GREEN)Running full quality check...$(NC)"
	@$(MAKE) format-check
	@$(MAKE) lint
	@$(MAKE) security
	@$(MAKE) test-coverage
	@echo "$(GREEN)✓ All checks passed$(NC)"

release-check: ## Pre-release validation
	@echo "$(GREEN)Running pre-release checks...$(NC)"
	@$(MAKE) clean
	@$(MAKE) full-check
	@$(MAKE) build
	@echo "$(GREEN)✓ Ready for release$(NC)"