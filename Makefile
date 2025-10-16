# Makefile for Tree Models Project
# Provides common development tasks and CI/CD commands

.PHONY: help install install-dev clean lint format test test-unit test-integration test-slow test-all coverage build docs serve-docs security check pre-commit setup-dev

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy
BANDIT := bandit
SAFETY := safety

# Project directories
SRC_DIRS := tree_models config src
TEST_DIR := tests
DOCS_DIR := docs

help: ## Show this help message
	@echo "Tree Models - Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup-dev     Setup development environment"
	@echo "  install       Install package in normal mode"
	@echo "  install-dev   Install package in development mode"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  format        Format code with black and isort"
	@echo "  lint          Run all linting checks"
	@echo "  check         Run all code quality checks"
	@echo "  security      Run security scans"
	@echo "  pre-commit    Run pre-commit hooks"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test          Run all tests (unit + integration, no slow)"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-slow     Run slow tests only"
	@echo "  test-all      Run all tests including slow ones"
	@echo "  coverage      Generate coverage report"
	@echo ""
	@echo "Build Commands:"
	@echo "  clean         Clean build artifacts"
	@echo "  build         Build package"
	@echo "  docs          Build documentation"
	@echo "  serve-docs    Serve documentation locally"
	@echo ""
	@echo "CI Commands:"
	@echo "  ci-lint       Run CI linting pipeline"
	@echo "  ci-test       Run CI testing pipeline"
	@echo "  ci-build      Run CI build pipeline"
	@echo "  ci-all        Run complete CI pipeline"

# Setup Commands
setup-dev: ## Setup development environment
	@echo "Setting up development environment..."
	$(PYTHON) -m venv venv
	@echo "Activate virtual environment with: source venv/bin/activate"
	@echo "Then run: make install-dev"

install: ## Install package in normal mode
	$(PIP) install -e .

install-dev: ## Install package in development mode with all dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,docs,all]"
	pre-commit install
	@echo "Development environment ready!"

# Code Quality Commands
format: ## Format code with black and isort
	@echo "Formatting code..."
	$(BLACK) $(SRC_DIRS) $(TEST_DIR)
	$(ISORT) $(SRC_DIRS) $(TEST_DIR)
	@echo "Code formatting complete!"

lint: ## Run all linting checks
	@echo "Running linting checks..."
	$(BLACK) --check --diff $(SRC_DIRS) $(TEST_DIR)
	$(ISORT) --check-only --diff $(SRC_DIRS) $(TEST_DIR)
	$(FLAKE8) $(SRC_DIRS) $(TEST_DIR)
	$(MYPY) $(SRC_DIRS) --ignore-missing-imports
	@echo "Linting complete!"

security: ## Run security scans
	@echo "Running security scans..."
	$(BANDIT) -r $(SRC_DIRS) -f json -o reports/bandit-report.json || true
	$(BANDIT) -r $(SRC_DIRS) -f txt
	$(SAFETY) check --json --output reports/safety-report.json || true
	$(SAFETY) check
	@echo "Security scan complete!"

check: lint security ## Run all code quality checks

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Testing Commands
test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	$(PYTEST) $(TEST_DIR) -v -m "unit and not slow" \
		--cov=$(SRC_DIRS) \
		--cov-report=term-missing \
		--cov-report=html:reports/htmlcov \
		--junit-xml=reports/junit-unit.xml

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	$(PYTEST) $(TEST_DIR) -v -m "integration and not slow" \
		--cov=$(SRC_DIRS) \
		--cov-append \
		--cov-report=term-missing \
		--junit-xml=reports/junit-integration.xml

test-slow: ## Run slow tests only
	@echo "Running slow tests..."
	$(PYTEST) $(TEST_DIR) -v -m "slow" \
		--cov=$(SRC_DIRS) \
		--cov-append \
		--cov-report=term-missing \
		--junit-xml=reports/junit-slow.xml \
		--timeout=3600

test: test-unit test-integration ## Run all tests (unit + integration, no slow)

test-all: test-unit test-integration test-slow ## Run all tests including slow ones

coverage: ## Generate detailed coverage report
	@echo "Generating coverage report..."
	$(PYTEST) $(TEST_DIR) \
		--cov=$(SRC_DIRS) \
		--cov-report=html:reports/htmlcov \
		--cov-report=xml:reports/coverage.xml \
		--cov-report=term-missing \
		--cov-fail-under=80
	@echo "Coverage report generated in reports/htmlcov/"

# Build Commands
clean: ## Clean build artifacts and cache files
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage .mypy_cache/
	rm -rf reports/htmlcov/ reports/*.xml reports/*.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"

build: clean ## Build package
	@echo "Building package..."
	mkdir -p reports
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*
	@echo "Package built successfully!"

# Documentation Commands
docs: ## Build documentation
	@echo "Building documentation..."
	mkdocs build --strict
	@echo "Documentation built in site/"

serve-docs: ## Serve documentation locally
	@echo "Serving documentation at http://localhost:8000"
	mkdocs serve

# CI Pipeline Commands
ci-lint: ## Run CI linting pipeline
	@echo "Running CI linting pipeline..."
	mkdir -p reports
	$(MAKE) format lint security
	@echo "CI linting pipeline complete!"

ci-test: ## Run CI testing pipeline
	@echo "Running CI testing pipeline..."
	mkdir -p reports
	$(MAKE) test coverage
	@echo "CI testing pipeline complete!"

ci-build: ## Run CI build pipeline
	@echo "Running CI build pipeline..."
	$(MAKE) build docs
	@echo "CI build pipeline complete!"

ci-all: ci-lint ci-test ci-build ## Run complete CI pipeline
	@echo "Complete CI pipeline finished successfully!"

# Development Helpers
install-pre-commit: ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

update-deps: ## Update development dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev,docs,all]"

run-example: ## Run example training script
	$(PYTHON) notebooks/train_xgb_baseline.py --create_sample --experiment_name "makefile_test"

demo-config: ## Run configuration system demo
	$(PYTHON) config/config_demo.py

# Quality Gates
quality-gate: lint test ## Quality gate for CI/CD
	@echo "All quality checks passed!"

release-check: quality-gate build ## Pre-release quality check
	@echo "Release quality check passed!"

# Docker Commands (if needed)
docker-build: ## Build Docker image
	docker build -t tree-models:latest .

docker-test: ## Run tests in Docker
	docker run --rm tree-models:latest make test

# Utility Commands
show-coverage: ## Show current test coverage
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIRS) --cov-report=term-missing --tb=no -q

list-todos: ## List TODO comments in code
	@echo "TODO items found:"
	@grep -r "TODO\|FIXME\|XXX" $(SRC_DIRS) $(TEST_DIR) --include="*.py" || echo "No TODOs found!"

count-lines: ## Count lines of code
	@echo "Lines of code:"
	@find $(SRC_DIRS) -name "*.py" | xargs wc -l | tail -1
	@echo "Lines of test code:"
	@find $(TEST_DIR) -name "*.py" | xargs wc -l | tail -1

# Performance Commands
profile-tests: ## Profile test execution time
	$(PYTEST) $(TEST_DIR) --durations=10 -v

benchmark: ## Run performance benchmarks
	@echo "Running benchmarks..."
	$(PYTHON) -m pytest benchmarks/ -v --benchmark-only

# Environment Management
create-env: ## Create virtual environment
	$(PYTHON) -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

freeze-deps: ## Freeze current dependencies
	$(PIP) freeze > requirements-frozen.txt
	@echo "Dependencies frozen to requirements-frozen.txt"

# Help and Info
info: ## Show project information
	@echo "Tree Models Project Information"
	@echo "==============================="
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Pip version: $(shell $(PIP) --version)"
	@echo "Project root: $(shell pwd)"
	@echo "Virtual env: $(VIRTUAL_ENV)"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'Not a git repo')"
	@echo "Git commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'Not a git repo')"

# Check if commands exist
check-commands: ## Check if required commands are available
	@echo "Checking required commands..."
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "Python not found!"; exit 1; }
	@command -v $(PIP) >/dev/null 2>&1 || { echo "Pip not found!"; exit 1; }
	@command -v git >/dev/null 2>&1 || { echo "Git not found!"; exit 1; }
	@echo "All required commands available!"

# Create directory structure
create-dirs: ## Create necessary directories
	mkdir -p reports logs data/raw data/processed models/saved outputs

# Full setup for new developers
full-setup: check-commands create-dirs setup-dev install-dev ## Complete setup for new developers
	@echo ""
	@echo "🎉 Full setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Activate virtual environment: source venv/bin/activate"
	@echo "2. Run tests: make test"
	@echo "3. Try the demo: make demo-config"
	@echo "4. Check code quality: make check"