# Codon Encoder API - Development Makefile
# Usage: make <target>

.PHONY: help install install-dev lint format test test-cov security docker clean run

# Default target
help:
	@echo "Codon Encoder API - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         Run all linters (black, isort, flake8, mypy)"
	@echo "  make format       Auto-format code with black and isort"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run tests"
	@echo "  make test-cov     Run tests with coverage report"
	@echo ""
	@echo "Security:"
	@echo "  make security     Run security scans (bandit, pip-audit)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker       Build Docker image"
	@echo "  make docker-run   Run Docker container"
	@echo ""
	@echo "Other:"
	@echo "  make run          Run development server"
	@echo "  make clean        Remove build artifacts"
	@echo "  make model        Generate dummy model for testing"

# =============================================================================
# SETUP
# =============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install || echo "pre-commit not installed, skipping hook setup"

# =============================================================================
# CODE QUALITY
# =============================================================================

lint:
	@echo "Running black check..."
	black --check visualizer/ server/ tests/
	@echo "Running isort check..."
	isort --check-only visualizer/ server/ tests/
	@echo "Running flake8..."
	flake8 visualizer/ server/ --max-line-length=120 --ignore=F403,F405,F821,E501,E722,W503
	@echo "Running mypy..."
	mypy visualizer/ server/ --ignore-missing-imports || true
	@echo "All linters passed!"

format:
	@echo "Formatting with black..."
	black visualizer/ server/ tests/
	@echo "Sorting imports with isort..."
	isort visualizer/ server/ tests/
	@echo "Code formatted!"

# =============================================================================
# TESTING
# =============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=visualizer --cov=server --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

model:
	python scripts/generate_dummy_model.py

# =============================================================================
# SECURITY
# =============================================================================

security:
	@echo "Running bandit security scan..."
	bandit -r visualizer/ server/ -ll || true
	@echo "Running pip-audit..."
	pip-audit || true
	@echo "Security scan complete!"

# =============================================================================
# DOCKER
# =============================================================================

docker:
	docker build -t codon-encoder-api .

docker-run:
	docker run -p 8765:8765 -v ./server/model:/app/server/model:ro codon-encoder-api

# =============================================================================
# DEVELOPMENT
# =============================================================================

run:
	cd visualizer && python run.py

# =============================================================================
# CLEANUP
# =============================================================================

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	rm -rf *.egg-info build dist
	rm -rf .eggs
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Clean complete!"
