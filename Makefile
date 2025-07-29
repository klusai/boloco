.PHONY: help install install-dev test test-verbose lint format type-check clean build publish run-cli demo

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install basic dependencies
	poetry install

install-dev: ## Install all dependencies including dev tools
	poetry install --with dev

install-enhanced: ## Install with enhanced features (HuggingFace + Rich)
	poetry install --with dev --extras enhanced

install-full: ## Install with all features
	poetry install --with dev --extras full

test: ## Run tests
	poetry run pytest tests/ -v

test-verbose: ## Run tests with verbose output
	poetry run pytest tests/ -vv -s

test-coverage: ## Run tests with coverage report
	poetry run pytest tests/ --cov=boloco --cov-report=html --cov-report=term

lint: ## Run linting checks
	poetry run black --check .
	poetry run isort --check-only .
	poetry run flake8 .

format: ## Format code with black and isort
	poetry run black .
	poetry run isort .

type-check: ## Run type checking with mypy
	poetry run mypy boloco --ignore-missing-imports

quality: ## Run all quality checks (lint + type-check)
	@make lint
	@make type-check

clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	poetry build

publish: ## Publish to PyPI (requires authentication)
	poetry publish

publish-test: ## Publish to Test PyPI
	poetry publish --repository testpypi

run-cli: ## Run the CLI with example parameters
	poetry run boloco generate --max-tokens 5 --output-dir ./demo_output --format all

demo: ## Run a quick demo
	poetry run boloco generate --max-tokens 3 --output-dir ./quick_demo --format json
	@echo "Demo output saved to ./quick_demo/"

test-all: ## Run all tests and quality checks
	@make quality
	@make test

pre-commit-install: ## Install pre-commit hooks
	poetry run pre-commit install

pre-commit-run: ## Run pre-commit on all files
	poetry run pre-commit run --all-files

dev-setup: ## Complete development setup
	@make install-dev
	@make pre-commit-install
	@echo "Development environment ready!"

version: ## Show current version
	poetry version

shell: ## Open poetry shell
	poetry shell