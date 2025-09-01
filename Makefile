.PHONY: help install dev test lint format clean experiments demo

help: ## Show this help message
	@echo "KV Cache Profiler - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

dev: ## Install development dependencies
	uv sync --dev

test: ## Run tests
	uv run pytest

lint: ## Run linting
	uv run ruff check .
	uv run mypy profiler/

format: ## Format code
	uv run ruff format .
	uv run ruff check . --fix

clean: ## Clean up generated files
	rm -rf data/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Experiment shortcuts
exp1: ## Run concurrency experiment
	uv run experiments/exp1_concurrency.py --model facebook/opt-1.3b --concurrency 1 2 4 --verbose

exp2: ## Run sequence length experiment
	uv run experiments/exp2_seqlen.py --model facebook/opt-1.3b --seqlens 128 512 1024 --verbose

exp3: ## Run mixed batch experiment
	uv run experiments/exp3_mixed_batch.py --model facebook/opt-1.3b --short 128 --long 1024 --verbose

experiments: exp1 exp2 exp3 ## Run all experiments

# Llama 3.1 experiments
llama-exp1: ## Run concurrency experiment with Llama 3.1 8B
	uv run experiments/exp1_concurrency.py --model meta-llama/Llama-3.1-8B-Instruct --concurrency 1 2 --verbose

llama-exp2: ## Run sequence length experiment with Llama 3.1 8B
	uv run experiments/exp2_seqlen.py --model meta-llama/Llama-3.1-8B-Instruct --seqlens 128 512 1024 --verbose

llama-bench: ## Run basic benchmark with Llama 3.1 8B
	uv run profiler/cli.py bench --model meta-llama/Llama-3.1-8B-Instruct --concurrency 1 --verbose

# CLI shortcuts
bench: ## Run basic benchmark
	uv run profiler/cli.py bench --model facebook/opt-1.3b --concurrency 2 --verbose

info: ## Show model info
	uv run profiler/cli.py info facebook/opt-1.3b

env: ## Show environment info
	uv run profiler/cli.py env

demo: bench ## Run demo benchmark

# Development
check: lint test ## Run all checks

setup: install ## Initial setup
	@echo "✅ KV Cache Profiler setup complete!"
	@echo ""
	@echo "Try these commands:"
	@echo "  make demo     # Run a quick benchmark"
	@echo "  make exp1     # Test concurrency scaling"
	@echo "  make env      # Check your environment"
	@echo ""
