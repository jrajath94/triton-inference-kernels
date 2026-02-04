.PHONY: install test bench lint clean run

install:
	pip install -e ".[dev,bench]"

test:
	pytest tests/ -v --tb=short --cov=src/triton_kernels --cov-report=term-missing --cov-report=xml

bench:
	python benchmarks/bench_kernels.py

run:
	python examples/quickstart.py

lint:
	ruff check src/ tests/ benchmarks/ examples/
	mypy src/ --ignore-missing-imports

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage
	rm -rf dist/ build/ *.egg-info/
