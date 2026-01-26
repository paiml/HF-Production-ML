.PHONY: all transpile compile run clean lint test coverage check

all: run

# Python quality gates (ephemeral via uv)
lint:
	@uv run ruff check fibonacci.py
	@uv run ruff format --check fibonacci.py

test:
	@uv run pytest test_fibonacci.py -v

coverage:
	@uv run pytest test_fibonacci.py --cov=fibonacci --cov-report=term-missing --cov-fail-under=100

check: lint coverage

# Transpile pipeline
transpile:
	@depyler transpile fibonacci.py

compile: transpile
	@rustc fibonacci.rs -o fibonacci

run: compile
	@./fibonacci

clean:
	@rm -f fibonacci fibonacci.rs .coverage
	@rm -rf __pycache__ .pytest_cache .ruff_cache
