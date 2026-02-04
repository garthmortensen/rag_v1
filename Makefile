.PHONY: install format lint test check commit

install:
	uv sync

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix

test:
	uv run pytest

check: format lint test
