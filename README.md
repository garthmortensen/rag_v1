# RAG Stress Testing

![CI](https://github.com/garthmortensen/rag_stress_testing/actions/workflows/ci.yml/badge.svg)

This repository contains a Retrieval-Augmented Generation (RAG) for public Stress Testing docs. The goal is to build a RAG using open-source tools to ingest, index, and query publically available stress testing documentation.

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast python dependency management.

### Installation

1. **Install uv**:

https://github.com/astral-sh/uv?tab=readme-ov-file#installation

2. **Sync dependencies**:
   ```bash
   uv sync
   ```
   This will create a virtual environment in `.venv` and install the dependencies specified in `pyproject.toml`.

3. **Activate environment**:
   ```bash
   source .venv/bin/activate
   ```

4. **Add new packages**:
   ```bash
   uv add <package_name>
   ```

### Pre-commit Checks

Before committing changes, ensure your code is formatted and passes tests:

```bash
# Format code (whitespace, layout, standardizes style)
uv run ruff format .

# Run linting (fixes logic errors, unused imports, bad practices)
uv run ruff check . --fix

# Run tests
uv run pytest
```

### Workflow & Versioning

This project uses [Commitizen](https://commitizen-tools.github.io/commitizen/) to standardize commit messages, automate versioning, and generate changelogs.

#### 1. Committing

Use the following command instead of `git commit` to launch the interactive wizard:

```bash
uv run cz commit
```

This creates structured commit messages (e.g., `feat: ...`, `fix: ...`) required for semantic versioning.

#### 2. Collaboration (Branches)

When working on features or fixes:
1. Create a branch (e.g., `feat/data-ingestion`, `fix/broken-link`).
2. Make your edits.
3. Commit using `uv run cz commit`.
4. Push and open a Pull Request against `main`.

#### 3. Releasing & Changelog

To bump the version, update `CHANGELOG.md`, and create a tag based on commit history:

```bash
uv run cz bump
```
