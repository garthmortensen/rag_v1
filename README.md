# RAG Stress Testing

![CI](https://github.com/garthmortensen/rag_stress_testing/actions/workflows/ci.yml/badge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![python](https://img.shields.io/badge/Python-3.13+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![tested with pytest](https://img.shields.io/badge/tested%20with-pytest-46M3D2.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/en/latest/)

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

## Data Acquisition

To download the stress testing corpus:

1. **Run the downloader**:
   ```bash
   uv run src/ingestion/downloader.py
   ```
   This script reads from `corpus/data_sources.csv` and downloads files to `corpus/raw_data/`.
   A summary log will be saved to `corpus/download.log`.
