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

## Ingestion Pipeline

To load and process the downloaded corpus:

1. **Run the ingestion pipeline**:
   ```bash
   uv run python -m src.ingestion.processor
   ```
   This reads all supported files from `corpus/raw_data/` and converts them into LangChain `Document` objects.

### Supported File Types

| Extension | Loader | Documents produced |
|-----------|--------|--------------------|
| `.html` | BSHTMLLoader | 1 per file |
| `.csv` | CSVLoader | 1 per row |
| `.pdf` | PyPDFLoader | 1 per page |
| `.xlsx`/`.xls` | UnstructuredExcelLoader | 1 per sheet |
| `.txt` | TextLoader | 1 per file |
| `.md` | UnstructuredMarkdownLoader | 1 per file |
| `.json` | JSONLoader | 1 per file |
| `.docx` | UnstructuredWordDocumentLoader | 1 per file |
| `.pptx` | UnstructuredPowerPointLoader | 1 per file |

## Project Structure

```text
rag_stress_testing/
├── corpus/
│   ├── data_sources.csv       # URLs and metadata for downloading
│   ├── metadata.csv           # Auto-generated download metadata
│   └── raw_data/              # Downloaded files (HTML, CSV, PDF, etc.)
├── src/
│   └── ingestion/
│       ├── __init__.py
│       ├── downloader.py      # Downloads files from data_sources.csv
│       ├── loaders.py         # File readers (HTML/CSV/PDF/etc.)
│       └── processor.py       # Main ingestion pipeline
├── tests/
│   └── test_downloader.py
├── docs/
│   ├── ADR.md                 # Architecture Decision Records
│   └── garth/
│       ├── TODO.md            # Project roadmap
│       └── phase_2.md         # Phase 2 design doc
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

## Roadmap

### Phase 1: setup
- [x] Initialize project structure with `uv`
- [x] Configure code quality tools (`ruff`)
- [x] Set up testing framework (`pytest`)
- [x] Establish CI/CD pipeline (GitHub Actions)
- [x] Define initial list of data sources (e.g., Federal Reserve DFAST/CCAR instructions, EBA guidelines)
- [ ] Ensure project works across different OS (Linux, Windows) with CI/CD

### Phase 2: get & process data
- [x] Build scrapers for public stress testing documentation (PDFs/HTML/TXT)
- [x] Move download_data.py to src/ingestion/downloader.py
- [x] Implement document loaders (HTML, CSV, PDF, Excel, text, Markdown, JSON, Word, PowerPoint)
- [x] Implement ingestion pipeline (processor.py)
- [ ] Implement text chunking strategy for long docs
- [ ] Implement metadata extraction (year, document type, source)
- [ ] Set up ChromaDB vector store
- [ ] Generate embeddings (HuggingFace `all-MiniLM-L6-v2`)
- [ ] Build ingestion script to upsert chunks into vector DB

### Phase 3: vector store
- [ ] Choose and set up vector database (ChromaDB recommended)
- [ ] Generate embeddings (HuggingFace `all-MiniLM-L6-v2`)
- [ ] Build ingestion script to upsert chunks into vector DB
- [ ] Deduplicate entries (idempotent upserts)
- [ ] Verify retrieval with test queries

### Phase 4: retrieval & generation
- [ ] Implement semantic search over vector store
- [ ] Integrate LLM for answer generation (RAG chain)
- [ ] Build a simple query interface (CLI or notebook)
- [ ] Add source citation to generated answers

### Phase 5: evaluation & hardening
- [ ] Build evaluation dataset (question/answer pairs)
- [ ] Measure retrieval quality (precision, recall, MRR)
- [ ] Measure generation quality (faithfulness, relevance)
- [ ] Tune chunk size, overlap, and top-k parameters
- [ ] Add logging and error handling
