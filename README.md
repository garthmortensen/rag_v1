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

## Configuration

Pipeline settings are controlled by `config.txt` at the project root:

```text
chunk_size       = 6000
chunk_overlap    = 200
collection_name  = stress_test_docs_6k
llm_provider     = ollama
llm_model        = llama3.2:3b
```

| Key | What it does | Default |
|-----|-------------|--------|
| `chunk_size` | Characters per chunk | `1000` |
| `chunk_overlap` | Overlap between consecutive chunks | `100` |
| `collection_name` | ChromaDB collection to write/read | `stress_test_docs_1k` |
| `llm_provider` | LLM backend: `ollama`, `openai`, `anthropic`, `google` | `ollama` |
| `llm_model` | Model name/tag for the chosen provider | `llama3.2:3b` |

### Switching between embedding sets

Each `collection_name` is a separate ChromaDB collection stored in `corpus/vector_db/`. Changing the name does **not** delete existing collections — they coexist side-by-side.

1. Edit `config.txt` with the desired settings.
2. Re-run the ingestion pipeline to build (or rebuild) that collection:
   ```bash
   uv run python -m src.ingestion.processor
   ```
3. Queries automatically use whichever `collection_name` is set in `config.txt`.

To switch back, just change `collection_name` and query — no re-ingestion needed if the collection already exists.

## Data Acquisition

To download the stress testing corpus:

1. **Run the downloader**:
   ```bash
   uv run src/ingestion/downloader.py
   ```
   This script reads from `corpus/data_sources.csv` and downloads files to `corpus/raw_data/`.
   A summary log will be saved to `corpus/download.log`.

## Ingestion Pipeline

The ingestion pipeline loads downloaded files, chunks them, embeds the chunks, and upserts everything into ChromaDB.

### Running ingestion

1. **Make sure files are downloaded** (see [Data Acquisition](#data-acquisition) above).
2. **Review `config.txt`** — set your desired `chunk_size`, `chunk_overlap`, and `collection_name`.
3. **Run the pipeline**:
   ```bash
   uv run python -m src.ingestion.processor
   ```

The pipeline will:
- Load all supported files from `corpus/raw_data/`
- Split them into chunks based on `chunk_size` / `chunk_overlap`
- Embed each chunk using HuggingFace sentence-transformers
- Enrich metadata from `corpus/metadata.csv` (doc_id, title, author, category, …)
- Upsert embeddings into the ChromaDB collection named in `config.txt`

### Re-ingesting with different settings

To create a second set of embeddings (e.g. larger chunks), just edit `config.txt` and run the pipeline again:

```bash
# Edit config.txt:
#   chunk_size      = 8000
#   collection_name = stress_test_docs_8k

uv run python -m src.ingestion.processor
```

Each `collection_name` produces a separate collection — existing ones are not deleted.

### Inspecting collections

To see which collections exist and how many chunks each contains:

```bash
python corpus/inspect_db.py
```

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

## Retrieval

Query the vector store for relevant document chunks. This is just retrieval — it shows which chunks matched, not a synthesized answer:

```bash
# Basic semantic search (top-5 results)
uv run python -m src.retrieval.query "What is the peak unemployment rate?"

# More results with a metadata filter
uv run python -m src.retrieval.query "CET1 capital ratio" --top-k 10 --filter source_type=pdf
```

## Answer Generation (RAG)

Add the `--answer` flag for a ChatGPT-like experience — retrieves the top-k chunks, builds a grounded prompt, and sends it to the configured LLM. Answers include source citations.

The generation pipeline uses **LangChain Expression Language (LCEL)** to compose a declarative chain:

```text
RAG_PROMPT | llm | StrOutputParser()
```

- `RAG_PROMPT` — a `ChatPromptTemplate` with system and human messages
- `llm` — the provider-specific chat model returned by `get_llm()`
- `StrOutputParser()` — extracts the plain-text response

LCEL enables native `.stream()` support — the Streamlit UI renders tokens as they arrive.

### LLM Providers

The LLM backend is configurable via `config.txt`:

| Provider | Package (install with `uv add`) | Example models | API key env var |
|----------|--------------------------------|---------------|----------------|
| **ollama** (default) | `langchain-ollama` (included) | `llama3.2:3b`, `phi3` | — (local) |
| **openai** | `langchain-openai` | `gpt-4o-mini`, `gpt-4o` | `OPENAI_API_KEY` |
| **anthropic** | `langchain-anthropic` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| **google** | `langchain-google-genai` | `gemini-2.0-flash` | `GOOGLE_API_KEY` |

To use a remote provider:

1. **Copy the env template** and add your API key:
   ```bash
   cp .env.example .env
   # edit .env — uncomment and fill in the key you need
   ```
2. **Install the provider package**:
   ```bash
   uv add langchain-openai   # or langchain-anthropic, langchain-google-genai
   ```
3. **Set the provider in `config.txt`**:
   ```text
   llm_provider = openai
   llm_model    = gpt-4o-mini
   ```

### Using Ollama (local, default)

1. **Install Ollama** (one-time):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3.2:3b
   ```

2. **Ask a question**:
   ```bash
   uv run python -m src.retrieval.query "What is CCAR?" --answer
   uv run python -m src.retrieval.query "capital ratios" --answer --model phi3
   ```

### CLI flags

| Flag | Description |
|------|-------------|
| `--answer` | Generate an LLM answer from retrieved chunks |
| `--model MODEL` | Override the model from `config.txt` |
| `--provider PROVIDER` | Override the provider from `config.txt` |

## Evaluation (ragas)

The project includes a built-in evaluation pipeline powered by [ragas](https://docs.ragas.io/) to measure RAG quality across four metrics:

| Metric | What it measures |
|--------|-----------------|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Response Relevancy** | Does the answer address the question? |
| **LLM Context Recall** | Did retrieval surface the information needed for the ground-truth answer? |
| **Factual Correctness** | Does the answer match the expected ground-truth? |

### Running evaluation

```bash
# Default: uses config.txt provider/model, top-k=5
uv run python -m src.evaluation.evaluate

# Override settings
uv run python -m src.evaluation.evaluate --provider openai --model gpt-4o-mini --top-k 10

# Export results to CSV
uv run python -m src.evaluation.evaluate --out results.csv
```

The evaluation dataset lives in `src/evaluation/dataset.py` — 8 curated question/ground-truth pairs covering stress testing topics (unemployment scenarios, capital ratios, regulatory requirements, etc.).

## LangSmith Tracing

[LangSmith](https://smith.langchain.com/) provides end-to-end observability for every LLM call, chain invocation, and retrieval step. Because the pipeline uses LCEL, tracing is **zero-config** — just set two environment variables:

```bash
cp .env.example .env
# In .env, uncomment and set:
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=ls-...
```

Once enabled, every `generate_answer()`, `stream_answer()`, and `ask()` call is automatically traced in the LangSmith dashboard with full input/output visibility, latency breakdowns, and token counts. No code changes required — `langsmith` is already installed as a `langchain-core` dependency.

## Web UI (Streamlit)

A multi-pane browser interface for interactive querying:

```bash
uv run streamlit run app.py
```

**Sidebar panes:**
- **config.txt** — shows active pipeline configuration
- **Collection picker** — switch between embedding collections (1k, 8k, etc.) with live chunk counts
- **Top-K slider** — control how many chunks are retrieved
- **Source types & categories** — filter by file type, source org, and category
- **LLM Provider & Model** — choose provider (ollama, openai, anthropic, google) and model
- **Temperature** — sampling temperature slider

**Main pane:**
- Chat-style interface with message history
- **Token-by-token streaming** — answers render word-by-word via `st.write_stream()` and the LCEL chain's `.stream()` method
- LLM-generated answers with source citations
- **Copy Q&A** — one-click button to copy the question and answer to clipboard
- Expandable retrieved sources showing rank, category, distance, and chunk text

## Project Structure

```text
rag_stress_testing/
├── app.py                     # Streamlit web UI
├── config.txt                 # Pipeline settings (chunk size, collection, LLM provider)
├── .env.example               # API key template (copy to .env)
├── corpus/
│   ├── data_sources.csv       # URLs and metadata for downloading
│   ├── metadata.csv           # Auto-generated download metadata
│   ├── raw_data/              # Downloaded files (HTML, CSV, PDF, etc.)
│   └── vector_db/             # ChromaDB persistent storage (HNSW index)
├── src/
│   ├── config.py              # Reads config.txt + .env (via python-dotenv)
│   ├── utils.py               # RAM-aware embedding model selection
│   ├── ingestion/
│   │   ├── downloader.py      # Downloads files from data_sources.csv
│   │   ├── loaders.py         # File readers (HTML/CSV/PDF/etc.)
│   │   └── processor.py       # Main ingestion pipeline
│   ├── embedding/
│   │   └── model.py           # Embed via HuggingFace, upsert to ChromaDB
│   ├── retrieval/
│   │   ├── query.py           # Semantic search: embed query → ANN lookup
│   │   └── query_logger.py    # Log query sessions to logs/
│   ├── generation/
│   │   └── llm.py             # LCEL chain: RAG_PROMPT | llm | StrOutputParser()
│   └── evaluation/
│       ├── dataset.py         # Curated Q&A pairs for ragas evaluation
│       └── evaluate.py        # CLI runner: ragas metrics + CSV export
├── tests/
│   ├── test_config.py
│   ├── test_downloader.py
│   ├── test_embedding.py
│   ├── test_evaluation.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   ├── test_query_logger.py
│   └── test_utils.py
├── docs/
│   └── architecture.md        # ADRs, pipeline diagrams, design details
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
- [x] Implement text chunking strategy for long docs
- [x] Implement metadata extraction (year, document type, source)
- [x] Set up ChromaDB vector store
- [x] Generate embeddings (HuggingFace `all-MiniLM-L6-v2`)
- [x] Build ingestion script to upsert chunks into vector DB

### Phase 3: vector store
- [x] Choose and set up vector database (ChromaDB recommended)
- [x] Generate embeddings (HuggingFace `all-MiniLM-L6-v2`)
- [x] Build ingestion script to upsert chunks into vector DB
- [x] Deduplicate entries (idempotent upserts)
- [x] Verify retrieval with test queries

### Phase 4: retrieval & generation
- [x] Implement semantic search over vector store (`src/retrieval/query.py`)
- [x] Integrate LLM for answer generation — Ollama via `langchain-ollama` (`src/generation/llm.py`)
- [x] Multi-provider LLM support (Ollama, OpenAI, Anthropic, Google) via `config.txt` + `.env`
- [x] Build a simple query interface (CLI with `--answer` flag)
- [x] Add source citation to generated answers

### Phase 5: evaluation & hardening
- [x] Rewrite generation pipeline to LCEL chain (`RAG_PROMPT | llm | StrOutputParser()`)
- [x] Token-by-token streaming in Streamlit via `st.write_stream()` + LCEL `.stream()`
- [x] Build evaluation dataset (8 curated Q&A pairs in `src/evaluation/dataset.py`)
- [x] Measure generation quality via **ragas** (Faithfulness, Response Relevancy, Context Recall, Factual Correctness)
- [x] Add LangSmith tracing support (zero-config via `LANGCHAIN_TRACING_V2` env var)
- [x] Document-based retrieval API (`retrieve_as_documents()` returning `list[Document]`)
- [ ] Measure retrieval quality (precision, recall, MRR)
- [ ] Tune chunk size, overlap, and top-k parameters
- [ ] Add logging and error handling

## Tech Stack

### Orchestration Framework

| Tool | Pros | Cons |
|------|------|------|
| **LangChain** | Massive ecosystem, standard interface, rich document loaders/splitters | Bloated/over-abstracted, hard to debug, frequent breaking changes |
| **Haystack** | Explicit pipeline design (DAGs), Pythonic/readable, production-ready | Smaller ecosystem than LangChain |
| **Pure Python** | Maximum control, zero dependency bloat, easy debugging | Reinventing wheels, higher maintenance code |

**Decision**: **LangChain** — its `Community` document loaders and text splitters save days of work.

### Vector Database

| Tool | Pros | Cons |
|------|------|------|
| **ChromaDB** | Open-source, in-process (no Docker), Python-native, simple storage/indexing | Newer project, SQLite/ClickHouse wrapper, scaling limits |
| **FAISS** | Gold standard for raw speed/efficiency | Index only — you manage text/metadata storage separately |
| **Qdrant** | Extremely fast (Rust), great filtering, production-grade | Requires separate service (Docker), more setup |

**Decision**: **ChromaDB** — simplest setup (`pip install chromadb`), no Docker needed.

### Embeddings Model

| Tool | Pros | Cons |
|------|------|------|
| **HuggingFace** (`all-MiniLM-L6-v2`) | Free, runs locally, data stays private, decent performance | Uses local CPU/GPU, need to manage model files |
| **OpenAI** (`text-embedding-3-small`) | Top-tier performance, simple API, no local compute | Paid, data privacy concerns, API latency |

**Decision**: **HuggingFace** — start local/free, upgrade later if needed.

### Chunking Strategy

- **Config driven**: `chunk_size`, `chunk_overlap`, and `collection_name` are read from `config.txt`
- **Splitter**: Recursive Character Text Splitter
- **Chunk Size**: Configurable (default 1000 chars; set to 6000 for richer context)
- **Overlap**: Configurable (default 100 chars)
- **Multiple collections**: Each config produces a separate ChromaDB collection — switch instantly by editing `config.txt`

