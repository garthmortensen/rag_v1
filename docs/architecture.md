# Architecture

## Project Structure

```text
rag_stress_testing_v1/
├── app.py                     # Streamlit web UI
├── config.txt                 # Pipeline settings (chunk size, collection, LLM provider)
├── .env.example               # API key template (copy to .env)
├── corpus/
│   ├── data_sources.csv       # URLs to download
│   ├── metadata.csv           # Per-file download metadata (doc_id, title, author, …)
│   ├── raw_data/              # Downloaded files (.html, .csv, .pdf, .xlsx)
│   └── vector_db/             # ChromaDB persistent storage (HNSW index)
├── src/
│   ├── config.py              # Reads config.txt + .env (via python-dotenv)
│   ├── utils.py               # RAM-aware embedding model selection
│   ├── ingestion/
│   │   ├── downloader.py      # download files from data_sources.csv
│   │   ├── loaders.py         # Extract: LangChain Community document loaders
│   │   └── processor.py       # Pipeline orchestrator (load → chunk → embed → store)
│   ├── embedding/
│   │   └── model.py           # Transform + Load: embed via HuggingFace, upsert to ChromaDB
│   ├── retrieval/
│   │   ├── query.py           # Semantic search: embed query → ANN lookup → ranked results
│   │   └── query_logger.py    # Log query sessions to logs/
│   └── generation/
│       └── llm.py             # Multi-provider LLM factory + RAG generation
├── tests/
│   ├── test_config.py
│   ├── test_downloader.py
│   ├── test_embedding.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_utils.py
├── docs/
│   └── architecture.md
└── pyproject.toml
```

---

## Pipeline (ETL)

```mermaid
graph LR
    subgraph Phase1["Phase 1 — Acquire"]
        CSV["corpus/data_sources.csv"]
        DL["downloader.py"]
        RAW["corpus/raw_data/"]
        META["corpus/metadata.csv"]

        CSV --> DL
        DL --> RAW
        DL --> META
    end

    subgraph Phase2["Phase 2 — Ingest & Index"]
        direction LR
        subgraph Extract
            LOAD["loaders.py<br/>load_directory()"]
        end

        subgraph Transform
            CHUNK["processor.py<br/>chunk_documents()<br/>1000 chars / 100 overlap"]
            EMBED["model.py<br/>HuggingFaceEmbeddings<br/>all-MiniLM-L6-v2 · 384d"]
        end

        subgraph Load
            CHROMA[("ChromaDB<br/>corpus/vector_db/<br/>collection: stress_test_docs_1k | _6k")]
        end

        LOAD --> CHUNK --> EMBED --> CHROMA
    end

    subgraph Phase3["Phase 3 — Retrieve"]
        QUERY["query.py<br/>retrieve() / retrieve_formatted()"]
        SEMBED["HuggingFaceEmbeddings<br/>embed_query()"]
        ANN["ChromaDB ANN search<br/>top-k nearest neighbours"]

        QUERY --> SEMBED --> ANN
    end

    subgraph Phase4["Phase 4 — Generate"]
        PROMPT["llm.py<br/>build_prompt()"]
        LLM["get_llm()<br/>ollama / openai / anthropic / google"]
        ANSWER["Grounded answer<br/>with source citations"]

        PROMPT --> LLM --> ANSWER
    end

    RAW --> LOAD
    META --> EMBED
    CHROMA --> ANN
    ANN --> PROMPT

    style Phase1 fill:none,stroke:#555
    style Phase2 fill:none,stroke:#555
    style Phase3 fill:none,stroke:#555
    style Phase4 fill:none,stroke:#555
    style Extract fill:none,stroke:#4a9eed
    style Transform fill:none,stroke:#4aed9e
    style Load fill:none,stroke:#ed4a9e
```

---

## Pipeline Sequence Diagram

```mermaid
sequenceDiagram
    autonumber

    actor User
    participant PROC as processor.py<br/>run()
    participant DL as downloader.py<br/>download_files()
    participant WEB as Remote Server
    participant FS as corpus/raw_data/
    participant META as corpus/metadata.csv
    participant LOAD as loaders.py<br/>load_directory()
    participant CHUNK as processor.py<br/>chunk_documents()
    participant EMB as model.py<br/>embed_and_store()
    participant HF as HuggingFaceEmbeddings<br/>all-MiniLM-L6-v2
    participant CHROMA as 🛢 ChromaDB<br/>stress_test_docs_*
    participant QUERY as query.py<br/>retrieve()
    participant LLM as llm.py<br/>ask()
    participant OLLAMA as get_llm()<br/>ollama / openai /<br/>anthropic / google

    rect rgba(0, 0, 0, 0)
        Note over DL,FS: Phase 1 — Acquire
        User ->> PROC: run()
        PROC ->> DL: download_files()
        DL ->> DL: read data_sources.csv
        loop Each URL in data_sources.csv
            DL ->> WEB: GET url
            WEB -->> DL: HTTP response (html/csv/pdf)
            DL ->> FS: write file to raw_data/
            DL ->> META: append row (doc_id, title, author, …)
        end
        DL -->> PROC: download complete
    end

    rect rgba(0, 0, 0, 0)
        Note over LOAD,CHROMA: Phase 2 — Ingest & Index
        PROC ->> LOAD: load_directory("corpus/raw_data/")
        loop Each supported file (LOADER_MAP)
            LOAD ->> LOAD: select loader by extension<br/>(.html→BSHTMLLoader, .csv→CSVLoader, …)
            LOAD ->> FS: read file
            FS -->> LOAD: file contents
            LOAD ->> LOAD: loader.load() → Document(page_content, metadata)
        end
        LOAD -->> PROC: list[Document]

        PROC ->> CHUNK: chunk_documents(docs)
        CHUNK ->> CHUNK: RecursiveCharacterTextSplitter<br/>size=1000, overlap=100
        CHUNK -->> PROC: list[Document] (chunked)

        PROC ->> EMB: embed_and_store(chunks)
        EMB ->> META: load_doc_id_map()
        META -->> EMB: {local_path → {doc_id, title, author, …}}
        EMB ->> EMB: assign chunk IDs<br/>{doc_id}_chunk_{n:04d}<br/>or SHA-256 fallback
        EMB ->> EMB: enrich metadata<br/>(doc_id, title, author, source_url, source_type)
        EMB ->> HF: embed_documents(chunk_texts)
        HF -->> EMB: list[384-dim vectors]

        loop Batches of 500
            EMB ->> CHROMA: collection.upsert(ids, embeddings, documents, metadatas)
        end
        EMB -->> PROC: int (count upserted)
    end

    rect rgba(0, 0, 0, 0)
        Note over QUERY,CHROMA: Phase 3 — Retrieve
        User ->> QUERY: retrieve("What is the peak unemployment rate?")
        QUERY ->> HF: embed_query(query)
        HF -->> QUERY: 384-dim query vector
        QUERY ->> CHROMA: collection.query(query_embedding, n_results=k)
        CHROMA -->> QUERY: {ids, documents, metadatas, distances}
        QUERY ->> QUERY: retrieve_formatted() → list[dict]<br/>(rank, id, distance, text, metadata)
        QUERY -->> User: top-k ranked chunks
    end

    rect rgba(0, 0, 0, 0)
        Note over LLM,OLLAMA: Phase 4 — Generate
        User ->> LLM: ask("What is the peak unemployment rate?")
        LLM ->> QUERY: retrieve_formatted(query, n_results=k)
        QUERY -->> LLM: list[dict] (retrieved chunks)
        LLM ->> LLM: build_prompt(query, chunks)<br/>CONTEXT + QUESTION + ANSWER template
        LLM ->> OLLAMA: invoke([system_prompt, human_prompt])
        OLLAMA -->> LLM: grounded answer with source citations
        LLM -->> User: {query, answer, chunks, model}
    end
```

---

## Retrieval & Generation Sequence Diagram

A focused view of what happens when a user submits a query with `--answer`.

```mermaid
sequenceDiagram
    autonumber

    actor User
    participant CLI as query.py<br/>main()
    participant CFG as config.txt<br/>CFG singleton
    participant RET as query.py<br/>retrieve_formatted()
    participant HF as HuggingFaceEmbeddings<br/>embed_query()
    participant CHROMA as 🛢 ChromaDB<br/>collection.query()
    participant GEN as query.py<br/>_generate_and_print_answer()
    participant LLM as llm.py<br/>generate_answer()
    participant PROMPT as llm.py<br/>build_prompt()
    participant OLLAMA as get_llm()<br/>ollama / openai /<br/>anthropic / google
    participant LOG as query_logger.py<br/>log_query_session()

    Note over User,CLI: User runs: python -m src.retrieval.query "question" --answer

    %% ── Startup ──
    rect rgba(0, 0, 0, 0)
        Note over CLI,CFG: Startup
        CLI ->> CLI: print_ascii_banner()
        CLI ->> CFG: read collection_name,<br/>beep_on_answer, …
        CFG -->> CLI: CFG dict
        CLI ->> CLI: print_config()
    end

    %% ── Parse filter ──
    opt --filter key=value provided
        CLI ->> CLI: _parse_filter("source_type=pdf")<br/>→ {"source_type": "pdf"}
    end

    %% ── Retrieval ──
    rect rgba(0, 0, 0, 0)
        Note over RET,CHROMA: Retrieval — semantic search
        CLI ->> RET: retrieve_formatted(query, n_results=k, where=filter)
        RET ->> HF: embed_query(query)
        HF ->> HF: tokenize → encode → pool
        HF -->> RET: 384-dim query vector
        RET ->> CHROMA: collection.query(<br/>  query_embeddings=[vec],<br/>  n_results=k,<br/>  where=filter<br/>)
        CHROMA ->> CHROMA: HNSW ANN search<br/>cosine distance
        CHROMA -->> RET: {ids, documents,<br/>metadatas, distances}
        RET ->> RET: flatten → list[dict]<br/>(rank, id, distance, text, metadata)
        RET -->> CLI: top-k chunks
    end

    CLI ->> CLI: _print_results(chunks)<br/>display ranked chunks to terminal

    %% ── Generation ──
    rect rgba(0, 0, 0, 0)
        Note over GEN,OLLAMA: Generation — LLM grounded answer
        CLI ->> GEN: _generate_and_print_answer(query, chunks, model)
        GEN ->> LLM: generate_answer(query, chunks)
        LLM ->> PROMPT: build_prompt(query, chunks)

        loop Each retrieved chunk
            PROMPT ->> PROMPT: format: [Source: title | path]<br/>chunk_text
        end
        PROMPT ->> PROMPT: assemble CONTEXT + QUESTION + ANSWER template
        PROMPT -->> LLM: prompt string

        LLM ->> OLLAMA: invoke([<br/>  ("system", SYSTEM_PROMPT),<br/>  ("human", prompt)<br/>])

        Note over OLLAMA: Local inference<br/>(temp=0.1)
        OLLAMA ->> OLLAMA: decode tokens
        OLLAMA -->> LLM: response.content
        LLM -->> GEN: answer string
        GEN ->> GEN: print answer in Rich Panel

        opt beep_on_answer = true
            GEN ->> User: 🔔 terminal bell (\a)
        end

        GEN -->> CLI: answer string
    end

    %% ── Logging ──
    rect rgba(0, 0, 0, 0)
        Note over LOG: Session logging
        CLI ->> LOG: log_query_session(<br/>  query, results,<br/>  answer, collection_name<br/>)
        LOG ->> LOG: write logs/YYYYMMDD_HHMMSS.log<br/>CONFIG + query + chunks + answer
    end

    CLI -->> User: done
```

---

## Data Flow Detail

```mermaid
graph TD
    subgraph Input["corpus/raw_data/"]
        HTML[".html"]
        CSVF[".csv"]
        PDF[".pdf"]
        XLSX[".xlsx"]
        OTHER[".txt .md .json<br/>.docx .pptx"]
    end

    subgraph Loaders["loaders.py — LOADER_MAP"]
        BS4["BSHTMLLoader"]
        CL["CSVLoader"]
        PY["PyPDFLoader"]
        XL["UnstructuredExcelLoader"]
        OL["TextLoader / …"]
    end

    subgraph Documents["LangChain Document"]
        DOC[".page_content + .metadata"]
    end

    subgraph Chunking["processor.py"]
        SPLIT["RecursiveCharacterTextSplitter<br/>size=1000 · overlap=100"]
    end

    subgraph Embedding["model.py"]
        META2["load_doc_id_map()<br/>corpus/metadata.csv"]
        IDS["Chunk IDs<br/>{doc_id}_chunk_{n:04d}<br/>or SHA-256 fallback"]
        EMB["HuggingFaceEmbeddings<br/>all-MiniLM-L6-v2"]
        ENRICH["Metadata enrichment<br/>doc_id · title · author<br/>source_url · source_type"]
    end

    subgraph Storage["ChromaDB"]
        UPSERT["collection.upsert()<br/>batch_size=500"]
        DB[("corpus/vector_db/<br/>HNSW index")]
    end

    HTML --> BS4
    CSVF --> CL
    PDF --> PY
    XLSX --> XL
    OTHER --> OL

    BS4 --> DOC
    CL --> DOC
    PY --> DOC
    XL --> DOC
    OL --> DOC

    DOC --> SPLIT
    SPLIT --> IDS
    META2 --> IDS
    META2 --> ENRICH
    SPLIT --> EMB
    IDS --> UPSERT
    EMB --> UPSERT
    ENRICH --> UPSERT
    UPSERT --> DB

    style Input fill:none,stroke:#555
    style Loaders fill:none,stroke:#4a9eed
    style Documents fill:none,stroke:#888
    style Chunking fill:none,stroke:#4aed9e
    style Embedding fill:none,stroke:#edd74a
    style Storage fill:none,stroke:#ed4a9e
```

---

## Module Dependency Graph

```mermaid
graph BT
    PROC["processor.py<br/><i>orchestrator</i>"]
    LOAD["loaders.py<br/><i>10 LangChain loaders</i>"]
    MODEL["model.py<br/><i>embed + store</i>"]
    DL["downloader.py<br/><i>HTTP fetcher</i>"]
    QUERY["query.py<br/><i>semantic search</i>"]
    GEN["llm.py<br/><i>multi-provider RAG</i>"]

    LC_SPLIT["langchain-text-splitters"]
    LC_COMM["langchain-community"]
    LC_HF["langchain-huggingface"]
    LC_OL["langchain-ollama<br/>langchain-openai<br/>langchain-anthropic<br/>langchain-google-genai"]
    CHROMA["chromadb"]

    PROC --> LOAD
    PROC --> MODEL
    LOAD --> LC_COMM
    PROC --> LC_SPLIT
    MODEL --> LC_HF
    MODEL --> CHROMA
    QUERY --> MODEL
    GEN --> QUERY
    GEN --> LC_OL
    DL --> META_CSV["corpus/metadata.csv"]
    MODEL --> META_CSV

    style PROC fill:none,stroke:#4aed9e
    style LOAD fill:none,stroke:#4a9eed
    style MODEL fill:none,stroke:#edd74a
    style DL fill:none,stroke:#888
    style QUERY fill:none,stroke:#4aed9e
    style GEN fill:none,stroke:#ed4aed
```

---

## Architecture Decision Records (ADR)

### ADR-001: Technology Stack

**Context:** Establishing the foundational architecture for the RAG Stress Testing project.

#### 1.1 Package Management: `uv`

**Context:** Python environments are traditionally slow. If we leverage github CI/CD, we should use uv for speed.

**Decision:** Use **uv**.

**Rationale:** Rust-based speed; unifies python version management, venv creation, and dependency resolution in one tool.

#### 1.2 Code Quality: `ruff`

**Context:** Linting and formatting usually require a slow chain of tools (Flake8, Black, isort).

**Decision:** Use **Ruff**.

**Rationale:** Single, incredibly fast binary that replaces the entire legacy linting blockchain with zero config overhead. That means github actions ci/cd runs faster.

#### 1.3 Testing: `pytest`

**Context:** Write and run tests.

**Decision:** Use **pytest**.

**Rationale:** De facto standard.

#### 1.4 Workflow: `commitizen` & GitHub Actions

**Context:** CI/CD and versioning depend on consistent commit history.

**Decision:** Use **Commitizen** and **GitHub Actions**.

**Rationale:** Enforces semantic versioning via conventional commits; automating checks (`uv run ruff`, `uv run pytest`) prevents bad code from merging.

#### 1.5 Document Loading: `langchain-community`

**Context:** We need to read HTML, CSV, PDF, and other file formats into a standard document representation for downstream processing.

**Decision:** Use **LangChain Community** document loaders.

**Rationale:** Provides pre-built loaders for 10+ file types (BSHTMLLoader, CSVLoader, PyPDFLoader, etc.) with a consistent interface — each returns a list of `Document` objects with `.page_content` and `.metadata`. Avoids writing and maintaining custom parsers for each format.

#### 1.6 Vector Database: `ChromaDB`

**Context:** We need a vector database to store embeddings and enable semantic search.

**Decision:** Use **ChromaDB** in embedded (in-process) mode.

**Rationale:** Simplest setup (`pip install chromadb`), runs in the same Python process with no Docker or server required. Stores data to disk at `corpus/vector_db/`. Sufficient for a single-user learning/development project.

#### 1.7 LLM Backend: Multi-provider via `config.txt`

**Context:** The retrieval module returns relevant document chunks, but users want natural-language answers grounded in those chunks (RAG generation). Not all users have a GPU-capable machine for local inference.

**Decision:** Default to **Ollama** for local inference; support **OpenAI**, **Anthropic**, and **Google Gemini** as remote alternatives. The provider is configured in `config.txt` (`llm_provider`) and API keys are loaded from `.env` via `python-dotenv`.

**Rationale:** Ollama runs fully offline — no API keys, no cloud costs, no data leakage. For users with weaker hardware, remote providers offer access to larger models without local GPU requirements. The `get_llm()` factory in `llm.py` lazily imports only the needed provider package, so unused providers add zero overhead. Each provider’s LangChain integration (`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`) is an optional dependency installed only when needed.

---

## Ingestion & Indexing Design

### Conceptual Overview (ETL for RAG)

Phase 1 output raw HTML, CSV, and PDF files. Phase 2 builds the ETL pipeline to create a vector database for retrieval.

1. **Extract**: Loaders read raw files (HTML, CSV, PDF) and convert them into standard `Document` objects (text + metadata).
2. **Transform**:
    - **Chunking**: Split documents into smaller segments using a recursive character splitter. This keeps sentence boundaries intact and adds overlap to preserve context.
    - **Embedding**: Run chunks through an encoder model (e.g., `all-MiniLM-L6-v2`) to generate dense vectors.
        - **Encoder Model**: A neural network trained to understand context and semantic similarity, not just keyword matching.
        - **Dense Vectors**: Fixed-size arrays of floating-point numbers (e.g., 384 dimensions) where closer proximity in space equals closer similarity in meaning.
3. **Load**: Upsert vectors and metadata into an HNSW index (ChromaDB) for approximate nearest neighbor (ANN) search.
    - **Upsert (Update + Insert)**: A database operation that updates an existing record if it exists, or inserts a new one if it doesn't. This ensures idempotency (running the script twice doesn't create duplicate entries).

### `src` Layout Rationale

The project uses a `src` layout, a standard pattern in modern Python packaging:

1. **Enforces Installation**: Prevents accidentally importing local code without installing it (avoiding "it works on my machine" errors).
2. **Cleaner Namespace**: Keeps the root directory for configuration (`pyproject.toml`, `README`) and tests, reducing clutter.
3. **Explicit Imports**: Ensures that tests run against the installed package, mirroring how a user would use it.

### Stack Selection

#### Orchestration Framework

| Tool | Pros | Cons |
| ---- | ---- | ---- |
| **LangChain** | Massive ecosystem, standard interface, rich document loaders/splitters | Bloated/over-abstracted, hard to debug, frequent breaking changes |
| **Haystack** | Explicit pipeline design (DAGs), Pythonic/readable, production-ready | Smaller ecosystem than LangChain |
| **Pure Python** | Maximum control, zero dependency bloat, easy debugging | Reinventing wheels, higher maintenance code |

**Decision**: **LangChain** — its `Community` document loaders and text splitters save days of work.

#### Vector Database

| Tool | Pros | Cons |
| ---- | ---- | ---- |
| **ChromaDB** | Open-source, in-process (no Docker), Python-native, simple storage/indexing | Newer project, SQLite/ClickHouse wrapper, scaling limits |
| **FAISS** | Gold standard for raw speed/efficiency | Index only — you manage text/metadata storage separately |
| **Qdrant** | Extremely fast (Rust), great filtering, production-grade | Requires separate service (Docker), more setup |

**Decision**: **ChromaDB** — simplest setup (`pip install chromadb`).

#### Embeddings Model

| Tool | Pros | Cons |
| ---- | ---- | ---- |
| **HuggingFace** (`all-MiniLM-L6-v2`) | Free, runs locally, data stays private, decent performance | Uses local CPU/GPU, need to manage model files |
| **OpenAI** (`text-embedding-3-small`) | Top-tier performance, simple API, no local compute | Paid, data privacy concerns, API latency |

**Decision**: **HuggingFace** — start local/free, upgrade later if needed.

### Implementation Details

#### Loaders (`src/ingestion/loaders.py`) ✅

Implemented via LangChain Community loaders with a strategy-pattern `LOADER_MAP` that maps 10 file extensions to their loader class. `load_directory()` iterates `corpus/raw_data/`, calls `load_file()` per supported file, and returns a flat list of `Document` objects with `.page_content` and `.metadata["source"]`.

#### Chunking (`src/ingestion/processor.py`) ✅

Implemented in `processor.py` → `chunk_documents()` using `RecursiveCharacterTextSplitter`.

- **Chunk Size**: 1,000 characters (~200–300 words).
- **Overlap**: 100 characters.

Each chunk inherits its parent document's metadata.

#### Embedding & Storage (`src/embedding/model.py`) ✅

Implemented in `embed_and_store()`. Called as Step 3 in `processor.run()` after chunking.

- **Library**: `langchain-huggingface` (`HuggingFaceEmbeddings` wrapper around `sentence-transformers`).
- **Model**: `all-MiniLM-L6-v2` — 384-dimensional dense vectors. Downloaded on first run and cached in `~/.cache/huggingface/`.
- All chunk texts are embedded in a single `embed_documents()` call.

#### Chunk ID Strategy

Fully traceable, idempotent IDs derived from `corpus/metadata.csv`:

1. `load_doc_id_map()` reads the CSV into a dict keyed by `local_path` (e.g. `{"corpus/raw_data/credit_risk_models.pdf": {"doc_id": "1JA8WZFYSY0", ...}}`).
2. Each chunk's `.metadata["source"]` is the same `local_path` value (set by `loaders.py`), so the join is a simple dict lookup.
3. IDs follow the format `{doc_id}_chunk_{n:04d}` — e.g. `"1JA8WZFYSY0_chunk_0042"`. The `n` is a zero-padded sequential counter per source file.

**Hash fallback**: If a source file has no entry in `metadata.csv` (e.g. it was manually placed in `raw_data/`), `_fallback_doc_id()` generates a deterministic 11-character SHA-256 hex prefix of the source path instead. A warning is logged.

#### Metadata Enrichment

Each chunk's ChromaDB metadata starts with the loader-provided fields (`source`, `page`, `row`, etc.) and is then enriched with fields from `metadata.csv`:

- `doc_id`, `title`, `author`, `source_url`, `source_type`

This enables filtered queries like `where={"author": "www.federalreserve.gov"}` or `where={"source_type": "pdf"}`.

#### Batch Upserts

ChromaDB's underlying SQLite backend limits the number of parameters per statement (~5,461 records on most systems). The corpus can easily exceed this (a 600-page PDF at 1,000-char chunks produces thousands of chunks).

`embed_and_store()` slices the upsert into batches of 500 (configurable via `batch_size`) and logs progress per batch. This is a manual loop rather than using `chromadb.utils.batch_utils.create_batches` for transparency.

#### ChromaDB Configuration

- **Mode**: Embedded/in-process via `chromadb.PersistentClient(path="corpus/vector_db/")`.
- **Collection**: Driven by `config.txt` (default `"stress_test_docs_1k"`), created via `get_or_create_collection()`.
- **Index**: HNSW (ChromaDB default) for approximate nearest neighbor search.
- `corpus/vector_db/` is gitignored.

### Pipeline Flow (`processor.run()`)

```text
Step 1: load_directory()        → list[Document]      (loaders.py)
Step 2: chunk_documents(docs)   → list[Document]      (processor.py)
Step 3: embed_and_store(chunks) → int (count upserted) (model.py)
```

### Module Constants (`src/embedding/model.py`)

| Constant | Value |
| -------- | ----- |
| `MODEL_NAME` | `"all-MiniLM-L6-v2"` |
| `VECTOR_DB_DIR` | `"corpus/vector_db"` |
| `COLLECTION_NAME` | from `config.txt` (default `"stress_test_docs_1k"`) |
| `METADATA_CSV` | `"corpus/metadata.csv"` |
| `DEFAULT_BATCH_SIZE` | `500` |

## Retireval visual

```text
         ╭─── 0.95 ─────────────────────────╮
       ╭─│── 0.87 ────────────────────────│─╮
     ╭─│─│── 0.74 ─────────────────────│─│─╮
     │ │ │                              │ │ │
     │ │ │  •stress_test    ◆ QUERY     │ │ │
     │ │ │   •CCAR                      │ │ │
     │ │ │              •credit_risk    │ │ │
     │ │ •Basel_III      •market_risk   │ │ │
     │ •VaR                             │ │ │
     ╰─│─│─────────────────────────│─│─╯ │ │
       ╰─│─────────────────────────│─╯   │ │
         ╰───────────────────────────────╯ │
           Cosine similarity → spatial proximity
```
