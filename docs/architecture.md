# Architecture

## Project Structure

```text
rag_stress_testing_v1/
├── corpus/
│   ├── data_sources.csv       # URLs to download
│   ├── metadata.csv           # Per-file download metadata (doc_id, title, author, …)
│   ├── raw_data/              # Downloaded files (.html, .csv, .pdf, .xlsx)
│   └── vector_db/             # ChromaDB persistent storage (HNSW index)
├── src/
│   ├── ingestion/
│   │   ├── downloader.py      # Phase 1: download files from data_sources.csv
│   │   ├── loaders.py         # Extract: LangChain Community document loaders
│   │   └── processor.py       # Pipeline orchestrator (load → chunk → embed → store)
│   └── embedding/
│       └── model.py           # Transform + Load: embed via HuggingFace, upsert to ChromaDB
├── tests/
│   ├── test_downloader.py
│   └── test_embedding.py
├── docs/
│   ├── ADR.md
│   ├── architecture.md
│   └── phase_2.md
└── pyproject.toml
```

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
            CHROMA[("ChromaDB<br/>corpus/vector_db/<br/>collection: stress_test_docs")]
        end

        LOAD --> CHUNK --> EMBED --> CHROMA
    end

    RAW --> LOAD
    META --> EMBED

    style Phase1 fill:#1a1a2e,stroke:#555,color:#fff
    style Phase2 fill:#16213e,stroke:#555,color:#fff
    style Extract fill:#1a3a5c,stroke:#4a9eed,color:#fff
    style Transform fill:#1a4a3a,stroke:#4aed9e,color:#fff
    style Load fill:#4a1a3a,stroke:#ed4a9e,color:#fff
```

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

    style Input fill:#2d2d2d,stroke:#555,color:#fff
    style Loaders fill:#1a3a5c,stroke:#4a9eed,color:#fff
    style Documents fill:#2a2a4a,stroke:#888,color:#fff
    style Chunking fill:#1a4a3a,stroke:#4aed9e,color:#fff
    style Embedding fill:#3a3a1a,stroke:#edd74a,color:#fff
    style Storage fill:#4a1a3a,stroke:#ed4a9e,color:#fff
```

## Module Dependency Graph

```mermaid
graph BT
    PROC["processor.py<br/><i>orchestrator</i>"]
    LOAD["loaders.py<br/><i>10 LangChain loaders</i>"]
    MODEL["model.py<br/><i>embed + store</i>"]
    DL["downloader.py<br/><i>HTTP fetcher</i>"]

    LC_SPLIT["langchain-text-splitters"]
    LC_COMM["langchain-community"]
    LC_HF["langchain-huggingface"]
    CHROMA["chromadb"]

    PROC --> LOAD
    PROC --> MODEL
    LOAD --> LC_COMM
    PROC --> LC_SPLIT
    MODEL --> LC_HF
    MODEL --> CHROMA
    DL --> META_CSV["corpus/metadata.csv"]
    MODEL --> META_CSV

    style PROC fill:#1a4a3a,stroke:#4aed9e,color:#fff
    style LOAD fill:#1a3a5c,stroke:#4a9eed,color:#fff
    style MODEL fill:#3a3a1a,stroke:#edd74a,color:#fff
    style DL fill:#2d2d2d,stroke:#888,color:#fff
```

## 
