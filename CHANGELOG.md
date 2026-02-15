## 0.10.2 (2026-02-15)

### Feat

- **banner**: add section overview banner to Streamlit UI showing available PDF sections/subsections
- **scan**: add `scan_pdf_sections()` lightweight probe for section metadata
- **models**: update `PROVIDER_MODELS` with latest model IDs (GPT-5.x, Claude Opus 4, Gemini 3)
- **rewriter**: add `src/generation/rewriter.py` — rewrite structured PDFs in plain English section-by-section (rewrite/summarize modes, CLI + programmatic API)
- **rewriter-ui**: add `app_rewriter.py` — Streamlit UI for PDF rewriting with per-section progress bar, live preview, and download
- **rewriter**: add `RewriteProgress` dataclass and `rewrite_pdf_iter()` generator for streaming progress to UIs

### Fix

- **config**: optimize chunk size to 2000 chars (from 10000) for better section-aware splitting

## 0.10.1 (2026-02-14)

### Refactor

- **refactor**: major refactor to add evaluation

## 0.10.0 (2026-02-14)

### Feat

- **langchain**: replace diy functionality with langchain

## 0.9.0 (2026-02-14)

### Feat

- **ui**: add streamlit ui, citations parser and support for multiple collections
- **multiple**: add logs, improve printouts, add db inspection, add beep, add configuration to printout
- **config**: implement config driven approach to allow for multiple chromadb collections, separated by embedding length (1k, 2k, ..., nk)

## 0.8.0 (2026-02-13)

### Feat

- **answer**: add llm answer step

## 0.7.0 (2026-02-12)

### Feat

- **retrieval**: implement retrieval and tests

## 0.6.0 (2026-02-12)

### Feat

- **embeddings**: implement embeddings and model selection based on VRAM

## 0.5.0 (2026-02-11)

### Feat

- **embeddings**: add starter embeddings feature
- **chunking**: implement chunking using langchain text_splitters
- **loaders**: implement langchain community loaders which use strategy pattern to load correct file ext loader, parse into docs, which is called by dir loader, called by pipeline
- **dir**: restructure around src standard package
- **downloading**: add retries and backoff, functionalize, cleanup
- **metadata**: capture additional metadata
- **data-pull**: add script which downloads all files listed in source csv file, with attractive console display

### Fix

- **raw-data**: document data sources
- **pytest**: remove pytest from ci to prevent build error
