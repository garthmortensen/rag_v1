## 0.11.0 (2026-02-15)

### Feat

- **rewrite**: add rewriter app with custom prompts that process ordered pdfs
- **banner**: add banner which shows simple TOC of pdfs for quick reference
- **splitting**: section splitting

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
