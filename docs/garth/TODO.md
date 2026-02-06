# Project Roadmap

## Phase 1: setup
- [x] Initialize project structure with `uv`
- [x] Configure code quality tools (`ruff`)
- [x] Set up testing framework (`pytest`)
- [x] Establish CI/CD pipeline (GitHub Actions)
- [x] Define initial list of data sources (e.g., Federal Reserve DFAST/CCAR instructions, EBA guidelines)
- [ ] Ensure project works across different OS (Linux, Windows) with CI/CD

## Phase 2: get & process data
- [x] Build scrapers for public stress testing documentation (PDFs/HTML/TXT)
- [ ] Implement text extraction pipeline
- [ ] Design chunking strategy for long docs
- [ ] Implement metadata extraction (year, document type, source)

## Phase 3: vector store
- [ ] Choose and set up vector database (ChromaDB recommended)
- [ ] Generate embeddings (HuggingFace `all-MiniLM-L6-v2`)
- [ ] Build ingestion script to upsert chunks into vector DB
- [ ] Deduplicate entries (idempotent upserts)
- [ ] Verify retrieval with test queries

## Phase 4: retrieval & generation
- [ ] Implement semantic search over vector store
- [ ] Integrate LLM for answer generation (RAG chain)
- [ ] Build a simple query interface (CLI or notebook)
- [ ] Add source citation to generated answers

## Phase 5: evaluation & hardening
- [ ] Build evaluation dataset (question/answer pairs)
- [ ] Measure retrieval quality (precision, recall, MRR)
- [ ] Measure generation quality (faithfulness, relevance)
- [ ] Tune chunk size, overlap, and top-k parameters
- [ ] Add logging and error handling
