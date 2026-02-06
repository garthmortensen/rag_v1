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

