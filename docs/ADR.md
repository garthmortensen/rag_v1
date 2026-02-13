# Architecture Decision Records (ADR)

This doc contains ADRs for the **RAG Stress Testing** project.

# ADR-001: Technology Stack

**Context:** Establishing the foundational architecture for the RAG Stress Testing project.

---

## 1.1 Package Management: `uv`

**Context:** Python environments are traditionally slow. If we leverage github CI/CD, we should use uv for speed.

**Decision:** Use **uv**.

**Rationale:** Rust-based speed; unifies python version management, venv creation, and dependency resolution in one tool.

---

## 1.2 Code Quality: `ruff`

**Context:** Linting and formatting usually require a slow chain of tools (Flake8, Black, isort).

**Decision:** Use **Ruff**.

**Rationale:** Single, incredibly fast binary that replaces the entire legacy linting blockchain with zero config overhead. That means github actions ci/cd runs faster.

---

## 1.3 Testing: `pytest`

**Context:** Write and run tests.

**Decision:** Use **pytest**.

**Rationale:** De facto standard.

---

## 1.4 Workflow: `commitizen` & GitHub Actions

**Context:** CI/CD and versioning depend on consistent commit history.

**Decision:** Use **Commitizen** and **GitHub Actions**.

**Rationale:** Enforces semantic versioning via conventional commits; automating checks (`uv run ruff`, `uv run pytest`) prevents bad code from merging.

---

## 1.5 Document Loading: `langchain-community`

**Context:** We need to read HTML, CSV, PDF, and other file formats into a standard document representation for downstream processing.

**Decision:** Use **LangChain Community** document loaders.

**Rationale:** Provides pre-built loaders for 10+ file types (BSHTMLLoader, CSVLoader, PyPDFLoader, etc.) with a consistent interface — each returns a list of `Document` objects with `.page_content` and `.metadata`. Avoids writing and maintaining custom parsers for each format.

---

## 1.6 Vector Database: `ChromaDB`

**Context:** We need a vector database to store embeddings and enable semantic search.

**Decision:** Use **ChromaDB** in embedded (in-process) mode.

**Rationale:** Simplest setup (`pip install chromadb`), runs in the same Python process with no Docker or server required. Stores data to disk at `corpus/vector_db/`. Sufficient for a single-user learning/development project.

---

## 1.7 LLM Backend: `Ollama`

**Context:** The retrieval module returns relevant document chunks, but users want natural-language answers grounded in those chunks (RAG generation).

**Decision:** Use **Ollama** as the local LLM runtime, integrated via **langchain-ollama**.

**Rationale:** Runs fully offline — no API keys, no cloud costs, no data leakage. Supports many open-weight models (Llama 3.2, Phi-3, Mistral, etc.) with a single `ollama pull` command. The `langchain-ollama` package provides a `ChatOllama` class that plugs directly into the existing LangChain ecosystem. Default model is `llama3.2:3b` — small enough for laptops yet capable enough for grounded Q&A.
