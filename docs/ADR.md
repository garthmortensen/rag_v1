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
