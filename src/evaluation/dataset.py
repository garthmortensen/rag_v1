"""Evaluation dataset for ragas scoring.

A curated set of question / ground-truth pairs drawn from the
active corpus. Each entry has:

* **question** — a natural-language query
* **ground_truth** — the expected factual answer (used to measure
  context recall)

Expand this list whenever you add new documents to the corpus or
want to test a new retrieval / generation behaviour.
"""

EVAL_QUESTIONS: list[dict[str, str]] = [
    {
        "question": "What is retrieval-augmented generation (RAG)?",
        "ground_truth": (
            "RAG combines retrieval of relevant documents with an LLM, "
            "grounding the answer in retrieved context."
        ),
    },
    {
        "question": "What does the ingestion pipeline do?",
        "ground_truth": (
            "It loads files, splits text into chunks, embeds the chunks, "
            "and stores embeddings plus metadata in the vector database."
        ),
    },
    {
        "question": "What is a ChromaDB collection used for in this project?",
        "ground_truth": (
            "A collection is a named group of embeddings and documents; "
            "the project reads/writes the active collection from config."
        ),
    },
    {
        "question": "What does chunk overlap do?",
        "ground_truth": (
            "Chunk overlap repeats some text between consecutive chunks "
            "to reduce boundary effects and improve retrieval continuity."
        ),
    },
    {
        "question": "What should the system prompt enforce?",
        "ground_truth": (
            "It should restrict answers to provided context, require citations, "
            "and require acknowledging when context is insufficient."
        ),
    },
    {
        "question": "What is a metadata filter used for during retrieval?",
        "ground_truth": (
            "It narrows results by restricting matches to documents with specific "
            "metadata values (e.g., source_type, category)."
        ),
    },
    {
        "question": "What is top-k in retrieval?",
        "ground_truth": (
            "Top-k is the number of nearest chunks returned by the vector search."
        ),
    },
    {
        "question": "When should you re-run ingestion?",
        "ground_truth": (
            "After adding or changing source documents, or after changing chunking "
            "settings or collection name so embeddings are rebuilt for that collection."
        ),
    },
]
