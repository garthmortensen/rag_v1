"""LLM-powered answer generation over retrieved document chunks.

Uses a local Ollama model (via langchain-ollama) to produce
grounded answers from the top-k chunks returned by the retrieval
module.  The prompt instructs the model to cite sources and
acknowledge uncertainty.

Prerequisites
-------------
1. Install the Ollama runtime:
       curl -fsSL https://ollama.com/install.sh | sh
2. Pull a model:
       ollama pull llama3.2:3b
3. Ensure the Ollama server is running:
       ollama serve          # or it may auto-start as a systemd service

Usage (programmatic):
    from src.generation.llm import ask
    answer = ask("What is the peak unemployment rate?")
"""

import logging
import textwrap

from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_K = 5

# ── System prompt ───────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""\
    You are a precise research assistant for U.S. Federal Reserve
    stress testing documents.

    RULES:
    1. Answer ONLY from the provided context chunks.
    2. Cite the source document for every claim (use the title or
       filename from the metadata).
    3. If the context does not contain enough information, say
       "I don't have enough information to answer that."
    4. Be concise — prefer bullet points over long paragraphs.
    5. Never fabricate data, figures, or document names.
""")


# ── Prompt builder ──────────────────────────────────────────────────

def build_prompt(query: str, chunks: list[dict]) -> str:
    """Assemble a RAG prompt from a query and retrieved chunks.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    chunks : list[dict]
        Output of ``retrieve_formatted()`` — each dict has keys
        ``rank``, ``id``, ``distance``, ``text``, ``metadata``.

    Returns
    -------
    str
        A formatted prompt ready to send to the LLM.
    """
    context_parts: list[str] = []
    for chunk in chunks:
        title = chunk["metadata"].get("title", "Unknown")
        source = chunk["metadata"].get("source", "Unknown")
        context_parts.append(
            f"[Source: {title} | {source}]\n{chunk['text']}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    prompt = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"ANSWER:"
    )
    return prompt


# ── LLM interaction ─────────────────────────────────────────────────

def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ChatOllama:
    """Create a ChatOllama instance.

    Parameters
    ----------
    model : str
        Ollama model tag (e.g. ``"llama3.2:3b"``).
    temperature : float
        Sampling temperature (lower = more deterministic).

    Returns
    -------
    ChatOllama
        A LangChain chat model backed by Ollama.
    """
    logger.info(f"Initialising Ollama LLM: model={model}, temp={temperature}")
    return ChatOllama(model=model, temperature=temperature)


def generate_answer(
    query: str,
    chunks: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Send a RAG prompt to the local Ollama model and return the answer.

    Parameters
    ----------
    query : str
        The user's question.
    chunks : list[dict]
        Retrieved document chunks (from ``retrieve_formatted()``).
    model : str
        Ollama model tag.
    temperature : float
        Sampling temperature.

    Returns
    -------
    str
        The model's generated answer text.

    Raises
    ------
    ConnectionError
        If the Ollama server is not reachable.
    """
    prompt = build_prompt(query, chunks)
    llm = get_llm(model=model, temperature=temperature)

    logger.info(
        f"Generating answer  (model={model}, "
        f"chunks={len(chunks)}, query='{query[:60]}')"
    )

    try:
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", prompt),
        ]
        response = llm.invoke(messages)
        answer = response.content
    except Exception as exc:
        # Surface a clear message when Ollama isn't running
        exc_str = str(exc).lower()
        if "connection" in exc_str or "refused" in exc_str:
            raise ConnectionError(
                "Could not reach the Ollama server at localhost:11434.\n"
                "Make sure Ollama is installed and running:\n"
                "  1. curl -fsSL https://ollama.com/install.sh | sh\n"
                "  2. ollama pull llama3.2:3b\n"
                "  3. ollama serve"
            ) from exc
        raise

    logger.info(f"Answer generated ({len(answer)} chars)")
    return answer


# ── Convenience wrapper ─────────────────────────────────────────────

def ask(
    query: str,
    n_results: int = DEFAULT_TOP_K,
    where: dict | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    persist_dir: str | None = None,
    collection_name: str | None = None,
) -> dict:
    """End-to-end RAG: retrieve chunks then generate an answer.

    This is the single-call entry point that wires retrieval to
    generation.

    Parameters
    ----------
    query : str
        Natural-language question.
    n_results : int
        Number of chunks to retrieve.
    where : dict | None
        Optional ChromaDB metadata filter.
    model : str
        Ollama model tag.
    temperature : float
        Sampling temperature.
    persist_dir : str | None
        ChromaDB directory (uses default if None).
    collection_name : str | None
        ChromaDB collection name (uses default if None).

    Returns
    -------
    dict
        ``{"query": ..., "answer": ..., "chunks": ..., "model": ...}``
    """
    # Lazy import to avoid circular deps and keep retrieval usable
    # without generation deps installed
    from src.retrieval.query import retrieve_formatted
    from src.embedding.model import COLLECTION_NAME, VECTOR_DB_DIR

    _persist_dir = persist_dir or VECTOR_DB_DIR
    _collection_name = collection_name or COLLECTION_NAME

    logger.info(f"ask() — retrieving top-{n_results} chunks")
    chunks = retrieve_formatted(
        query,
        n_results=n_results,
        where=where,
        persist_dir=_persist_dir,
        collection_name=_collection_name,
    )

    answer = generate_answer(
        query,
        chunks,
        model=model,
        temperature=temperature,
    )

    return {
        "query": query,
        "answer": answer,
        "chunks": chunks,
        "model": model,
    }
