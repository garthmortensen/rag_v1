"""LLM-powered answer generation over retrieved document chunks.

Uses a configurable LLM provider to produce grounded answers from
the top-k chunks returned by the retrieval module.  The prompt
instructs the model to cite sources and acknowledge uncertainty.

Supported providers (set ``llm_provider`` in config.txt):

* **ollama** — local Ollama server (default, no API key needed)
* **openai** — OpenAI API (requires ``OPENAI_API_KEY`` in .env)
* **anthropic** — Anthropic API (requires ``ANTHROPIC_API_KEY``)
* **groq** — Groq API (requires ``GROQ_API_KEY``, free tier available)
* **google** — Google Gemini API (requires ``GOOGLE_API_KEY``)

API keys are loaded from a ``.env`` file at the project root via
python-dotenv (see ``.env.example``).

Usage (programmatic):
    from src.generation.llm import ask
    answer = ask("What is the peak unemployment rate?")
"""

import logging
import textwrap

from langchain_ollama import ChatOllama

from src.config import CFG

logger = logging.getLogger(__name__)

# ── Defaults (read from config.txt, fall back to built-in) ──────────
DEFAULT_PROVIDER: str = str(CFG.get("llm_provider", "ollama"))
DEFAULT_MODEL: str = str(CFG.get("llm_model", "llama3.2:3b"))
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_K = 5

# ── Provider → default model mapping ───────────────────────────────
PROVIDER_DEFAULTS: dict[str, str] = {
    "ollama": "llama3.2:3b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "groq": "llama-3.3-70b-versatile",
    "google": "gemini-2.0-flash",
}

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
    provider: str = DEFAULT_PROVIDER,
):
    """Create a LangChain chat model for the given provider.

    Parameters
    ----------
    model : str
        Model name/tag for the chosen provider.
    temperature : float
        Sampling temperature (lower = more deterministic).
    provider : str
        One of ``"ollama"``, ``"openai"``, ``"anthropic"``,
        ``"groq"``, ``"google"``.

    Returns
    -------
    BaseChatModel
        A LangChain chat model instance.

    Raises
    ------
    ValueError
        If *provider* is not recognised.
    ImportError
        If the required provider package is not installed.
    """
    provider = provider.lower().strip()
    logger.info(
        f"Initialising LLM: provider={provider}, model={model}, temp={temperature}"
    )

    if provider == "ollama":
        return ChatOllama(model=model, temperature=temperature)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is required for the openai provider.\n"
                "  Run: uv add langchain-openai"
            )
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is required for the anthropic provider.\n"
                "  Run: uv add langchain-anthropic"
            )
        return ChatAnthropic(model=model, temperature=temperature)

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "langchain-groq is required for the groq provider.\n"
                "  Run: uv add langchain-groq"
            )
        return ChatGroq(model=model, temperature=temperature)

    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for the google provider.\n"
                "  Run: uv add langchain-google-genai"
            )
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    raise ValueError(
        f"Unknown llm_provider '{provider}'. "
        f"Supported: {', '.join(PROVIDER_DEFAULTS)}"
    )


def generate_answer(
    query: str,
    chunks: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    provider: str = DEFAULT_PROVIDER,
) -> str:
    """Send a RAG prompt to the configured LLM and return the answer.

    Parameters
    ----------
    query : str
        The user's question.
    chunks : list[dict]
        Retrieved document chunks (from ``retrieve_formatted()``).
    model : str
        Model name/tag for the chosen provider.
    temperature : float
        Sampling temperature.
    provider : str
        LLM provider name (ollama, openai, anthropic, groq, google).

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
    llm = get_llm(model=model, temperature=temperature, provider=provider)

    logger.info(
        f"Generating answer  (provider={provider}, model={model}, "
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
        # Surface a clear message when a local Ollama isn't running
        exc_str = str(exc).lower()
        if "connection" in exc_str or "refused" in exc_str:
            raise ConnectionError(
                f"Could not reach the LLM server (provider={provider}).\n"
                + (
                    "Make sure Ollama is installed and running:\n"
                    "  1. curl -fsSL https://ollama.com/install.sh | sh\n"
                    "  2. ollama pull llama3.2:3b\n"
                    "  3. ollama serve"
                    if provider == "ollama"
                    else f"Check your API key and network for provider '{provider}'."
                )
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
    provider: str = DEFAULT_PROVIDER,
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
        Model name/tag for the chosen provider.
    temperature : float
        Sampling temperature.
    provider : str
        LLM provider name (ollama, openai, anthropic, groq, google).
    persist_dir : str | None
        ChromaDB directory (uses default if None).
    collection_name : str | None
        ChromaDB collection name (uses default if None).

    Returns
    -------
    dict
        ``{"query": ..., "answer": ..., "chunks": ..., "model": ...,
        "provider": ...}``
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
        provider=provider,
    )

    return {
        "query": query,
        "answer": answer,
        "chunks": chunks,
        "model": model,
        "provider": provider,
    }
