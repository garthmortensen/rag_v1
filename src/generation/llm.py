"""LLM-powered answer generation over retrieved document chunks.

Built around a **LangChain LCEL chain** that composes:

    prompt → llm → output parser

The chain can be used standalone or via the convenience wrappers
:func:`generate_answer` and :func:`stream_answer`.

Supported providers (set ``llm_provider`` in config.txt):

* **ollama** — local Ollama server (default, no API key needed)
* **openai** — OpenAI API (requires ``OPENAI_API_KEY`` in .env)
* **anthropic** — Anthropic API (requires ``ANTHROPIC_API_KEY``)
* **google** — Google Gemini API (requires ``GOOGLE_API_KEY``)

API keys are loaded from a ``.env`` file at the project root via
python-dotenv (see ``.env.example``).

If ``LANGCHAIN_TRACING_V2=true`` and ``LANGCHAIN_API_KEY`` are set
in ``.env``, every call is automatically traced to LangSmith — no
code changes required.

Usage (programmatic)::

    from src.generation.llm import rag_chain, stream_answer
    chain  = rag_chain()                    # reusable LCEL chain
    answer = chain.invoke({"context": "...", "question": "..."})
"""

import logging
import textwrap
from collections.abc import Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
    "google": "gemini-2.0-flash",
}

# ── Provider → curated model lists ─────────────────────────────────
PROVIDER_MODELS: dict[str, list[str]] = {
    "ollama": [
        "llama3.2:3b",
        "llama3.2:1b",
        "llama3.3:70b",
        "phi3",
        "mistral",
        "gemma2",
    ],
    "openai": [
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5",
        "gpt-4.1",
        "gpt-4.1-mini",
        "o4-mini",
        "o3-mini",
    ],
    "anthropic": ["claude-opus-4-6", "claude-sonnet-4-5", "claude-haiku-4-5"],
    "google": ["gemini-3-pro", "gemini-3-flash", "gemini-2.5-pro", "gemini-2.5-flash"],
}

# ── System prompt ───────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""\
    You are a precise research assistant.

    RULES:
    1. Answer ONLY from the provided context chunks.
    2. Cite the source document for every claim (use the title or
       filename from the metadata).
    3. If the context does not contain enough information, say
       "I don't have enough information to answer that."
    4. Be concise — prefer bullet points over long paragraphs.
    5. Never fabricate data, figures, or document names.
""")


# ── Chat prompt template (LangChain) ───────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"),
    ]
)


# ── Context formatter ──────────────────────────────────────────────


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt.

    Parameters
    ----------
    chunks : list[dict]
        Output of ``retrieve_formatted()`` — each dict has keys
        ``rank``, ``id``, ``distance``, ``text``, ``metadata``.

    Returns
    -------
    str
        Newline-separated context with source citations.
    """
    context_parts: list[str] = []
    for chunk in chunks:
        title = chunk["metadata"].get("title", "Unknown")
        source = chunk["metadata"].get("source", "Unknown")
        context_parts.append(f"[Source: {title} | {source}]\n{chunk['text']}")
    return "\n\n---\n\n".join(context_parts)


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
        ``"google"``.

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
        f"Unknown llm_provider '{provider}'. Supported: {', '.join(PROVIDER_DEFAULTS)}"
    )


# ── LCEL chain ──────────────────────────────────────────────────────


def rag_chain(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    provider: str = DEFAULT_PROVIDER,
):
    """Build a reusable LCEL chain: prompt → llm → string output.

    The chain expects a dict with keys ``context`` (str) and
    ``question`` (str).

    Returns
    -------
    Runnable
        ``RAG_PROMPT | llm | StrOutputParser()``
    """
    llm = get_llm(model=model, temperature=temperature, provider=provider)
    return RAG_PROMPT | llm | StrOutputParser()


# ── Generate ────────────────────────────────────────────────────────


def generate_answer(
    query: str,
    chunks: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    provider: str = DEFAULT_PROVIDER,
) -> str:
    """Send a RAG prompt to the configured LLM and return the answer.

    Uses the LCEL chain internally.

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
        LLM provider name (ollama, openai, anthropic, google).

    Returns
    -------
    str
        The model's generated answer text.

    Raises
    ------
    ConnectionError
        If the LLM server is not reachable.
    """
    logger.info(
        f"Generating answer  (provider={provider}, model={model}, "
        f"chunks={len(chunks)}, query='{query[:60]}')"
    )

    try:
        chain = rag_chain(
            model=model,
            temperature=temperature,
            provider=provider,
        )
        context_block = format_context(chunks)
        answer = chain.invoke({"context": context_block, "question": query})
    except Exception as exc:
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


# ── Streaming (for Streamlit) ───────────────────────────────────────


def stream_answer(
    query: str,
    chunks: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    provider: str = DEFAULT_PROVIDER,
) -> Iterator[str]:
    """Stream an answer token-by-token using the LCEL chain.

    Yields string fragments as they arrive from the LLM.

    Parameters
    ----------
    query : str
        The user's question.
    chunks : list[dict]
        Retrieved document chunks (from ``retrieve_formatted()``).
    model, temperature, provider
        LLM configuration — same as :func:`generate_answer`.

    Yields
    ------
    str
        Token-by-token chunks of the answer.
    """
    chain = rag_chain(
        model=model,
        temperature=temperature,
        provider=provider,
    )
    context_block = format_context(chunks)

    logger.info(
        f"Streaming answer  (provider={provider}, model={model}, "
        f"chunks={len(chunks)}, query='{query[:60]}')"
    )

    yield from chain.stream({"context": context_block, "question": query})
