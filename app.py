"""Streamlit UI for RAG Stress Testing.

Multi-pane browser interface with sidebar controls and a
chat-style main area for querying stress testing documents.

Answers stream token-by-token using the LCEL chain's
``.stream()`` method via ``st.write_stream()``.

Launch:
    uv run streamlit run app.py
"""

import time
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

# ── Page config (must be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="RAG Stress Testing",
    page_icon="🏦",
    layout="wide",
)

from src.config import CFG, config_as_text
from src.embedding.model import VECTOR_DB_DIR, COLLECTION_NAME
from src.generation.llm import (
    stream_answer,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
    PROVIDER_DEFAULTS,
    PROVIDER_MODELS,
)
from src.retrieval.query import retrieve_formatted
from src.retrieval.query_logger import log_query_session

import chromadb


# ── Sidebar: config & controls ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # Show active config.txt
    with st.expander("config.txt", expanded=True):
        st.code(config_as_text(), language="ini")

    st.divider()

    # Collection picker — detect available collections
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    collections = [c.name for c in client.list_collections()]
    default_idx = (
        collections.index(COLLECTION_NAME)
        if COLLECTION_NAME in collections
        else 0
    )
    selected_collection = st.selectbox(
        "Collection", collections, index=default_idx
    )

    # Show collection stats
    col = client.get_collection(selected_collection)
    st.caption(f"📄 {col.count():,} chunks")

    st.divider()

    # Retrieval settings
    st.subheader("🔍 Retrieval")
    top_k = st.slider("Top-K results", min_value=1, max_value=30, value=5)

    # File type filter checkboxes
    FILE_TYPES = ["pdf", "html", "csv", "xlsx"]
    st.markdown("**Source types**")
    selected_types = []
    cols = st.columns(2)
    for i, ft in enumerate(FILE_TYPES):
        with cols[i % 2]:
            if st.checkbox(ft.upper(), value=True, key=f"ft_{ft}"):
                selected_types.append(ft)

    st.divider()

    # ── Hierarchical Source → Category filter ───────────────────────
    st.markdown("**Sources & Categories**")
    col_meta = col.get(include=["metadatas"])

    # Build source_org → {categories} mapping from collection metadata
    source_category_map: dict[str, set[str]] = {}
    for m in col_meta["metadatas"]:
        if not m:
            continue
        org = m.get("source_org", "") or "Unknown"
        cat = m.get("category", "")
        source_category_map.setdefault(org, set())
        if cat:
            source_category_map[org].add(cat)

    # Sort for stable display
    sorted_sources = sorted(source_category_map.keys())

    # Parent (source_org) checkboxes with child (category) checkboxes
    selected_sources: list[str] = []
    selected_categories: list[str] = []
    all_categories: list[str] = sorted(
        {cat for cats in source_category_map.values() for cat in cats}
    )

    for src in sorted_sources:
        cats = sorted(source_category_map[src])
        # Parent checkbox
        src_on = st.checkbox(f"🏛 {src}", value=True, key=f"src_{src}")
        if src_on:
            selected_sources.append(src)
            # Child category checkboxes, indented
            if cats:
                for cat in cats:
                    if st.checkbox(
                        f"　　📂 {cat}",
                        value=True,
                        key=f"cat_{src}_{cat}",
                    ):
                        selected_categories.append(cat)

    st.divider()

    # LLM settings
    st.subheader("🤖 Generation")
    provider_list = list(PROVIDER_DEFAULTS.keys())
    default_provider_idx = (
        provider_list.index(DEFAULT_PROVIDER)
        if DEFAULT_PROVIDER in provider_list
        else 0
    )
    provider = st.selectbox(
        "LLM Provider", provider_list, index=default_provider_idx
    )

    # Model selector: curated list + "Other" escape hatch
    model_options = PROVIDER_MODELS.get(provider, [])
    default_model = (
        DEFAULT_MODEL
        if provider == DEFAULT_PROVIDER
        else PROVIDER_DEFAULTS.get(provider, "")
    )
    # Build display list: curated models + "Other…"
    display_options = list(model_options) + ["Other…"]
    default_idx = (
        model_options.index(default_model)
        if default_model in model_options
        else 0
    )
    model_choice = st.selectbox(
        "Model", display_options, index=default_idx
    )
    if model_choice == "Other…":
        model = st.text_input("Custom model name", value="")
    else:
        model = model_choice

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.05,
    )

# ── Main area ───────────────────────────────────────────────────────
st.title("🏦 RAG Stress Testing")
st.caption(
    f"Collection: **{selected_collection}** · "
    f"Provider: **{provider}** · "
    f"Model: **{model}** · "
    f"Top-K: **{top_k}**"
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Copy helper ─────────────────────────────────────────────────────

def _copy_button(question: str, answer: str, key: str) -> None:
    """Render a single-click copy-to-clipboard button for a Q&A pair."""
    import html as _html

    qa_text = f"Q: {question}\n\nA: {answer}"
    escaped = _html.escape(qa_text).replace("`", "&#96;").replace("\n", "\\n")
    components.html(
        f"""
        <button onclick="navigator.clipboard.writeText(`{escaped}`.replace(/\\n/g,'\n'))
            .then(()=>{{this.textContent='✅ Copied!';setTimeout(()=>this.textContent='📋 Copy Q&A',1500)}})"
            style="background:#262730;color:#fafafa;border:1px solid #4a4a5a;
                   border-radius:6px;padding:4px 14px;cursor:pointer;font-size:14px;">
            📋 Copy Q&A
        </button>
        """,
        height=42,
    )


# ── Chunk renderer ──────────────────────────────────────────────────

def _render_chunk(chunk: dict) -> None:
    """Render a single retrieved chunk inside an expander."""
    meta = chunk["metadata"]
    title = meta.get("title", "Unknown")
    source = meta.get("source", "Unknown")
    category = meta.get("category", "—")
    source_org = meta.get("source_org", "—")
    distance = chunk.get("distance", "—")
    if isinstance(distance, float):
        distance = f"{distance:.4f}"

    st.markdown(
        f"**#{chunk['rank']}** — {title}  \n"
        f"`source_org: {source_org}` · `category: {category}` · `distance: {distance}`  \n"
        f"`source: {source}`"
    )
    st.code(chunk["text"][:3000], language="text")
    st.divider()


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "timestamp" in msg:
            ts = msg["timestamp"]
            if msg["role"] == "user":
                st.caption(f"🕐 Asked: {ts.strftime('%A, %H:%M:%S')}")
            elif msg["role"] == "assistant":
                label = f"🕐 Answered: {ts.strftime('%A, %H:%M:%S')}"
                if "query_time" in msg:
                    label = f"🕐 Asked: {msg['query_time'].strftime('%H:%M:%S')}  →  Answered: {ts.strftime('%H:%M:%S')}"
                if "delta" in msg:
                    label += f"  ·  ⏱ {msg['delta']}"
                st.caption(label)
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(
                f"📚 Retrieved Sources ({len(msg['sources'])} chunks)",
                expanded=False,
            ):
                for chunk in msg["sources"]:
                    _render_chunk(chunk)
        st.markdown(msg["content"])
        # Copy Q&A button for assistant messages
        if msg["role"] == "assistant" and "query" in msg:
            _copy_button(msg["query"], msg["content"], key=f"copy_{id(msg)}")


# ── Chat input ──────────────────────────────────────────────────────
query = st.chat_input("Ask a question about stress testing documents…")

if query:
    # Show user message
    query_time = datetime.now()
    st.session_state.messages.append(
        {"role": "user", "content": query, "timestamp": query_time}
    )
    with st.chat_message("user"):
        st.caption(f"🕐 Asked: {query_time.strftime('%A, %H:%M:%S')}")
        st.markdown(query)

    # Build metadata filter from checkboxes
    conditions = []
    if selected_types and len(selected_types) < len(FILE_TYPES):
        if len(selected_types) == 1:
            conditions.append({"source_type": selected_types[0]})
        else:
            conditions.append({"source_type": {"$in": selected_types}})
    if selected_sources and len(selected_sources) < len(sorted_sources):
        if len(selected_sources) == 1:
            conditions.append({"source_org": selected_sources[0]})
        else:
            conditions.append({"source_org": {"$in": selected_sources}})
    if selected_categories and len(selected_categories) < len(all_categories):
        if len(selected_categories) == 1:
            conditions.append({"category": selected_categories[0]})
        else:
            conditions.append({"category": {"$in": selected_categories}})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    # Retrieve chunks
    with st.status("🔍 Retrieving chunks…", expanded=False) as retrieval_status:
        chunks = retrieve_formatted(
            query,
            n_results=top_k,
            where=where,
            persist_dir=VECTOR_DB_DIR,
            collection_name=selected_collection,
        )
        retrieval_status.update(
            label=f"✅ Retrieved {len(chunks)} chunks", state="complete",
        )

    if not chunks:
        with st.chat_message("assistant"):
            st.warning("No matching chunks found. Try a different query or filter.")
        st.stop()

    # Show retrieved sources immediately
    with st.expander(
        f"📚 Retrieved Sources ({len(chunks)} chunks)", expanded=True
    ):
        for chunk in chunks:
            _render_chunk(chunk)

    # Generate LLM answer (streaming, token-by-token)
    with st.chat_message("assistant"):
        with st.status(f"🤖 Generating answer with {provider}/{model}…", expanded=False) as gen_status:
            start = time.time()
            gen_status.update(label=f"🤖 Generating answer with {provider}/{model}…", state="running")
        try:
            answer = st.write_stream(
                stream_answer(
                    query,
                    chunks,
                    model=model,
                    temperature=temperature,
                    provider=provider,
                )
            )
        except ConnectionError as exc:
            st.error(str(exc))
            st.stop()
        except ImportError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            st.stop()

        elapsed = time.time() - start
        mins, secs = divmod(int(elapsed), 60)
        delta_str = f"{mins:02d}:{secs:02d}"
        elapsed_detailed = f"{elapsed:.1f}s"

        response_time = datetime.now()
        st.caption(
            f"🕐 Asked: {query_time.strftime('%H:%M:%S')}  →  "
            f"Answered: {response_time.strftime('%H:%M:%S')}  ·  "
            f"⏱ {delta_str} ({elapsed_detailed})"
        )

    # Copy Q&A to clipboard
    _copy_button(query, answer, key="copy_latest")

    # Save assistant message with sources for history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "query": query,
            "query_time": query_time,
            "sources": chunks,
            "timestamp": response_time,
            "delta": f"{delta_str} ({elapsed_detailed})",
        }
    )

    # Log the full session to logs/
    log_query_session(
        query=query,
        results=chunks,
        answer=answer,
        collection_name=selected_collection,
        query_time=query_time,
        response_time=response_time,
        elapsed_seconds=elapsed,
        provider=provider,
        model=model,
    )
