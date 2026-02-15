"""Streamlit UI for rewriting structured PDFs in plain English.

Provides a point-and-click interface for the PDF rewriter with
per-section progress indicators, live preview, and download.

Launch:
    uv run streamlit run app_rewriter.py
"""

import glob
import html as _html
import os
import queue
import threading
import time

import streamlit as st
import streamlit.components.v1 as components

# ── Page config (must be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="PDF Rewriter",
    page_icon="📝",
    layout="wide",
)

from src.config import CFG  # noqa: E402
from src.generation.llm import (  # noqa: E402
    PROVIDER_DEFAULTS,
    PROVIDER_MODELS,
)
from src.generation.rewriter import (  # noqa: E402
    rewrite_pdf_iter,
    DEFAULT_OUTPUT_DIR,
)
from src.ingestion.pdf_section_splitter import (  # noqa: E402
    has_section_headers,
    scan_pdf_sections,
)

# ── Constants ───────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), "corpus", "raw_data")
DEFAULT_PROVIDER = str(CFG.get("llm_provider", "ollama"))
DEFAULT_MODEL = str(CFG.get("llm_model", "llama3.2:3b"))


# ── Discover structured PDFs (cached) ──────────────────────────────

@st.cache_data(show_spinner="Scanning PDFs…")
def _discover_structured_pdfs(raw_dir: str) -> dict[str, str]:
    """Return {display_name: filepath} for PDFs with section headers."""
    result: dict[str, str] = {}
    for pdf in sorted(glob.glob(os.path.join(raw_dir, "*.pdf"))):
        if has_section_headers(pdf):
            short = (
                os.path.basename(pdf)
                .replace(".pdf", "")
                .replace("_", " ")
                .title()
            )
            result[short] = pdf
    return result


@st.cache_data(show_spinner=False)
def _get_section_preview(filepath: str) -> dict[str, list[str]]:
    """Return section → subsection mapping for a PDF."""
    return scan_pdf_sections(filepath)


# ── Copy helpers ────────────────────────────────────────────────────

def _copy_button(text: str, label: str, key: str) -> None:
    """Render a single-click copy-to-clipboard button."""
    escaped = _html.escape(text).replace("`", "&#96;").replace("\n", "\\n")
    components.html(
        f"""
        <button onclick="navigator.clipboard.writeText(`{escaped}`.replace(/\\n/g,'\\n'))
            .then(()=>{{this.textContent='✅ Copied!';setTimeout(()=>this.textContent='{label}',1500)}})"
            style="background:#262730;color:#fafafa;border:1px solid #4a4a5a;
                   border-radius:6px;padding:4px 14px;cursor:pointer;font-size:14px;">
            {label}
        </button>
        """,
        height=42,
    )


# ── Discover available PDFs ─────────────────────────────────────────
structured_pdfs = _discover_structured_pdfs(RAW_DIR)

# ── Title ───────────────────────────────────────────────────────────
st.title("📝 PDF Rewriter")
st.caption(
    "Rewrite structured model documentation PDFs in plain English, "
    "section by section."
)

if not structured_pdfs:
    st.warning(
        "No structured PDFs found in `corpus/raw_data/`. "
        "Download the corpus first with `uv run src/ingestion/downloader.py`."
    )
    st.stop()

# ── PDF picker (main area — full width for long names) ──────────────
pdf_names = list(structured_pdfs.keys())
selected_name = st.selectbox("📄 PDF to rewrite", pdf_names)
selected_path = structured_pdfs[selected_name]


# ── Sidebar controls ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Mode
    mode = st.radio(
        "✍️ Mode",
        options=["rewrite", "summarize"],
        format_func=lambda m: {
            "rewrite": "📖 Full Rewrite — plain English, all content",
            "summarize": "📋 Summarize — 2–3 sentences per section",
        }[m],
    )

    st.divider()

    # LLM settings
    st.subheader("🤖 LLM")
    provider_list = list(PROVIDER_DEFAULTS.keys())
    default_idx = (
        provider_list.index(DEFAULT_PROVIDER)
        if DEFAULT_PROVIDER in provider_list
        else 0
    )
    provider = st.selectbox("Provider", provider_list, index=default_idx)

    model_options = PROVIDER_MODELS.get(provider, [])
    display_options = list(model_options) + ["Other…"]
    default_model = (
        DEFAULT_MODEL
        if provider == DEFAULT_PROVIDER
        else PROVIDER_DEFAULTS.get(provider, "")
    )
    default_model_idx = (
        model_options.index(default_model)
        if default_model in model_options
        else 0
    )
    model_choice = st.selectbox("Model", display_options, index=default_model_idx)
    if model_choice == "Other…":
        model = st.text_input("Custom model name", value="")
    else:
        model = model_choice

    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05
    )

    st.divider()

    # Custom prompt
    st.subheader("💬 Custom Instructions")

    _DEFAULT_CUSTOM_PROMPT = """\
Narrative-First, Code-Aided, Simple-English, Minimal-Math Rewrite Guide

**Goal:** Rewrite the doc as a readable technical story: what problem \
we're solving, what drives outcomes, how the model works. Math is secondary.

**Principles**
* Explain *why + so what* before *how*.
* Simple Equations only.
* Map everything to business objects (borrower/loan/collateral/macro/outcomes).
* Use toy examples and Python to make it real.
* Almost always provide a mermaid diagram to illustrate the flow of data and decisions.
* Try to provide memory aids like intuition, metaphors, and checklists to make it stick.

**Key points**
1. **Purpose + output + where it fits** (CCAR/PPNR/loss models)
2. **Narrative pipeline**: inputs → transforms → prediction → aggregation → reporting
3. **Key variables** (plain English bullets)
4. **Intuition/metaphor** (1–2 lines)
5. **Python toy example** (tiny dataframe + simple model + 1 sanity check)
6. **Mermaid diagrams like flow, sequence, etc** (conceptual, not math)"""

    custom_prompt = st.text_area(
        "Extra instructions for the LLM",
        value=_DEFAULT_CUSTOM_PROMPT,
        height=320,
        help=(
            "These instructions are injected into the prompt alongside "
            "the default rewrite/summarize rules. Clear the box for defaults only."
        ),
    )
    # Treat empty string as None
    custom_prompt = custom_prompt.strip() or None


# ── Section preview ────────────────────────────────────────────────
sections = _get_section_preview(selected_path)
section_count = len(sections)
subsection_count = sum(len(subs) for subs in sections.values())

col1, col2, col3 = st.columns(3)
col1.metric("Sections", section_count)
col2.metric("Subsections", subsection_count)
col3.metric("Mode", mode.title())

with st.expander(
    f"📑 Sections in **{selected_name}**",
    expanded=False,
):
    for sec, subs in sections.items():
        if subs:
            st.markdown(f"- **{sec}** — _{', '.join(subs)}_")
        else:
            st.markdown(f"- **{sec}**")


# ── Run button ──────────────────────────────────────────────────────
st.divider()

# Persist results across Streamlit re-runs so download URLs stay valid
if "rewrite_result" not in st.session_state:
    st.session_state.rewrite_result = None

if st.button(
    f"🚀 {mode.title()} this PDF",
    type="primary",
    use_container_width=True,
):
    if not model:
        st.error("Please select or enter a model name.")
        st.stop()

    # Estimate total sections from the splitter (subsections + intro blocks)
    # The actual count comes from rewrite_pdf_iter as it runs
    progress_bar = st.progress(0, text="Loading PDF sections…")
    status_container = st.status(
        f"📝 {mode.title()}ing with {provider}/{model}…",
        expanded=True,
    )

    output_container = st.container()
    start_time = time.time()

    # Placeholder for the "currently processing" live timer
    active_section_placeholder = st.empty()

    try:
        result_path = None
        last_progress = None
        all_rewritten: list[str] = []
        section_start: float | None = None
        # Info about the section currently being processed (for timer)
        active_label: str = ""
        active_bar_text: str = ""

        # Run the iterator in a background thread so the main thread
        # can tick the elapsed timer every second while the LLM works.
        progress_q: queue.Queue = queue.Queue()

        def _run_iter() -> None:
            """Push progress events onto the queue from a thread."""
            try:
                for p in rewrite_pdf_iter(
                    selected_path,
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    mode=mode,
                    custom_prompt=custom_prompt,
                    output_dir=DEFAULT_OUTPUT_DIR,
                ):
                    progress_q.put(p)
            except Exception as exc:
                progress_q.put(exc)
            finally:
                progress_q.put(None)  # sentinel

        worker = threading.Thread(target=_run_iter, daemon=True)
        worker.start()

        finished = False
        while not finished:
            # Drain all available progress events
            while True:
                try:
                    item = progress_q.get_nowait()
                except queue.Empty:
                    break

                if item is None:
                    finished = True
                    break
                if isinstance(item, Exception):
                    raise item

                progress = item

                if progress.phase == "done":
                    result_path = progress.output_path
                    progress_bar.progress(1.0, text="✅ Complete!")
                    active_section_placeholder.empty()
                    last_progress = progress
                    finished = True
                    break

                if progress.phase == "processing":
                    section_start = time.time()
                    active_label = (
                        f"⏳ **[{progress.step}/{progress.total}]** "
                        f"Processing **{progress.section}** › "
                        f"**{progress.subsection}** "
                        f"({progress.chars:,} chars)…"
                    )
                    active_bar_text = (
                        f"⏳ [{progress.step}/{progress.total}] "
                        f"Processing {progress.section} › "
                        f"{progress.subsection} "
                        f"({progress.chars:,} chars)…"
                    )
                    progress_bar.progress(
                        max((progress.step - 1) / progress.total, 0.0),
                        text=active_bar_text,
                    )
                    continue

                # phase == "rewriting" — section finished
                last_progress = progress

                sec_elapsed = (
                    time.time() - section_start if section_start else 0
                )
                sec_m, sec_s = divmod(int(sec_elapsed), 60)
                section_start = None
                active_label = ""

                pct = progress.step / progress.total
                progress_bar.progress(
                    pct,
                    text=(
                        f"[{progress.step}/{progress.total}] "
                        f"{progress.section} › {progress.subsection} "
                        f"({progress.chars:,} chars)"
                    ),
                )

                active_section_placeholder.empty()

                elapsed = time.time() - start_time
                mins, secs = divmod(int(elapsed), 60)
                with status_container:
                    st.write(
                        f"✅ **[{progress.step}/{progress.total}]** "
                        f"{progress.section} › {progress.subsection} "
                        f"— {progress.chars:,} chars "
                        f"({sec_m:02d}:{sec_s:02d} section · "
                        f"{mins:02d}:{secs:02d} elapsed)"
                    )

                with output_container:
                    with st.expander(
                        f"**{progress.section}** › {progress.subsection}",
                        expanded=(progress.step == progress.total),
                    ):
                        st.markdown(progress.rewritten)
                        _copy_button(
                            progress.rewritten,
                            label="📋 Copy Section",
                            key=f"copy_sec_{progress.step}",
                        )
                    all_rewritten.append(
                        f"## {progress.section} › {progress.subsection}\n\n"
                        f"{progress.rewritten}"
                    )

            if finished:
                break

            # No events ready — tick the live timer
            if active_label:
                overall = time.time() - start_time
                ov_m, ov_s = divmod(int(overall), 60)
                sec_el = time.time() - (section_start or time.time())
                sc_m, sc_s = divmod(int(sec_el), 60)
                active_section_placeholder.info(
                    f"{active_label}  \n"
                    f"_Section: {sc_m:02d}:{sc_s:02d} · "
                    f"Overall: {ov_m:02d}:{ov_s:02d}_"
                )

            time.sleep(1)

        # ── Done ────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)

        total_sections = last_progress.total if last_progress else 0

        status_container.update(
            label=(
                f"✅ {mode.title()} complete — "
                f"{total_sections} sections in {mins:02d}:{secs:02d}"
            ),
            state="complete",
            expanded=False,
        )

        # Download and Copy All buttons
        if result_path and os.path.isfile(result_path):
            config_path = os.path.join(os.path.dirname(result_path), "config.txt")
            st.success(
                f"✅ Saved to disk  ·  ⏱ {mins:02d}:{secs:02d}\n\n"
                f"📄 **Markdown:** `{result_path}`  \n"
                f"⚙️ **Config:** `{config_path}`"
            )
            with open(result_path, "r", encoding="utf-8") as f:
                md_content = f.read()

            # Persist for copy button to survive re-runs
            st.session_state.rewrite_result = {
                "md_content": md_content,
                "all_rewritten": "\n\n".join(all_rewritten),
                "result_path": result_path,
            }

    except ValueError as exc:
        st.error(f"❌ {exc}")
    except ConnectionError as exc:
        st.error(f"❌ Connection error: {exc}")
    except ImportError as exc:
        st.error(f"❌ Missing package: {exc}")
    except Exception as exc:
        st.error(f"❌ Unexpected error: {exc}")

# ── Persistent copy button (survives re-runs) ──────────────────────
if st.session_state.rewrite_result:
    _res = st.session_state.rewrite_result
    _copy_button(
        _res["all_rewritten"],
        label="📋 Copy All Sections",
        key="copy_all",
    )
