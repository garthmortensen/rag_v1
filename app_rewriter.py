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
    refine_markdown_iter,
    discover_rewrite_outputs,
    _split_markdown_sections,
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
            short = os.path.basename(pdf).replace(".pdf", "").replace("_", " ").title()
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
    "Rewrite structured model documentation PDFs in plain English, section by section."
)


# ── Sidebar controls ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Mode
    mode = st.radio(
        "✍️ Mode",
        options=["rewrite", "summarize", "refine"],
        format_func=lambda m: {
            "rewrite": "📖 Full Rewrite — plain English, all content",
            "summarize": "📋 Summarize — 2–3 sentences per section",
            "refine": "🔄 Refine — further edit an existing output",
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
        model_options.index(default_model) if default_model in model_options else 0
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

    _DEFAULT_REWRITE_PROMPT = """\
Rewrite this section as a plain-English technical story. Lead with why the model exists and what it produces, then walk through the narrative pipeline (inputs → transforms → prediction → aggregation → reporting), keeping math minimal. Map every concept to real business objects (borrower, loan, collateral, macro, outcomes) and include a small Python toy example with a sanity check, a mermaid diagram showing the data/decision flow, and a one- or two-line intuition or metaphor to make it stick."""

    _DEFAULT_REFINE_PROMPT = """Describe what you want changed.  Examples:

* "Add a glossary of technical terms at the end of each section."
* "Make every section shorter — maximum 200 words."
* "Add more mermaid diagrams."
* "Convert bullet lists into narrative paragraphs."
* "Translate to Spanish."
"""

    _prompt_default = (
        _DEFAULT_REFINE_PROMPT if mode == "refine" else _DEFAULT_REWRITE_PROMPT
    )
    _prompt_help = (
        "Tell the LLM exactly how to edit the existing Markdown. "
        "This is the core instruction for the refine pass."
        if mode == "refine"
        else (
            "These instructions are injected into the prompt alongside "
            "the default rewrite/summarize rules. Clear the box for defaults only."
        )
    )

    custom_prompt = st.text_area(
        "Extra instructions for the LLM"
        if mode != "refine"
        else "Refine instructions (required)",
        value=_prompt_default,
        height=320,
        help=_prompt_help,
    )
    # Treat empty string as None
    custom_prompt = custom_prompt.strip() or None


# ── Source picker (PDF or existing Markdown depending on mode) ──────
selected_path: str | None = None
selected_name: str = ""

if mode == "refine":
    # Show existing rewrite outputs instead of PDFs
    rewrite_outputs = discover_rewrite_outputs(DEFAULT_OUTPUT_DIR)
    if not rewrite_outputs:
        st.warning(
            "No rewrite/summarize outputs found in `output/rewrites/`. "
            "Run a rewrite or summarize first to create Markdown files."
        )
        st.stop()
    md_names = list(rewrite_outputs.keys())
    selected_name = st.selectbox("📄 Markdown to refine", md_names)
    selected_path = rewrite_outputs[selected_name]
else:
    if not structured_pdfs:
        st.warning(
            "No structured PDFs found in `corpus/raw_data/`. "
            "Download the corpus first with `uv run src/ingestion/downloader.py`."
        )
        st.stop()
    pdf_names = list(structured_pdfs.keys())
    selected_name = st.selectbox("📄 PDF to rewrite", pdf_names)
    selected_path = structured_pdfs[selected_name]


# ── Section preview ────────────────────────────────────────────────
if mode == "refine":
    # Parse the markdown to show sections
    with open(selected_path, "r", encoding="utf-8") as _f:
        _md_text = _f.read()
    _md_sections = _split_markdown_sections(_md_text)
    # Group subsections by section
    _sec_map: dict[str, list[str]] = {}
    for _s in _md_sections:
        _sec_map.setdefault(_s["section"], []).append(_s["subsection"])
    section_count = len(_sec_map)
    subsection_count = len(_md_sections)
else:
    sections = _get_section_preview(selected_path)
    _sec_map = sections
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
    for sec, subs in _sec_map.items():
        if subs:
            st.markdown(f"- **{sec}** — _{', '.join(subs)}_")
        else:
            st.markdown(f"- **{sec}**")


# ── Section selection (checkboxes) ─────────────────────────────────
if mode != "refine":
    st.subheader("✅ Select sections/subsections to process")
    st.caption("Only checked items will be rewritten or summarized.")
else:
    st.subheader("📋 Sections in this Markdown")
    st.caption("Refine mode processes all sections in the file.")

selected_sections: dict[str, list[str]] = {}

if mode == "refine":
    # Refine mode: process all sections
    for sec in _sec_map.keys():
        selected_sections[sec] = _sec_map[sec]
else:
    # Rewrite/summarize mode: use checkboxes
    for sec, subs in _sec_map.items():
        sec_key = f"selected_sec::{selected_name}::{sec}"
        sec_checked = st.checkbox(
            sec,
            value=False,
            key=sec_key,
            help="Enable this section, then choose subsections below.",
        )

        if not sec_checked:
            continue

        if not subs:
            selected_sections[sec] = []
            continue

        st.caption(f"Subsections for **{sec}**")
        chosen_subsections: list[str] = []
        for sub in subs:
            sub_key = f"selected_sub::{selected_name}::{sec}::{sub}"
            checked = st.checkbox(
                f"↳ {sub}",
                value=False,
                key=sub_key,
            )
            if checked:
                chosen_subsections.append(sub)

        if chosen_subsections:
            selected_sections[sec] = chosen_subsections

if mode != "refine":
    selected_subsection_count = sum(
        len(subs) if subs else 1 for subs in selected_sections.values()
    )
    st.caption(
        f"Selected: {len(selected_sections)} section(s), "
        f"{selected_subsection_count} subsection target(s)."
    )


# ── Run button ──────────────────────────────────────────────────────
st.divider()

# Persist results across Streamlit re-runs so download URLs stay valid
if "rewrite_result" not in st.session_state:
    st.session_state.rewrite_result = None

_run_label = (
    f"🚀 {mode.title()} this Markdown"
    if mode == "refine"
    else f"🚀 {mode.title()} this PDF"
)

if st.button(
    _run_label,
    type="primary",
    use_container_width=True,
):
    if not model:
        st.error("Please select or enter a model name.")
        st.stop()
    if mode == "refine" and not custom_prompt:
        st.error("Please enter custom instructions for refining.")
        st.stop()
    if mode != "refine" and not selected_sections:
        st.error("Please check at least one section/subsection to process.")
        st.stop()

    progress_bar = st.progress(0, text="Loading sections…")
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
                if mode == "refine":
                    _iter = refine_markdown_iter(
                        selected_path,
                        provider=provider,
                        model=model,
                        temperature=temperature,
                        custom_prompt=custom_prompt,
                        output_dir=DEFAULT_OUTPUT_DIR,
                    )
                else:
                    _iter = rewrite_pdf_iter(
                        selected_path,
                        provider=provider,
                        model=model,
                        temperature=temperature,
                        mode=mode,
                        custom_prompt=custom_prompt,
                        output_dir=DEFAULT_OUTPUT_DIR,
                        selected_sections=selected_sections,
                    )
                for p in _iter:
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

                sec_elapsed = time.time() - section_start if section_start else 0
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
