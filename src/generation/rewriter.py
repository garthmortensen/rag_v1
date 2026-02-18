"""Rewrite structured PDFs section-by-section in plain English.

Uses the section-aware PDF splitter to preserve document structure,
then sends each subsection through the configured LLM with a
simplification prompt.  Output is a Markdown file with the original
section / subsection hierarchy intact.

Supports two modes:

* **rewrite** — full plain-English rewrite of every section
* **summarize** — condensed 2–3 sentence summary per section

Usage (CLI)::

    uv run python -m src.generation.rewriter \\
        corpus/raw_data/proposed_stress_test_model_documentation_credit_risk_models.pdf

    uv run python -m src.generation.rewriter \\
        corpus/raw_data/proposed_stress_test_model_documentation_credit_risk_models.pdf \\
        --provider openai --model gpt-4.1 --mode summarize

Usage (programmatic)::

    from src.generation.rewriter import rewrite_pdf
    path = rewrite_pdf("corpus/raw_data/credit_risk_models.pdf")
"""

import logging
import os
import textwrap
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

from src.config import CFG
from src.generation.llm import get_llm
from src.ingestion.pdf_section_splitter import (
    has_section_headers,
    load_pdf_by_section,
)
from src.retrieval.query_logger import (
    log_rewrite_session,
    RewriteSectionLog,
)

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────
DEFAULT_PROVIDER: str = str(CFG.get("llm_provider", "ollama"))
DEFAULT_MODEL: str = str(CFG.get("llm_model", "llama3.2:3b"))
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "output", "rewrites"
)
MAX_CHUNK_CHARS = 6000  # keep well within any model's context window

# ── Prompts ─────────────────────────────────────────────────────────

REWRITE_PROMPT = textwrap.dedent("""\
    You are a plain-language editor.  Rewrite the following passage
    in simple, clear English that a non-expert could understand.

    RULES:
    1. Preserve ALL factual content, numbers, and key terms.
    2. Define any technical jargon when first used.
    3. Use short sentences and paragraphs.
    4. Do NOT add information that is not in the original text.
    5. Include a mermaid diagram if it would help explain the content. Put the diagram before the related text. but avoid html tags and < or > signs
    {custom_instructions}
    ORIGINAL TEXT:
    {text}

    PLAIN-ENGLISH REWRITE:
""")

SUMMARIZE_PROMPT = textwrap.dedent("""\
    You are a plain-language editor.  Summarize the following passage
    in 2–3 sentences that a non-expert could understand.  Preserve
    key facts.
    Include a mermaid diagram where appropriate, but avoid html tags and < or > signs.
                                   {custom_instructions}
    ORIGINAL TEXT:
    {text}

    SUMMARY:
""")

SECTION_SUMMARY_PROMPT = textwrap.dedent("""\
    Summarize the following rewritten passage in 2–3 sentences.
    Keep it simple and preserve key facts and numbers.
    Include a mermaid diagram where appropriate, but avoid html tags and < or > signs.

    TEXT:
    {text}

    SUMMARY:
""")

REFINE_PROMPT = textwrap.dedent("""\
    You are a skilled editor.  Apply the instructions below to the
    following Markdown section.  Preserve the Markdown formatting.

    INSTRUCTIONS:
    {custom_instructions}

    ORIGINAL MARKDOWN:
    {text}

    REFINED MARKDOWN:
""")


def _build_prompt(mode: str, custom_prompt: str | None = None) -> str:
    """Return the prompt template for the given mode.

    If *custom_prompt* is provided it is injected as an
    ``ADDITIONAL INSTRUCTIONS`` block inside the template.

    For ``"refine"`` mode the custom prompt is mandatory and becomes
    the core editing instruction inside :data:`REFINE_PROMPT`.
    """
    if mode == "refine":
        return REFINE_PROMPT.replace(
            "{custom_instructions}",
            (custom_prompt or "").strip() or "Improve clarity and formatting.",
        )

    base = REWRITE_PROMPT if mode == "rewrite" else SUMMARIZE_PROMPT
    if custom_prompt:
        instructions = (
            f"\nADDITIONAL INSTRUCTIONS:\n{custom_prompt.strip()}\n"
        )
    else:
        instructions = ""
    return base.replace("{custom_instructions}", instructions)


def _write_config_file(
    config_path: str,
    *,
    filepath: str,
    provider: str,
    model: str,
    temperature: float,
    mode: str,
    custom_prompt: str | None,
    num_sections: int,
    prompt_template: str,
    selected_sections: dict[str, list[str]] | None = None,
) -> None:
    """Write rewrite session settings to a config.txt file."""
    lines = [
        f"source_pdf: {os.path.basename(filepath)}",
        f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"provider: {provider}",
        f"model: {model}",
        f"temperature: {temperature}",
        f"mode: {mode}",
        f"sections: {num_sections}",
        "",
    ]
    if custom_prompt:
        lines.append(f"custom_instructions: {custom_prompt.strip()}")
        lines.append("")
    if selected_sections:
        lines.append("selected_sections:")
        for section, subsections in selected_sections.items():
            if subsections:
                lines.append(f"  - {section}: {', '.join(subsections)}")
            else:
                lines.append(f"  - {section}: (all subsections)")
        lines.append("")
    lines.append("--- prompt template ---")
    lines.append(prompt_template.strip())
    lines.append("")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _filter_docs_by_selection(
    docs,
    selected_sections: dict[str, list[str]] | None,
):
    """Filter section Documents by selected section/subsection names.

    Parameters
    ----------
    docs
        Iterable of langchain Document items with ``section`` and
        ``subsection`` metadata.
    selected_sections : dict[str, list[str]] | None
        Mapping of section name to selected subsection names.
        If a section maps to an empty list, all subsections of that
        section are included.
    """
    if not selected_sections:
        return list(docs)

    filtered = []
    for doc in docs:
        section = doc.metadata.get("section", "Unknown")
        subsection = doc.metadata.get("subsection", "Unknown")

        if section not in selected_sections:
            continue

        wanted_subsections = selected_sections[section]
        if not wanted_subsections or subsection in wanted_subsections:
            filtered.append(doc)

    return filtered


# ── Public API ──────────────────────────────────────────────────────


def rewrite_pdf(
    filepath: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    mode: str = "rewrite",
    custom_prompt: str | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    selected_sections: dict[str, list[str]] | None = None,
) -> str:
    """Rewrite a structured PDF in plain English, section by section.

    Parameters
    ----------
    filepath : str
        Path to a PDF with ``Model Documentation:`` page headers.
    provider : str, optional
        LLM provider (defaults to config.txt value).
    model : str, optional
        Model name (defaults to config.txt value).
    temperature : float
        Sampling temperature — slightly higher than RAG default for
        more natural prose (default 0.3).
    mode : str
        ``"rewrite"`` for full plain-English rewrite,
        ``"summarize"`` for a condensed summary per section.
    custom_prompt : str, optional
        Extra instructions injected into the prompt, e.g.
        ``"Write at an elementary-school reading level."``
    output_dir : str
        Directory for the output Markdown file.
    selected_sections : dict[str, list[str]], optional
        Optional filter of sections/subsections to process.
        Keys are section names; values are subsection names. If the
        subsection list is empty, all subsections in that section are
        included.

    Returns
    -------
    str
        Absolute path to the generated Markdown file.

    Raises
    ------
    ValueError
        If the PDF does not have section headers.
    FileNotFoundError
        If *filepath* does not exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"PDF not found: {filepath}")

    if not has_section_headers(filepath):
        raise ValueError(
            f"{filepath} does not have section headers. "
            "Only structured model documentation PDFs are supported."
        )

    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODEL
    llm = get_llm(provider=provider, model=model, temperature=temperature)

    prompt_template = _build_prompt(mode, custom_prompt)

    # ── Load section-split documents ────────────────────────────────
    docs = load_pdf_by_section(filepath)
    docs = _filter_docs_by_selection(docs, selected_sections)
    if not docs:
        raise ValueError(
            "No matching sections were selected. "
            "Please choose at least one valid section/subsection."
        )
    logger.info(
        f"Rewriting {os.path.basename(filepath)}: "
        f"{len(docs)} section(s) via {provider}/{model} (mode={mode})"
    )

    # ── Build output directory (YYYYMMDD_HHMM subfolder) ───────────
    basename = os.path.splitext(os.path.basename(filepath))[0]
    title = basename.replace("_", " ").title()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    out_path = os.path.join(run_dir, f"{basename}_{mode}.md")
    out_path = os.path.abspath(out_path)

    # ── Write config.txt with prompt & settings ─────────────────────
    config_path = os.path.join(run_dir, "config.txt")
    _write_config_file(
        config_path,
        filepath=filepath,
        provider=provider,
        model=model,
        temperature=temperature,
        mode=mode,
        custom_prompt=custom_prompt,
        num_sections=len(docs),
        prompt_template=prompt_template,
        selected_sections=selected_sections,
    )

    # ── Build clean Markdown header ─────────────────────────────────
    header_lines: list[str] = [
        f"# {title}",
        "",
        "---",
        "",
    ]

    current_section: str | None = None
    section_logs: list[RewriteSectionLog] = []
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header_lines))

        for i, doc in enumerate(docs, 1):
            section = doc.metadata.get("section", "Unknown")
            subsection = doc.metadata.get("subsection", "Unknown")
            start_page = doc.metadata.get("start_page", "?")
            end_page = doc.metadata.get("end_page", "?")

            # Section heading (only when section changes)
            if section != current_section:
                current_section = section
                fh.write(f"## {section}\n\n")

            # Subsection heading
            fh.write(f"### {subsection}\n")
            fh.write(
                f"*Pages {start_page}–{end_page} | "
                f"{len(doc.page_content):,} chars*\n\n"
            )

            # Rewrite via LLM
            logger.info(
                f"  [{i}/{len(docs)}] {section} > {subsection} "
                f"({len(doc.page_content):,} chars)"
            )
            rewritten = _rewrite_chunk(llm, prompt_template, doc.page_content)

            # Place a brief summary at the top of each rewritten section
            if mode == "rewrite":
                summary = _invoke_llm(llm, SECTION_SUMMARY_PROMPT, rewritten)
                fh.write(f"> **Summary:** {summary}\n\n")
                rewritten = f"> **Summary:** {summary}\n\n{rewritten}"

            fh.write(rewritten)
            fh.write("\n\n---\n\n")
            fh.flush()

            section_logs.append(RewriteSectionLog(
                section=section,
                subsection=subsection,
                source_chars=len(doc.page_content),
                rewritten_chars=len(rewritten),
                rewritten_text=rewritten,
            ))

    elapsed = time.time() - t0
    logger.info(f"✅ Wrote {out_path}")

    # ── Log the session ─────────────────────────────────────────────
    log_rewrite_session(
        pdf_path=filepath,
        mode=mode,
        provider=provider,
        model=model,
        temperature=temperature,
        custom_prompt=custom_prompt,
        sections=section_logs,
        output_path=out_path,
        elapsed_seconds=elapsed,
    )

    return out_path


# ── Progress-yielding variant (for Streamlit UI) ───────────────────


@dataclass
class RewriteProgress:
    """Progress update yielded by :func:`rewrite_pdf_iter`."""

    step: int          # 1-based index of the current section
    total: int         # total number of sections
    section: str       # top-level section name
    subsection: str    # subsection name being processed
    chars: int         # character count of the source text
    rewritten: str     # the rewritten text for this section
    phase: str         # "loading" | "rewriting" | "done"
    output_path: str | None = None  # set only when phase == "done"


def rewrite_pdf_iter(
    filepath: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    mode: str = "rewrite",
    custom_prompt: str | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    selected_sections: dict[str, list[str]] | None = None,
) -> Iterator[RewriteProgress]:
    """Like :func:`rewrite_pdf`, but yields progress after each section.

    This is the API consumed by the Streamlit UI to show per-section
    progress bars and live output.

    Yields
    ------
    RewriteProgress
        One update per section, plus a final ``phase="done"`` update.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"PDF not found: {filepath}")

    if not has_section_headers(filepath):
        raise ValueError(
            f"{filepath} does not have section headers. "
            "Only structured model documentation PDFs are supported."
        )

    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODEL
    llm = get_llm(provider=provider, model=model, temperature=temperature)

    prompt_template = _build_prompt(mode, custom_prompt)

    docs = load_pdf_by_section(filepath)
    docs = _filter_docs_by_selection(docs, selected_sections)
    if not docs:
        raise ValueError(
            "No matching sections were selected. "
            "Please choose at least one valid section/subsection."
        )

    # ── Build output directory (YYYYMMDD_HHMM subfolder) ───────────
    basename = os.path.splitext(os.path.basename(filepath))[0]
    title = basename.replace("_", " ").title()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    out_path = os.path.join(run_dir, f"{basename}_{mode}.md")
    out_path = os.path.abspath(out_path)

    # ── Write config.txt with prompt & settings ─────────────────────
    config_path = os.path.join(run_dir, "config.txt")
    _write_config_file(
        config_path,
        filepath=filepath,
        provider=provider,
        model=model,
        temperature=temperature,
        mode=mode,
        custom_prompt=custom_prompt,
        num_sections=len(docs),
        prompt_template=prompt_template,
        selected_sections=selected_sections,
    )

    # ── Build clean Markdown header ─────────────────────────────────
    header_lines: list[str] = [
        f"# {title}",
        "",
        "---",
        "",
    ]

    current_section: str | None = None
    section_logs: list[RewriteSectionLog] = []
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header_lines))

        for i, doc in enumerate(docs, 1):
            section = doc.metadata.get("section", "Unknown")
            subsection = doc.metadata.get("subsection", "Unknown")
            start_page = doc.metadata.get("start_page", "?")
            end_page = doc.metadata.get("end_page", "?")

            if section != current_section:
                current_section = section
                fh.write(f"## {section}\n\n")

            fh.write(f"### {subsection}\n")
            fh.write(
                f"*Pages {start_page}–{end_page} | "
                f"{len(doc.page_content):,} chars*\n\n"
            )

            # Signal that we're about to process this section
            yield RewriteProgress(
                step=i,
                total=len(docs),
                section=section,
                subsection=subsection,
                chars=len(doc.page_content),
                rewritten="",
                phase="processing",
            )

            rewritten = _rewrite_chunk(llm, prompt_template, doc.page_content)

            # Place a brief summary at the top of each rewritten section
            if mode == "rewrite":
                summary = _invoke_llm(llm, SECTION_SUMMARY_PROMPT, rewritten)
                fh.write(f"> **Summary:** {summary}\n\n")
                rewritten = f"> **Summary:** {summary}\n\n{rewritten}"

            fh.write(rewritten)
            fh.write("\n\n---\n\n")
            fh.flush()

            section_logs.append(RewriteSectionLog(
                section=section,
                subsection=subsection,
                source_chars=len(doc.page_content),
                rewritten_chars=len(rewritten),
                rewritten_text=rewritten,
            ))

            yield RewriteProgress(
                step=i,
                total=len(docs),
                section=section,
                subsection=subsection,
                chars=len(doc.page_content),
                rewritten=rewritten,
                phase="rewriting",
            )

    elapsed = time.time() - t0

    # ── Log the session ─────────────────────────────────────────────
    log_rewrite_session(
        pdf_path=filepath,
        mode=mode,
        provider=provider,
        model=model,
        temperature=temperature,
        custom_prompt=custom_prompt,
        sections=section_logs,
        output_path=out_path,
        elapsed_seconds=elapsed,
    )

    yield RewriteProgress(
        step=len(docs),
        total=len(docs),
        section="",
        subsection="",
        chars=0,
        rewritten="",
        phase="done",
        output_path=out_path,
    )


# ── Markdown refine helpers ────────────────────────────────────────


def _split_markdown_sections(text: str) -> list[dict[str, str]]:
    """Split a Markdown document into sections on ``## `` headings.

    Returns a list of dicts with keys ``section``, ``subsection``,
    and ``body``.  If a ``### `` sub-heading is found inside a
    ``## `` block it becomes the *subsection*; otherwise subsection
    is ``"(full section)"``.

    Any text before the first ``## `` heading is returned as a
    ``(preamble)`` section.
    """
    sections: list[dict[str, str]] = []
    current_section = "(preamble)"
    current_subsection = "(intro)"
    current_body: list[str] = []

    for line in text.splitlines(keepends=True):
        if line.startswith("## ") and not line.startswith("### "):
            # Flush previous
            if current_body:
                body_text = "".join(current_body).strip()
                if body_text and body_text.replace("-", "").strip():
                    sections.append({
                        "section": current_section,
                        "subsection": current_subsection,
                        "body": body_text,
                    })
                current_body = []
            section_name = line.lstrip("# ").strip()
            # Skip synthetic section names from the parser itself
            if section_name not in ("(preamble)", "(intro)"):
                current_section = section_name
                current_subsection = "(full section)"
            continue

        if line.startswith("### "):
            # Flush previous subsection
            if current_body:
                body_text = "".join(current_body).strip()
                if body_text and body_text.replace("-", "").strip():
                    sections.append({
                        "section": current_section,
                        "subsection": current_subsection,
                        "body": body_text,
                    })
                current_body = []
            subsection_name = line.lstrip("# ").strip()
            # Skip synthetic subsection names
            if subsection_name not in ("(intro)", "(full section)"):
                current_subsection = subsection_name
            continue

        current_body.append(line)

    # Flush last section
    if current_body:
        body_text = "".join(current_body).strip()
        if body_text and body_text.replace("-", "").strip():
            sections.append({
                "section": current_section,
                "subsection": current_subsection,
                "body": body_text,
            })

    return sections


def discover_rewrite_outputs(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict[str, str]:
    """Return ``{display_name: filepath}`` for existing ``.md`` outputs.

    Scans ``output_dir`` for timestamped sub-directories and any
    top-level ``.md`` files produced by previous rewrite / summarize
    runs.
    """
    import glob as _glob

    results: dict[str, str] = {}
    if not os.path.isdir(output_dir):
        return results

    # Timestamped sub-dirs (YYYYMMDD_HHMM/<name>.md)
    for md in sorted(
        _glob.glob(os.path.join(output_dir, "*", "*.md")), reverse=True
    ):
        dirname = os.path.basename(os.path.dirname(md))
        fname = os.path.basename(md)
        label = f"{dirname} / {fname}"
        results[label] = md

    # Top-level .md files
    for md in sorted(
        _glob.glob(os.path.join(output_dir, "*.md")), reverse=True
    ):
        results[os.path.basename(md)] = md

    return results


def refine_markdown_iter(
    md_path: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    custom_prompt: str | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Iterator[RewriteProgress]:
    """Refine an existing Markdown file section-by-section.

    Reads *md_path*, splits on ``## `` / ``### `` headings, sends
    each section through the LLM with the user's *custom_prompt*,
    and writes a new ``.md`` file in a timestamped sub-directory.

    Yields :class:`RewriteProgress` updates for the Streamlit UI.
    """
    if not os.path.isfile(md_path):
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    with open(md_path, "r", encoding="utf-8") as fh:
        md_text = fh.read()

    sections = _split_markdown_sections(md_text)
    if not sections:
        raise ValueError("No sections found in the Markdown file.")

    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODEL
    llm = get_llm(provider=provider, model=model, temperature=temperature)

    prompt_template = _build_prompt("refine", custom_prompt)

    # ── Build output directory ──────────────────────────────────────
    basename = os.path.splitext(os.path.basename(md_path))[0]
    # Strip a trailing _rewrite / _summarize / _refine tag if present
    for suffix in ("_rewrite", "_summarize", "_refine"):
        if basename.endswith(suffix):
            basename = basename[: -len(suffix)]
            break
    title = basename.replace("_", " ").title()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    out_path = os.path.abspath(
        os.path.join(run_dir, f"{basename}_refine.md")
    )

    # ── Write config.txt ────────────────────────────────────────────
    config_path = os.path.join(run_dir, "config.txt")
    _write_config_file(
        config_path,
        filepath=md_path,
        provider=provider,
        model=model,
        temperature=temperature,
        mode="refine",
        custom_prompt=custom_prompt,
        num_sections=len(sections),
        prompt_template=prompt_template,
    )

    # ── Process each section ────────────────────────────────────────
    header_lines = [f"# {title}", "", "---", ""]
    current_section: str | None = None
    section_logs: list[RewriteSectionLog] = []
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header_lines))

        for i, sec in enumerate(sections, 1):
            section = sec["section"]
            subsection = sec["subsection"]
            body = sec["body"]

            if section != current_section:
                current_section = section
                fh.write(f"## {section}\n\n")

            fh.write(f"### {subsection}\n\n")

            yield RewriteProgress(
                step=i,
                total=len(sections),
                section=section,
                subsection=subsection,
                chars=len(body),
                rewritten="",
                phase="processing",
            )

            rewritten = _rewrite_chunk(llm, prompt_template, body)

            fh.write(rewritten)
            fh.write("\n\n---\n\n")
            fh.flush()

            section_logs.append(RewriteSectionLog(
                section=section,
                subsection=subsection,
                source_chars=len(body),
                rewritten_chars=len(rewritten),
                rewritten_text=rewritten,
            ))

            yield RewriteProgress(
                step=i,
                total=len(sections),
                section=section,
                subsection=subsection,
                chars=len(body),
                rewritten=rewritten,
                phase="rewriting",
            )

    elapsed = time.time() - t0

    log_rewrite_session(
        pdf_path=md_path,
        mode="refine",
        provider=provider,
        model=model,
        temperature=temperature,
        custom_prompt=custom_prompt,
        sections=section_logs,
        output_path=out_path,
        elapsed_seconds=elapsed,
    )

    yield RewriteProgress(
        step=len(sections),
        total=len(sections),
        section="",
        subsection="",
        chars=0,
        rewritten="",
        phase="done",
        output_path=out_path,
    )


# ── Private helpers ─────────────────────────────────────────────────


def _rewrite_chunk(llm, prompt_template: str, text: str) -> str:
    """Rewrite a single chunk of text via the LLM.

    If *text* exceeds :data:`MAX_CHUNK_CHARS`, it is split on
    paragraph boundaries, each piece is rewritten independently,
    and the results are joined.

    Parameters
    ----------
    llm
        A LangChain chat model returned by :func:`get_llm`.
    prompt_template : str
        A format string with a ``{text}`` placeholder.
    text : str
        The raw section/subsection text to rewrite.

    Returns
    -------
    str
        The rewritten text.
    """
    if len(text) <= MAX_CHUNK_CHARS:
        return _invoke_llm(llm, prompt_template, text)

    # Split into blocks that fit within the context window
    blocks = _split_on_paragraphs(text, MAX_CHUNK_CHARS)
    logger.debug(
        f"  Split {len(text):,} chars into {len(blocks)} block(s)"
    )
    rewritten_parts = [_invoke_llm(llm, prompt_template, block) for block in blocks]
    return "\n\n".join(rewritten_parts)


def _invoke_llm(llm, prompt_template: str, text: str) -> str:
    """Send a single prompt to the LLM and return the response text."""
    prompt = prompt_template.format(text=text)
    response = llm.invoke(prompt)
    # LangChain chat models return AIMessage; plain strings from some
    # wrappers — handle both.  Anthropic models may return a list of
    # content blocks rather than a plain string.
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", str(block))
                if isinstance(block, dict)
                else str(block)
                for block in content
            )
        return content.strip()
    return str(response).strip()


def _split_on_paragraphs(text: str, max_chars: int) -> list[str]:
    """Split *text* into blocks of at most *max_chars* on paragraph boundaries.

    Paragraphs are delimited by blank lines (``\\n\\n``).  If a single
    paragraph exceeds *max_chars* it is included as-is (the LLM will
    truncate internally if necessary).

    Returns
    -------
    list[str]
        Non-empty text blocks.
    """
    paragraphs = text.split("\n\n")
    blocks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            blocks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        blocks.append("\n\n".join(current))

    return [b for b in blocks if b.strip()]


# ── CLI entry point ─────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run the rewriter."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Rewrite a structured PDF in plain English",
    )
    parser.add_argument(
        "filepath",
        help="Path to a structured model-documentation PDF",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider (default: config.txt)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: config.txt)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)",
    )
    parser.add_argument(
        "--mode",
        choices=["rewrite", "summarize", "refine"],
        default="rewrite",
        help="Full rewrite, condensed summary, or refine existing markdown (default: rewrite)",
    )
    parser.add_argument(
        "--custom-prompt",
        default=None,
        help='Extra instructions for the LLM, e.g. "Write at an elementary-school reading level."',
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: output/rewrites)",
    )
    args = parser.parse_args()

    out = rewrite_pdf(
        args.filepath,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        mode=args.mode,
        custom_prompt=args.custom_prompt,
        output_dir=args.output_dir,
    )
    print(f"\n✅ Written to: {out}")


if __name__ == "__main__":
    main()
