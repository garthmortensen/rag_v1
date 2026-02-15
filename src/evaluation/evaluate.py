"""Automated RAG evaluation using ragas.

Runs each question in the evaluation dataset through the full
RAG pipeline (retrieve + generate), then scores the results on
four quality dimensions:

* **Faithfulness** — does the answer stick to the retrieved context?
* **Response Relevancy** — is the answer relevant to the question?
* **LLM Context Recall** — did retrieval find the chunks needed?
* **Factual Correctness** — does the answer match the ground truth?

All four metrics use an LLM-as-judge (the same provider/model
configured in config.txt).

Usage (CLI)::

    python -m src.evaluation.evaluate
    python -m src.evaluation.evaluate --provider openai --model gpt-4o-mini
    python -m src.evaluation.evaluate --top-k 10 --out results.csv

The results table is printed to stdout and, optionally, saved to CSV.
"""

import argparse
import logging
import sys

import pandas as pd
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics._faithfulness import Faithfulness
from rich.console import Console
from rich.table import Table

from src.evaluation.dataset import EVAL_QUESTIONS
from src.generation.llm import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TOP_K,
    generate_answer,
    get_llm,
)
from src.retrieval.query import retrieve_formatted

logger = logging.getLogger(__name__)
console = Console()


# ── Build ragas evaluation dataset from pipeline outputs ────────────

def build_eval_dataset(
    n_results: int = DEFAULT_TOP_K,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
) -> EvaluationDataset:
    """Run the RAG pipeline for each eval question and collect samples.

    Parameters
    ----------
    n_results : int
        Top-K chunks to retrieve per question.
    model : str
        LLM model name for generation.
    provider : str
        LLM provider name.

    Returns
    -------
    EvaluationDataset
        A ragas dataset ready for ``evaluate()``.
    """
    samples: list[SingleTurnSample] = []

    for i, item in enumerate(EVAL_QUESTIONS, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        console.print(
            f"  [{i}/{len(EVAL_QUESTIONS)}] {question[:70]}…",
            style="dim",
        )

        # Retrieve
        chunks = retrieve_formatted(question, n_results=n_results)
        contexts = [c["text"] for c in chunks]

        # Generate
        try:
            answer = generate_answer(
                question, chunks, model=model, provider=provider,
            )
        except Exception as exc:
            logger.warning(f"Generation failed for Q{i}: {exc}")
            answer = f"[generation error: {exc}]"

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )

    return EvaluationDataset(samples=samples)


# ── Run evaluation ──────────────────────────────────────────────────

def run_evaluation(
    n_results: int = DEFAULT_TOP_K,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Run the full ragas evaluation pipeline.

    Parameters
    ----------
    n_results : int
        Top-K for retrieval.
    model : str
        LLM model name.
    provider : str
        LLM provider.
    output_csv : str | None
        If provided, save results to this CSV path.

    Returns
    -------
    pd.DataFrame
        Per-question scores for each metric.
    """
    console.print("\n[bold cyan]Building evaluation dataset…[/bold cyan]")
    dataset = build_eval_dataset(
        n_results=n_results, model=model, provider=provider,
    )

    # Use the same LLM as the judge
    evaluator_llm = LangchainLLMWrapper(
        get_llm(model=model, provider=provider)
    )

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
        FactualCorrectness(llm=evaluator_llm),
    ]

    console.print("[bold cyan]Scoring with ragas…[/bold cyan]\n")
    result = evaluate(dataset=dataset, metrics=metrics)
    df = result.to_pandas()

    # Pretty-print
    table = Table(
        title="ragas Evaluation Results",
        title_style="bold green",
        border_style="green",
    )
    for col in df.columns:
        table.add_column(str(col), overflow="fold")
    for _, row in df.iterrows():
        table.add_row(*(str(v)[:80] for v in row))
    console.print(table)

    # Summary
    metric_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    if metric_cols:
        console.print("\n[bold]Averages:[/bold]")
        for col in metric_cols:
            try:
                avg = df[col].astype(float).mean()
                console.print(f"  {col}: {avg:.3f}")
            except (ValueError, TypeError):
                pass

    if output_csv:
        df.to_csv(output_csv, index=False)
        console.print(f"\n[dim]Results saved to {output_csv}[/dim]")

    return df


# ── CLI ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.evaluate",
        description="Run ragas evaluation on the RAG pipeline.",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Top-K retrieval (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"LLM model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--provider", type=str, default=DEFAULT_PROVIDER,
        help=f"LLM provider (default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Save results to a CSV file",
    )
    args = parser.parse_args(argv)

    run_evaluation(
        n_results=args.top_k,
        model=args.model,
        provider=args.provider,
        output_csv=args.out,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    main()
