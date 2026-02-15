"""Evaluation dataset for ragas scoring.

A curated set of question / ground-truth pairs drawn from the
stress-testing corpus.  Each entry has:

* **question** — a natural-language query
* **ground_truth** — the expected factual answer (used to measure
  context recall)

Expand this list whenever you add new documents to the corpus or
want to test a new retrieval / generation behaviour.
"""

EVAL_QUESTIONS: list[dict[str, str]] = [
    {
        "question": (
            "What is the peak unemployment rate in the severely "
            "adverse scenario?"
        ),
        "ground_truth": (
            "The unemployment rate peaks at 10 percent in the "
            "severely adverse scenario."
        ),
    },
    {
        "question": (
            "What is the real GDP growth rate in the baseline scenario?"
        ),
        "ground_truth": (
            "The baseline scenario projects positive real GDP growth "
            "consistent with moderate economic expansion."
        ),
    },
    {
        "question": (
            "What are the transparency proposals for the stress tests?"
        ),
        "ground_truth": (
            "The Federal Reserve proposed enhanced transparency "
            "measures including public disclosure of stress test "
            "models and scenarios to improve accountability."
        ),
    },
    {
        "question": "What is the CET1 capital ratio minimum?",
        "ground_truth": (
            "Banks must maintain a minimum Common Equity Tier 1 "
            "(CET1) capital ratio as part of the stress test "
            "capital requirements."
        ),
    },
    {
        "question": (
            "What is the House Price Index decline in the severely "
            "adverse scenario?"
        ),
        "ground_truth": (
            "The severely adverse scenario projects a significant "
            "decline in the House Price Index."
        ),
    },
    {
        "question": (
            "What does the Dodd-Frank Act require for stress testing?"
        ),
        "ground_truth": (
            "The Dodd-Frank Act requires the Federal Reserve to "
            "conduct annual stress tests on large bank holding "
            "companies to assess their capital adequacy under "
            "adverse economic conditions."
        ),
    },
    {
        "question": (
            "How are credit risk losses estimated in the stress tests?"
        ),
        "ground_truth": (
            "Credit risk losses are estimated using models that "
            "project loan losses under stress scenarios based on "
            "macroeconomic variables and loan-level characteristics."
        ),
    },
    {
        "question": (
            "What is the global market shock component?"
        ),
        "ground_truth": (
            "The global market shock is an additional component "
            "applied to firms with significant trading or "
            "counterparty exposures, modeling instantaneous "
            "market price changes."
        ),
    },
]
