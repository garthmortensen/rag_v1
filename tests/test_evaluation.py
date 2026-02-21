"""Tests for src/evaluation.

Verifies the evaluation dataset and build_eval_dataset() without
requiring a live LLM or vector DB.

Run with:
    pytest tests/test_evaluation.py -v
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.dataset import EVAL_QUESTIONS
from src.evaluation.evaluate import build_eval_dataset


# ── Dataset tests ───────────────────────────────────────────────────


class TestEvalDataset(unittest.TestCase):
    """Tests for the curated evaluation question set."""

    def test_dataset_is_not_empty(self):
        """There should be at least one evaluation question."""
        self.assertGreater(len(EVAL_QUESTIONS), 0)

    def test_each_entry_has_question_and_ground_truth(self):
        """Every entry must have 'question' and 'ground_truth' keys."""
        for i, item in enumerate(EVAL_QUESTIONS):
            self.assertIn("question", item, f"Item {i} missing 'question'")
            self.assertIn("ground_truth", item, f"Item {i} missing 'ground_truth'")

    def test_questions_are_non_empty_strings(self):
        for i, item in enumerate(EVAL_QUESTIONS):
            self.assertIsInstance(item["question"], str)
            self.assertTrue(
                len(item["question"].strip()) > 0, f"Item {i} has empty question"
            )

    def test_ground_truths_are_non_empty_strings(self):
        for i, item in enumerate(EVAL_QUESTIONS):
            self.assertIsInstance(item["ground_truth"], str)
            self.assertTrue(
                len(item["ground_truth"].strip()) > 0,
                f"Item {i} has empty ground_truth",
            )


# ── build_eval_dataset tests ───────────────────────────────────────


class TestBuildEvalDataset(unittest.TestCase):
    """Tests for the eval dataset builder (mocked, no live LLM/DB)."""

    @patch("src.evaluation.evaluate.generate_answer", return_value="Mocked answer.")
    @patch("src.evaluation.evaluate.retrieve_formatted")
    def test_builds_samples_for_each_question(self, mock_retrieve, mock_gen):
        """build_eval_dataset() creates one sample per EVAL_QUESTIONS entry."""
        mock_retrieve.return_value = [
            {
                "rank": 1,
                "id": "chunk_0",
                "distance": 0.1,
                "text": "Some context text.",
                "metadata": {"source": "test.pdf", "title": "Test"},
            }
        ]

        dataset = build_eval_dataset()
        self.assertEqual(len(dataset.samples), len(EVAL_QUESTIONS))

    @patch("src.evaluation.evaluate.generate_answer", return_value="Mocked answer.")
    @patch("src.evaluation.evaluate.retrieve_formatted")
    def test_samples_have_required_fields(self, mock_retrieve, mock_gen):
        """Each SingleTurnSample has user_input, response, retrieved_contexts, reference."""
        mock_retrieve.return_value = [
            {
                "rank": 1,
                "id": "chunk_0",
                "distance": 0.1,
                "text": "Context.",
                "metadata": {"source": "test.pdf", "title": "Test"},
            }
        ]

        dataset = build_eval_dataset()
        for sample in dataset.samples:
            self.assertIsNotNone(sample.user_input)
            self.assertIsNotNone(sample.response)
            self.assertIsNotNone(sample.retrieved_contexts)
            self.assertIsNotNone(sample.reference)

    @patch("src.evaluation.evaluate.generate_answer", side_effect=Exception("fail"))
    @patch("src.evaluation.evaluate.retrieve_formatted", return_value=[])
    def test_handles_generation_errors_gracefully(self, mock_retrieve, mock_gen):
        """If generation fails, the sample still has an error placeholder."""
        dataset = build_eval_dataset()
        for sample in dataset.samples:
            self.assertIn("[generation error", sample.response)


if __name__ == "__main__":
    unittest.main()
