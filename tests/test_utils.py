"""Tests for src/utils.py.

Covers RAM detection, model selection logic (including the 30%
overhead factor), edge cases, and the MemoryError fallback.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import utils


class TestGetAvailableRamMb(unittest.TestCase):
    """Tests for get_available_ram_mb()."""

    @patch("src.utils.psutil.virtual_memory")
    def test_returns_available_in_mb(self, mock_vm):
        mock_vm.return_value = MagicMock(available=2 * 1024 * 1024 * 1024)  # 2 GB
        result = utils.get_available_ram_mb()
        self.assertAlmostEqual(result, 2048.0)

    @patch("src.utils.psutil.virtual_memory")
    def test_returns_float(self, mock_vm):
        mock_vm.return_value = MagicMock(available=1_500_000_000)  # ~1.4 GB
        result = utils.get_available_ram_mb()
        self.assertIsInstance(result, float)


class TestSelectBestModel(unittest.TestCase):
    """Tests for select_best_model()."""

    def test_selects_mpnet_with_plenty_of_ram(self):
        """With 4 GB available, should select the best model (mpnet)."""
        model = utils.select_best_model(available_ram_mb=4096)
        self.assertEqual(model["name"], "all-mpnet-base-v2")

    def test_selects_mpnet_at_exact_threshold(self):
        """mpnet needs 600 * 1.3 = 780 MB.  At exactly 780, it should fit."""
        model = utils.select_best_model(available_ram_mb=780)
        self.assertEqual(model["name"], "all-mpnet-base-v2")

    def test_selects_minilm_l12_when_mpnet_too_large(self):
        """Below 780 MB, mpnet won't fit; L12 needs 260 * 1.3 = 338 MB."""
        model = utils.select_best_model(available_ram_mb=700)
        self.assertEqual(model["name"], "all-MiniLM-L12-v2")

    def test_selects_minilm_l6_when_l12_too_large(self):
        """Below 338 MB, L12 won't fit; L6 needs 200 * 1.3 = 260 MB."""
        model = utils.select_best_model(available_ram_mb=300)
        self.assertEqual(model["name"], "all-MiniLM-L6-v2")

    def test_selects_minilm_l6_at_exact_threshold(self):
        """L6 needs 200 * 1.3 = 260 MB.  At exactly 260, it should fit."""
        model = utils.select_best_model(available_ram_mb=260)
        self.assertEqual(model["name"], "all-MiniLM-L6-v2")

    def test_raises_memory_error_when_too_little_ram(self):
        """Below 260 MB, even L6 won't fit — should raise MemoryError."""
        with self.assertRaises(MemoryError) as ctx:
            utils.select_best_model(available_ram_mb=200)
        self.assertIn("all-MiniLM-L6-v2", str(ctx.exception))

    def test_raises_memory_error_at_zero(self):
        with self.assertRaises(MemoryError):
            utils.select_best_model(available_ram_mb=0)

    def test_overhead_factor_is_1_30(self):
        """Verify the 30% overhead constant."""
        self.assertEqual(utils.OVERHEAD_FACTOR, 1.30)

    def test_returns_dict_with_expected_keys(self):
        model = utils.select_best_model(available_ram_mb=4096)
        for key in ("name", "params_millions", "dimensions", "ram_mb", "description"):
            self.assertIn(key, model)

    @patch("src.utils.get_available_ram_mb", return_value=1024)
    def test_uses_psutil_when_no_override(self, _mock_ram):
        """When available_ram_mb is None, should call get_available_ram_mb()."""
        model = utils.select_best_model()
        _mock_ram.assert_called_once()
        self.assertEqual(model["name"], "all-mpnet-base-v2")


class TestEmbeddingModelsRegistry(unittest.TestCase):
    """Tests for the EMBEDDING_MODELS constant."""

    def test_ordered_largest_to_smallest(self):
        """Models should be ordered from most params to fewest."""
        params = [m["params_millions"] for m in utils.EMBEDDING_MODELS]
        self.assertEqual(params, sorted(params, reverse=True))

    def test_at_least_three_models(self):
        self.assertGreaterEqual(len(utils.EMBEDDING_MODELS), 3)

    def test_all_have_required_fields(self):
        for model in utils.EMBEDDING_MODELS:
            self.assertIn("name", model)
            self.assertIn("ram_mb", model)
            self.assertIn("dimensions", model)
            self.assertIn("params_millions", model)


class TestLogSystemInfo(unittest.TestCase):
    """Tests for log_system_info()."""

    @patch("src.utils.psutil.virtual_memory")
    def test_runs_without_error(self, mock_vm):
        mock_vm.return_value = MagicMock(
            total=8 * 1024**3,
            available=4 * 1024**3,
            percent=50.0,
        )
        # Should not raise
        utils.log_system_info()


if __name__ == "__main__":
    unittest.main()
