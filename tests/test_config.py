"""Tests for src/config.py.

Covers the key=value parser, defaults when config.txt is missing,
integer auto-casting, comment/blank-line handling, and malformed lines.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config, DEFAULTS


class TestLoadConfigDefaults(unittest.TestCase):
    """When the config file is missing, defaults should be returned."""

    def test_returns_defaults_when_file_missing(self):
        cfg = load_config("/nonexistent/config.txt")
        self.assertEqual(cfg, DEFAULTS)

    def test_default_chunk_size_is_int(self):
        cfg = load_config("/nonexistent/config.txt")
        self.assertIsInstance(cfg["chunk_size"], int)

    def test_default_collection_name(self):
        cfg = load_config("/nonexistent/config.txt")
        self.assertEqual(cfg["collection_name"], "stress_test_docs_1k")


class TestLoadConfigParsing(unittest.TestCase):
    """Verify parsing of well-formed and edge-case config files."""

    def _write_temp(self, content: str) -> str:
        """Write *content* to a temp file and return its path."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
        self.addCleanup(os.unlink, path)
        return path

    def test_parses_key_value_pairs(self):
        path = self._write_temp("chunk_size = 6000\ncollection_name = big_chunks\n")
        cfg = load_config(path)
        self.assertEqual(cfg["chunk_size"], 6000)
        self.assertEqual(cfg["collection_name"], "big_chunks")

    def test_integer_auto_cast(self):
        path = self._write_temp("chunk_size = 500\nchunk_overlap = 50\n")
        cfg = load_config(path)
        self.assertIsInstance(cfg["chunk_size"], int)
        self.assertIsInstance(cfg["chunk_overlap"], int)

    def test_string_values_stay_strings(self):
        path = self._write_temp("collection_name = my_collection\n")
        cfg = load_config(path)
        self.assertIsInstance(cfg["collection_name"], str)

    def test_comments_and_blank_lines_are_ignored(self):
        content = "# this is a comment\n\nchunk_size = 2000\n\n# another\n"
        path = self._write_temp(content)
        cfg = load_config(path)
        self.assertEqual(cfg["chunk_size"], 2000)
        # defaults still present for keys not in file
        self.assertEqual(cfg["collection_name"], "stress_test_docs_1k")

    def test_malformed_line_is_skipped(self):
        content = "chunk_size = 3000\nno_equals_here\ncollection_name = ok\n"
        path = self._write_temp(content)
        cfg = load_config(path)
        self.assertEqual(cfg["chunk_size"], 3000)
        self.assertEqual(cfg["collection_name"], "ok")

    def test_whitespace_around_key_and_value_is_stripped(self):
        path = self._write_temp("  chunk_size   =   4000   \n")
        cfg = load_config(path)
        self.assertEqual(cfg["chunk_size"], 4000)

    def test_extra_keys_are_preserved(self):
        path = self._write_temp("custom_key = hello\n")
        cfg = load_config(path)
        self.assertEqual(cfg["custom_key"], "hello")

    def test_empty_file_returns_defaults(self):
        path = self._write_temp("")
        cfg = load_config(path)
        self.assertEqual(cfg, DEFAULTS)

    def test_value_with_equals_sign(self):
        """Values containing '=' should keep everything after the first '='."""
        path = self._write_temp("collection_name = a=b=c\n")
        cfg = load_config(path)
        self.assertEqual(cfg["collection_name"], "a=b=c")

    def test_bool_true_variants(self):
        """true, yes, 1, on (case-insensitive) should all parse to True."""
        for word in ("true", "True", "TRUE", "yes", "Yes", "1", "on", "ON"):
            path = self._write_temp(f"beep_on_answer = {word}\n")
            cfg = load_config(path)
            self.assertIs(cfg["beep_on_answer"], True, f"Failed for {word!r}")

    def test_bool_false_variants(self):
        """false, no, 0, off (case-insensitive) should all parse to False."""
        for word in ("false", "False", "FALSE", "no", "No", "0", "off", "OFF"):
            path = self._write_temp(f"beep_on_answer = {word}\n")
            cfg = load_config(path)
            self.assertIs(cfg["beep_on_answer"], False, f"Failed for {word!r}")

    def test_default_beep_on_answer(self):
        """beep_on_answer should default to True when not in config file."""
        cfg = load_config("/nonexistent/config.txt")
        self.assertIs(cfg["beep_on_answer"], True)

    def test_default_llm_provider(self):
        """llm_provider should default to 'ollama' when not in config file."""
        cfg = load_config("/nonexistent/config.txt")
        self.assertEqual(cfg["llm_provider"], "ollama")

    def test_default_llm_model(self):
        """llm_model should default to 'llama3.2:3b' when not in config file."""
        cfg = load_config("/nonexistent/config.txt")
        self.assertEqual(cfg["llm_model"], "llama3.2:3b")

    def test_custom_llm_provider_and_model(self):
        """llm_provider and llm_model can be overridden in the config file."""
        path = self._write_temp(
            "llm_provider = openai\nllm_model = gpt-4o-mini\n"
        )
        cfg = load_config(path)
        self.assertEqual(cfg["llm_provider"], "openai")
        self.assertEqual(cfg["llm_model"], "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
