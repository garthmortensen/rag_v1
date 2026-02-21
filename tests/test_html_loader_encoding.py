import os
import tempfile
import unittest


class TestHtmlLoaderEncoding(unittest.TestCase):
    def test_load_file_handles_non_utf8_html(self):
        from src.ingestion.loaders import load_file

        # cp1252 byte 0xE9 (é) is invalid as a standalone UTF-8 byte.
        html_bytes = b"<html><body>caf\xe9</body></html>"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy.html")
            with open(path, "wb") as f:
                f.write(html_bytes)

            docs = load_file(path)

        self.assertEqual(len(docs), 1)
        self.assertIn("café", docs[0].page_content)
        self.assertEqual(docs[0].metadata.get("source"), path)
