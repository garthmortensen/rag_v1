import os
import sys
import tempfile
import unittest
from unittest.mock import patch, mock_open, MagicMock

# Ensure we can import the module from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion import downloader


class TestDownloadData(unittest.TestCase):
    def test_sanitize_filename(self):
        self.assertEqual(downloader.sanitize_filename("Simple Name"), "simple_name")
        self.assertEqual(
            downloader.sanitize_filename("Name with @#$%^& chars"), "name_with_chars"
        )
        self.assertEqual(
            downloader.sanitize_filename("   Trailing Spaces   "), "trailing_spaces"
        )
        self.assertEqual(
            downloader.sanitize_filename("multiple__underscores"),
            "multiple_underscores",
        )
        self.assertEqual(
            downloader.sanitize_filename("dashes-and-spaces"), "dashes_and_spaces"
        )

    def test_construct_filepath(self):
        # Depending on OS, path separator might differ. usage of os.path.join handles it.
        expected = os.path.join("data", "category_name.csv")
        result = downloader.construct_filepath("data", "Category", "Name", "csv")
        self.assertEqual(result, expected)

        # Test with messy inputs
        expected_messy = os.path.join("raw", "my_cat_my_file.html")
        result_messy = downloader.construct_filepath(
            "raw", "My Cat!", " My File ", "html"
        )
        self.assertEqual(result_messy, expected_messy)

    def test_get_headers(self):
        headers = downloader.get_headers()
        self.assertIsInstance(headers, dict)
        self.assertIn("User-Agent", headers)
        self.assertTrue(len(headers["User-Agent"]) > 0)

    @patch("src.ingestion.downloader.os.makedirs")
    @patch("src.ingestion.downloader.os.path.exists")
    def test_ensure_directory_creates_if_not_exists(self, mock_exists, mock_makedirs):
        mock_exists.return_value = False
        downloader.ensure_directory("new_dir")
        mock_makedirs.assert_called_once_with("new_dir")

    @patch("src.ingestion.downloader.os.makedirs")
    @patch("src.ingestion.downloader.os.path.exists")
    def test_ensure_directory_does_nothing_if_exists(self, mock_exists, mock_makedirs):
        mock_exists.return_value = True
        downloader.ensure_directory("existing_dir")
        mock_makedirs.assert_not_called()

    @patch("src.ingestion.downloader.save_metadata")
    @patch("src.ingestion.downloader.load_existing_metadata", return_value={})
    @patch("src.ingestion.downloader.console.print")  # Mock console to keep output clean
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.ingestion.downloader.os.path.exists")
    def test_download_files_no_csv(
        self, mock_exists, mock_file, mock_print, mock_load_meta, mock_save_meta
    ):
        # Simulate FileNotFoundError when opening the CSV
        mock_file.side_effect = FileNotFoundError

        downloader.download_files()

        # Should print error message
        print_calls = [str(call) for call in mock_print.mock_calls]
        self.assertTrue(any("CSV file not found" in c for c in print_calls))

    @patch("src.ingestion.downloader.save_metadata")
    @patch("src.ingestion.downloader.load_existing_metadata", return_value={})
    @patch("src.ingestion.downloader.polite_sleep")
    @patch("src.ingestion.downloader.get_session")
    @patch("src.ingestion.downloader.os.path.exists")
    @patch("src.ingestion.downloader.ensure_directory")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="Category,Name,Link,Filetype\nTestCat,TestDoc,http://example.com,pdf",
    )
    def test_download_files_downloading(
        self,
        mock_file,
        mock_ensure,
        mock_exists,
        mock_get_session,
        mock_sleep,
        mock_load_meta,
        mock_save_meta,
    ):
        # Setup
        mock_exists.return_value = False  # File doesn't exist, so download

        mock_response = MagicMock()
        mock_response.content = b"fake pdf content"
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {
            "Last-Modified": "Wed, 01 Jan 2026 00:00:00 GMT",
            "Content-Type": "application/pdf",
            "Content-Length": "16",
        }
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_get_session.return_value = mock_session

        # Execute
        downloader.download_files()

        # Assertions
        mock_ensure.assert_called_with(downloader.DEFAULT_DIR)
        mock_session.get.assert_called_with("http://example.com", timeout=30)

        expected_path = os.path.join(downloader.DEFAULT_DIR, "testcat_testdoc.pdf")
        file_handle = mock_file()
        file_handle.write.assert_called()
        mock_file.assert_any_call(expected_path, "wb")

        # Verify metadata was saved
        mock_save_meta.assert_called_once()
        saved_metadata = mock_save_meta.call_args[0][0]
        self.assertIn(expected_path, saved_metadata)
        self.assertEqual(saved_metadata[expected_path]["source_type"], "pdf")
        self.assertEqual(saved_metadata[expected_path]["title"], "TestDoc")
        self.assertEqual(
            saved_metadata[expected_path]["content_type"], "application/pdf"
        )
        self.assertEqual(saved_metadata[expected_path]["content_length_bytes"], "16")

    @patch("src.ingestion.downloader.save_metadata")
    @patch("src.ingestion.downloader.load_existing_metadata", return_value={})
    @patch("src.ingestion.downloader.os.path.getmtime", return_value=1735689600.0)
    @patch("src.ingestion.downloader.console.print")
    @patch("src.ingestion.downloader.get_session")
    @patch("src.ingestion.downloader.os.path.exists")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="Category,Name,Link,Filetype\nTestCat,SkippedDoc,http://example.com,pdf",
    )
    def test_download_files_skipping(
        self, mock_file, mock_exists, mock_get_session, mock_print, mock_getmtime, mock_load_meta, mock_save_meta
    ):
        # Setup: File exists
        mock_exists.return_value = True
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Execute
        downloader.download_files()

        # Assertions
        mock_session.get.assert_not_called()  # Should not download
        # Logic prints "Skipped (Exists)" or similar
        # We can check if console log captured it, but mock_get not called is the main proof.

    def test_get_session_has_retry(self):
        session = downloader.get_session()
        adapter = session.get_adapter("https://example.com")
        self.assertEqual(adapter.max_retries.total, downloader.RETRY_TOTAL)
        self.assertEqual(
            adapter.max_retries.backoff_factor, downloader.RETRY_BACKOFF_FACTOR
        )
        self.assertEqual(
            adapter.max_retries.status_forcelist, downloader.RETRY_STATUS_FORCELIST
        )


class TestGenerateDocId(unittest.TestCase):
    def test_returns_string(self):
        doc_id = downloader.generate_doc_id()
        self.assertIsInstance(doc_id, str)
        self.assertTrue(len(doc_id) > 0)

    def test_unique_ids(self):
        ids = {downloader.generate_doc_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)


class TestExtractAuthor(unittest.TestCase):
    def test_federal_reserve(self):
        url = "https://www.federalreserve.gov/some/path.pdf"
        self.assertEqual(downloader.extract_author(url), "www.federalreserve.gov")

    def test_other_domain(self):
        url = "https://example.com/file.csv"
        self.assertEqual(downloader.extract_author(url), "example.com")

    def test_empty_url(self):
        self.assertEqual(downloader.extract_author(""), "Unknown")


class TestMetadataPersistence(unittest.TestCase):
    def test_save_and_load_metadata(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            metadata = {
                "path/a.pdf": {
                    "doc_id": "B",
                    "source_type": "pdf",
                    "source_url": "https://example.com/a.pdf",
                    "local_path": "path/a.pdf",
                    "title": "Doc A",
                    "category": "",
                    "source_org": "",
                    "author": "Test Author",
                    "retrieved_at": "2026-01-01T00:00:00+00:00",
                    "last_modified_at": "",
                    "content_type": "application/pdf",
                    "content_length_bytes": "12345",
                },
                "path/b.csv": {
                    "doc_id": "A",
                    "source_type": "csv",
                    "source_url": "https://example.com/b.csv",
                    "local_path": "path/b.csv",
                    "title": "Doc B",
                    "category": "",
                    "source_org": "",
                    "author": "Test Author",
                    "retrieved_at": "2026-01-02T00:00:00+00:00",
                    "last_modified_at": "Wed, 01 Jan 2026 00:00:00 GMT",
                    "content_type": "text/csv",
                    "content_length_bytes": "6789",
                },
            }

            downloader.save_metadata(metadata, tmp_path)
            loaded = downloader.load_existing_metadata(tmp_path)

            self.assertEqual(len(loaded), 2)
            self.assertIn("path/a.pdf", loaded)
            self.assertIn("path/b.csv", loaded)
            self.assertEqual(loaded["path/a.pdf"]["title"], "Doc A")
            self.assertEqual(loaded["path/b.csv"]["source_type"], "csv")
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent_file(self):
        result = downloader.load_existing_metadata("/tmp/does_not_exist_12345.csv")
        self.assertEqual(result, {})

    def test_save_sorts_by_doc_id(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            metadata = {
                "path/z.pdf": {
                    "doc_id": "Z",
                    "source_type": "pdf",
                    "source_url": "",
                    "local_path": "path/z.pdf",
                    "title": "Z",
                    "category": "",
                    "source_org": "",
                    "author": "",
                    "retrieved_at": "",
                    "last_modified_at": "",
                    "content_type": "",
                    "content_length_bytes": "",
                },
                "path/a.pdf": {
                    "doc_id": "A",
                    "source_type": "pdf",
                    "source_url": "",
                    "local_path": "path/a.pdf",
                    "title": "A",
                    "category": "",
                    "source_org": "",
                    "author": "",
                    "retrieved_at": "",
                    "last_modified_at": "",
                    "content_type": "",
                    "content_length_bytes": "",
                },
            }

            downloader.save_metadata(metadata, tmp_path)

            # Read raw to check order
            with open(tmp_path, "r") as f:
                import csv

                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(rows[0]["doc_id"], "A")
            self.assertEqual(rows[1]["doc_id"], "Z")
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
