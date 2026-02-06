import os
import sys
import unittest
from unittest.mock import patch, mock_open, MagicMock

# Ensure we can import the module from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import download_data


class TestDownloadData(unittest.TestCase):
    def test_sanitize_filename(self):
        self.assertEqual(download_data.sanitize_filename("Simple Name"), "simple_name")
        self.assertEqual(
            download_data.sanitize_filename("Name with @#$%^& chars"), "name_with_chars"
        )
        self.assertEqual(
            download_data.sanitize_filename("   Trailing Spaces   "), "trailing_spaces"
        )
        self.assertEqual(
            download_data.sanitize_filename("multiple__underscores"),
            "multiple_underscores",
        )
        self.assertEqual(
            download_data.sanitize_filename("dashes-and-spaces"), "dashes_and_spaces"
        )

    def test_construct_filepath(self):
        # Depending on OS, path separator might differ. usage of os.path.join handles it.
        expected = os.path.join("data", "category_name.csv")
        result = download_data.construct_filepath("data", "Category", "Name", "csv")
        self.assertEqual(result, expected)

        # Test with messy inputs
        expected_messy = os.path.join("raw", "my_cat_my_file.html")
        result_messy = download_data.construct_filepath(
            "raw", "My Cat!", " My File ", "html"
        )
        self.assertEqual(result_messy, expected_messy)

    def test_get_headers(self):
        headers = download_data.get_headers()
        self.assertIsInstance(headers, dict)
        self.assertIn("User-Agent", headers)
        self.assertTrue(len(headers["User-Agent"]) > 0)

    @patch("download_data.os.makedirs")
    @patch("download_data.os.path.exists")
    def test_ensure_directory_creates_if_not_exists(self, mock_exists, mock_makedirs):
        mock_exists.return_value = False
        download_data.ensure_directory("new_dir")
        mock_makedirs.assert_called_once_with("new_dir")

    @patch("download_data.os.makedirs")
    @patch("download_data.os.path.exists")
    def test_ensure_directory_does_nothing_if_exists(self, mock_exists, mock_makedirs):
        mock_exists.return_value = True
        download_data.ensure_directory("existing_dir")
        mock_makedirs.assert_not_called()

    @patch("download_data.console.print")  # Mock console to keep output clean
    @patch("builtins.open", new_callable=mock_open)
    @patch("download_data.os.path.exists")
    def test_download_files_no_csv(self, mock_exists, mock_file, mock_print):
        # Simulate FileNotFoundError when opening the CSV
        mock_file.side_effect = FileNotFoundError

        download_data.download_files()

        # Should print error message
        print_calls = [str(call) for call in mock_print.mock_calls]
        self.assertTrue(any("CSV file not found" in c for c in print_calls))

    @patch("download_data.polite_sleep")
    @patch("download_data.requests.get")
    @patch("download_data.os.path.exists")
    @patch("download_data.ensure_directory")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="Category,Name,Link,Filetype\nTestCat,TestDoc,http://example.com,pdf",
    )
    def test_download_files_downloading(
        self, mock_file, mock_ensure, mock_exists, mock_get, mock_sleep
    ):
        # Setup
        mock_exists.return_value = False  # File doesn't exist, so download

        mock_response = MagicMock()
        mock_response.content = b"fake pdf content"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Execute
        download_data.download_files()

        # Assertions
        mock_ensure.assert_called_with(download_data.DEFAULT_DIR)
        mock_get.assert_called_with(
            "http://example.com", headers=download_data.get_headers(), timeout=30
        )

        # Check if file was written.
        # Note: open is called twice: once for reading CSV, once for writing LOG_FILE,
        # and once for writing the downloaded file.
        # We need to find the call that writes to the constructed path.
        expected_path = os.path.join(download_data.DEFAULT_DIR, "testcat_testdoc.pdf")

        file_handle = mock_file()
        # Verify writing content
        file_handle.write.assert_called()
        # We can't easily distinguish which 'open' call returned which handle with simple mock_open setup
        # if we iterate, but we can check if open was called with the filepath 'wb'
        mock_file.assert_any_call(expected_path, "wb")

    @patch("download_data.console.print")
    @patch("download_data.requests.get")
    @patch("download_data.os.path.exists")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="Category,Name,Link,Filetype\nTestCat,SkippedDoc,http://example.com,pdf",
    )
    def test_download_files_skipping(
        self, mock_file, mock_exists, mock_get, mock_print
    ):
        # Setup: File exists
        mock_exists.return_value = True

        # Execute
        download_data.download_files()

        # Assertions
        mock_get.assert_not_called()  # Should not download
        # Logic prints "Skipped (Exists)" or similar
        # We can check if console log captured it, but mock_get not called is the main proof.


if __name__ == "__main__":
    unittest.main()
