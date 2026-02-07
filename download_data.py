import csv
import os
import time
import requests
import random
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import base32_crockford
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel

# Default Configuration
DEFAULT_CSV = "corpus/data_sources.csv"
DEFAULT_DIR = "corpus/raw_data"
LOG_FILE = "corpus/download.log"
METADATA_CSV = "corpus/metadata.csv"
METADATA_FIELDS = [
    "doc_id",
    "source_type",
    "source_url",
    "local_path",
    "title",
    "author",
    "retrieved_at",
    "last_modified_at",
]
MIN_DELAY = 1
MAX_DELAY = 3

_doc_id_counter = 0

console = Console()


def print_ascii_banner():
    console.print(
        Panel.fit(
            """[bold deep_sky_blue1]
     ▌       ▜      ▌   ▌  ▗   
    ▛▌▛▌▌▌▌▛▌▐ ▛▌▀▌▛▌  ▛▌▀▌▜▘▀▌
    ▙▌▙▌▚▚▘▌▌▐▖▙▌█▌▙▌▄▖▙▌█▌▐▖█▌
[/bold deep_sky_blue1]
 [grey70]Federal Reserve Data Acquisition [/grey70]
 --------------------------------
""",
            border_style="grey39",
        )
    )


def sanitize_filename(text):
    # Remove non-alphanumeric characters except spaces and hyphens
    clean_text = re.sub(r"[^\w\s-]", "", text)
    # Replace spaces, hyphens, and existing underscores with a single underscore
    clean_text = re.sub(r"[\s_-]+", "_", clean_text)
    # Lowercase and strip leading/trailing underscores
    return clean_text.strip("_").lower()


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    }


def generate_doc_id():
    """Generate a unique Crockford Base32 doc_id from timestamp + counter."""
    global _doc_id_counter
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)  # eg 1685625600000 for 2023-06-01T00:00:00Z
    unique_int = ts * 1000 + _doc_id_counter  # eg 1685625600000000 + counter
    _doc_id_counter += 1
    return base32_crockford.encode(unique_int)  # eg "3W5E11264SGS" for 1685625600000000


def extract_author(url):
    """Derive author/publisher from the URL domain."""
    domain_map = {
        "federalreserve.gov": "Federal Reserve",
    }
    hostname = urlparse(url).hostname or ""
    for domain, author in domain_map.items():
        if hostname.endswith(domain):
            return author
    return hostname or "Unknown"


def load_existing_metadata(path):
    """Load existing metadata CSV into a dict keyed by local_path."""
    metadata = {}
    if os.path.exists(path):
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row["local_path"]] = row
    return metadata


def save_metadata(metadata, path):
    """Write metadata dict to CSV, sorted by doc_id."""
    rows = sorted(metadata.values(), key=lambda r: r["doc_id"])
    with open(path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def construct_filepath(base_dir, category, name, filetype):
    # Flatten structure: all files go directly into base_dir
    # We include the category in the filename to keep them organized and unique
    clean_category = sanitize_filename(category)
    clean_name = sanitize_filename(name)

    filename = f"{clean_category}_{clean_name}.{filetype}"
    return os.path.join(base_dir, filename)


def polite_sleep(min_d, max_d):
    """Sleep with a countdown."""
    sleep_time = random.uniform(min_d, max_d)
    steps = int(sleep_time * 10)  # Update every 0.1s

    with console.status(
        f"[yellow]Cooling down for {sleep_time:.1f}s...[/yellow]"
    ) as status:
        for _ in range(steps):
            time.sleep(0.1)
            sleep_time -= 0.1
            status.update(
                f"[yellow]Cooling down: {max(0, sleep_time):.1f}s...[/yellow]"
            )


def download_files():
    print_ascii_banner()

    ensure_directory(DEFAULT_DIR)
    headers = get_headers()
    metadata = load_existing_metadata(METADATA_CSV)

    results = []

    # read the CSV file which maps corpus and links
    try:
        with open(DEFAULT_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        console.print(
            f"[bold red]Error:[/bold red] CSV file not found at {DEFAULT_CSV}"
        )
        return

    # progress bar
    progress_layout = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("({task.completed}/{task.total})"),
    ]

    with Progress(*progress_layout, console=console) as progress:
        task_id = progress.add_task("Processing...", total=len(rows))

        # step through each row in csv and download the file if it doesn't exist, then update progress bar
        for row in rows:
            category = row.get("Category", "Uncategorized").strip()
            name = row.get("Name", "Unknown").strip()
            filetype = row.get("Filetype", "html").strip().lower()
            url = row.get("Link", "").strip()

            if not url:
                progress.advance(task_id)
                continue

            # this is used for both the download path and the metadata entry, so we want it before we check for existing file
            filepath = construct_filepath(DEFAULT_DIR, category, name, filetype)
            result_status = "Unknown"

            progress.update(task_id, description=f"Processing: [bold]{name}[/bold]")

            # Check if the file marked for download already exists
            if os.path.exists(filepath):
                result_status = "[yellow]Skipped (Exists)[/yellow]"
                console.print(f"  Existing: {name}")
            else:
                try:
                    console.print(f"  Downloading: {name}...")
                    response = requests.get(url, headers=headers, timeout=30)
                    response.raise_for_status()
                    with open(filepath, "wb") as out_file:
                        out_file.write(response.content)
                    result_status = "[green]Downloaded[/green]"

                    # Capture metadata
                    last_modified_raw = response.headers.get("Last-Modified", "")
                    if last_modified_raw:
                        last_modified = datetime.strptime(
                            last_modified_raw, "%a, %d %b %Y %H:%M:%S %Z"
                        ).strftime("%Y%m%d%H%M%S")
                    else:
                        last_modified = ""
                    metadata[filepath] = {
                        "doc_id": generate_doc_id(),
                        "source_type": filetype,
                        "source_url": url,
                        "local_path": filepath,
                        "title": name,
                        "author": extract_author(url),
                        "retrieved_at": datetime.now(timezone.utc).strftime("%Y%m%d%H%M"),
                        "last_modified_at": last_modified,
                    }

                    # Only sleep if we actually downloaded something
                    polite_sleep(MIN_DELAY, MAX_DELAY)

                except Exception as e:
                    result_status = "[red]Failed[/red]"
                    console.print(f"  [red]Error:[/red] {e}")

            results.append(
                {"name": name, "category": category, "status": result_status}
            )
            progress.advance(task_id)

    # Save metadata
    save_metadata(metadata, METADATA_CSV)
    console.print(f"Metadata saved to: [bold]{METADATA_CSV}[/bold]")

    # Final Summary Table
    console.print("\n")
    table = Table(
        title="Download Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Category", style="cyan")
    table.add_column("Document Name", style="white")
    table.add_column("Status", justify="right")

    for res in results:
        table.add_row(res["category"], res["name"], res["status"])

    console.print(table)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        file_console = Console(file=f, force_terminal=False)
        file_console.print(table)

    console.print(
        f"\n[green]Job Complete.[/green] Files saved to: [bold]{DEFAULT_DIR}[/bold]"
    )
    console.print(f"Metadata saved to: [bold]{METADATA_CSV}[/bold]")
    console.print(f"Summary saved to: [bold]{LOG_FILE}[/bold]")


if __name__ == "__main__":
    download_files()
