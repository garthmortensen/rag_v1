import csv
import os
import time
import requests
import random
import re
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

# Default Configuration
DEFAULT_CSV = 'corpus/data_sources.csv'
DEFAULT_DIR = 'corpus/raw_data'
LOG_FILE = 'corpus/download.log'
MIN_DELAY = 1
MAX_DELAY = 3

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
    clean_text = re.sub(r'[^\w\s-]', '', text)
    clean_text = re.sub(r'[\s-]+', '_', clean_text)
    return clean_text.strip().lower()

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_headers():
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/91.0.4472.124 Safari/537.36'
    }

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
    
    with console.status(f"[yellow]Cooling down for {sleep_time:.1f}s...[/yellow]") as status:
        for _ in range(steps):
            time.sleep(0.1)
            sleep_time -= 0.1
            status.update(f"[yellow]Cooling down: {max(0, sleep_time):.1f}s...[/yellow]")

def download_files():
    print_ascii_banner()
    
    ensure_directory(DEFAULT_DIR)
    headers = get_headers()
    
    results = []
    
    try:
        with open(DEFAULT_CSV, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] CSV file not found at {DEFAULT_CSV}")
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

        for row in rows:
            category = row.get('Category', 'Uncategorized').strip()
            name = row.get('Name', 'Unknown').strip()
            filetype = row.get('Filetype', 'html').strip().lower()
            url = row.get('Link', '').strip()

            if not url:
                progress.advance(task_id)
                continue

            filepath = construct_filepath(DEFAULT_DIR, category, name, filetype)
            result_status = "Unknown"
            
            progress.update(task_id, description=f"Processing: [bold]{name}[/bold]")

            if os.path.exists(filepath):
                result_status = "[yellow]Skipped (Exists)[/yellow]"
                console.print(f"  Existing: {name}")
            else:
                try:
                    console.print(f"  Downloading: {name}...")
                    response = requests.get(url, headers=headers, timeout=30)
                    response.raise_for_status()
                    with open(filepath, 'wb') as out_file:
                        out_file.write(response.content)
                    result_status = "[green]Downloaded[/green]"
                    
                    # Only sleep if we actually downloaded something
                    polite_sleep(MIN_DELAY, MAX_DELAY)
                    
                except Exception as e:
                    result_status = f"[red]Failed[/red]"
                    console.print(f"  [red]Error:[/red] {e}")

            results.append({"name": name, "category": category, "status": result_status})
            progress.advance(task_id)

    # Final Summary Table
    console.print("\n")
    table = Table(title="Download Summary", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Document Name", style="white")
    table.add_column("Status", justify="right")

    for res in results:
        table.add_row(res["category"], res["name"], res["status"])

    console.print(table)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        file_console = Console(file=f, force_terminal=False)
        file_console.print(table)

    console.print(f"\n[green]Job Complete.[/green] Files saved to: [bold]{DEFAULT_DIR}[/bold]")
    console.print(f"Summary saved to: [bold]{LOG_FILE}[/bold]")

if __name__ == "__main__":
    download_files()
