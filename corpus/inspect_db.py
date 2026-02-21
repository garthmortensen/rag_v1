"""Inspect the ChromaDB vector database.

Quick utility to list collections and chunk counts without
loading the full ChromaDB client.

Usage:
    python corpus/inspect_db.py
"""

import os
import sqlite3

from rich.console import Console
from rich.table import Table

DB_PATH = os.path.join(os.path.dirname(__file__), "vector_db", "chroma.sqlite3")

console = Console()


def inspect():
    if not os.path.isfile(DB_PATH):
        console.print(f"[bold red]Database not found:[/bold red] {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT c.name,
               COUNT(e.id)                          AS chunks,
               ROUND(AVG(LENGTH(em.string_value)))  AS avg_chars,
               MAX(LENGTH(em.string_value))         AS max_chars
        FROM collections c
        JOIN segments s ON s.collection = c.id
        LEFT JOIN embeddings e  ON e.segment_id = s.id
        LEFT JOIN embedding_metadata em
               ON em.id = e.id AND em.key = 'chroma:document'
        GROUP BY c.name
        ORDER BY c.name
        """
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        console.print("[yellow]No collections found.[/yellow]")
        return

    table = Table(
        title="Available Database Collections",
        title_style="bold cyan",
        border_style="bright_blue",
        show_header=True,
        header_style="bold",
        padding=(0, 2),
    )
    table.add_column("Collection", style="white bold")
    table.add_column("Chunks", style="green", justify="right")
    table.add_column("Avg Chars", style="yellow", justify="right")
    table.add_column("Max Chars", style="yellow", justify="right")

    for name, count, avg_chars, max_chars in rows:
        style = "dim" if count == 0 else ""
        table.add_row(
            name,
            f"{count:,}",
            f"{int(avg_chars):,}" if avg_chars is not None else "—",
            f"{int(max_chars):,}" if max_chars is not None else "—",
            style=style,
        )

    console.print()
    console.print(table)
    console.print()


if __name__ == "__main__":
    inspect()
