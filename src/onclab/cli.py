"""Typer CLI: the user-facing surface.

Why typer instead of argparse:
    Typer reads function signatures and turns them into argparse for you.
    Type hints become CLI flag types, docstrings become --help text. Less
    boilerplate, harder to drift between code and docs. Wei's other
    portfolio projects already use it; matching the convention.

Commands:
    onclab ingest    — build the ChromaDB index from synthetic notes
    onclab extract   — run entity extraction over every ingested note
    onclab query     — one-off RAG query for interactive poking
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import load_settings
from .extract import ENTITY_TYPES, ExtractionRequest, extract_all
from .ingest import build_index, list_notes, load_index
from .llm_client import OllamaClient
from .phoenix_setup import setup_phoenix
from .rag import build_query_engine

app = typer.Typer(help="Local oncology RAG testbed.")
console = Console()


@app.command()
def ingest(
    notes_dir: Path = typer.Option(None, help="Override the notes directory."),
    persist_dir: Path = typer.Option(None, help="Override the ChromaDB persist directory."),
) -> None:
    """Chunk + embed + index the synthetic notes into ChromaDB."""
    settings = load_settings(notes_dir=notes_dir, persist_dir=persist_dir)
    notes = list_notes(settings.notes_dir)
    console.print(f"[bold]Ingesting[/bold] {len(notes)} notes from {settings.notes_dir}")

    build_index(settings)
    console.print(f"[green]Done.[/green] Index persisted to {settings.persist_dir}")


@app.command()
def extract(
    persist_dir: Path = typer.Option(None, help="ChromaDB directory built by `ingest`."),
    trace: bool = typer.Option(False, "--trace/--no-trace", help="Enable Phoenix tracing."),
) -> None:
    """Run entity extraction across every ingested note. Pretty-prints results.

    Why this command exists separately from `query`:
        Extraction has a fixed schema (the ENTITY_TYPES tuple). It's a
        batch operation suited to the eval suite. `query` is a free-form
        REPL-style command for poking the pipeline with ad-hoc questions.
    """
    settings = load_settings(persist_dir=persist_dir)
    if trace:
        setup_phoenix(project_name=settings.phoenix_project)

    client = OllamaClient(host=settings.ollama_host, model=settings.llm_model)
    notes = list_notes(settings.notes_dir)

    table = Table(title="Extracted entities", show_lines=True)
    table.add_column("Patient", style="cyan")
    for et in ENTITY_TYPES:
        table.add_column(et)

    for note_path in notes:
        text = note_path.read_text(encoding="utf-8")
        # Patient ID is the first line of the synthetic notes ("PATIENT ID: SYN-001").
        # In real Ontada data this would come from a structured field upstream.
        patient_id = _parse_patient_id(text) or note_path.stem
        request = ExtractionRequest(patient_id=patient_id, note_text=text)
        results = extract_all(request, client=client, settings=settings)
        table.add_row(patient_id, *(r.value for r in results))

    console.print(table)


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural-language question to ask the RAG pipeline."),
    persist_dir: Path = typer.Option(None, help="ChromaDB directory built by `ingest`."),
    trace: bool = typer.Option(False, "--trace/--no-trace", help="Enable Phoenix tracing."),
) -> None:
    """Free-form RAG query. Useful for sanity-checking the index."""
    settings = load_settings(persist_dir=persist_dir)
    if trace:
        setup_phoenix(project_name=settings.phoenix_project)

    index = load_index(settings)
    engine = build_query_engine(index, settings)
    response = engine.query(question)

    console.rule("[bold]Answer[/bold]")
    console.print(str(response))
    console.rule("[bold]Sources[/bold]")
    # Each source_node has the chunk text + a similarity score. Surfacing
    # these makes the pipeline's "why did it say that" visible to the user
    # without leaving the terminal. Same idea as Phoenix tracing, just
    # cheaper to read.
    for sn in response.source_nodes:
        console.print(f"[dim]{sn.score:.3f}[/dim] {sn.node.text[:160].strip()!r}")


def _parse_patient_id(note_text: str) -> str | None:
    """Pull 'SYN-001' out of 'PATIENT ID: SYN-001\\n...'. Best-effort."""
    for line in note_text.splitlines()[:5]:
        if line.upper().startswith("PATIENT ID:"):
            return line.split(":", 1)[1].strip()
    return None


if __name__ == "__main__":
    app()
