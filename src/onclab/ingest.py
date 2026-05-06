"""Ingestion pipeline: notes on disk -> chunked, embedded, indexed in ChromaDB.

The "ingest" step is what RAG papers call **indexing**. It happens once
per corpus version. Retrieval (rag.py) and generation reuse the index
without re-doing this work.

Why split it into its own module/CLI command:
    Ingestion is slow (embeddings cost CPU/GPU per chunk). Extraction and
    eval are fast against a built index. If they share one entry point,
    every test re-indexes the world. With them split, you `make ingest`
    once and then iterate on prompts/queries cheaply.

What this file does, top to bottom:
    1. Read every .txt file in the notes directory.
    2. Run the recursive splitter to make ~500-char chunks with 50-char overlap.
    3. Send each chunk through the embedding model (Ollama nomic-embed-text).
    4. Persist (text, embedding, metadata) tuples to ChromaDB on disk.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

# llama-index ships its own global `Settings` object; alias it to LIxSettings
# so our config.Settings stays unambiguous in the rest of this file.
from llama_index.core import (
    Settings as LIxSettings,
)
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import Settings


def build_index(settings: Settings) -> VectorStoreIndex:
    """Index the notes directory into ChromaDB and return a ready VectorStoreIndex.

    Why return the index object: callers who already have it built (e.g.,
    a long-running CLI session) can pass it straight to rag.build_query_engine
    without re-loading from disk. For one-shot CLI use we just throw it away;
    the persisted Chroma files are what survive.
    """

    # -- Step 1: configure the embedding model -------------------------------
    # llama-index reads its global Settings.embed_model when building an index.
    # Setting it here means downstream code doesn't have to thread it through
    # every function. The tradeoff is global state — fine for a single-process
    # CLI; you'd rethink it for a long-running server.
    LIxSettings.embed_model = OllamaEmbedding(
        model_name=settings.embed_model,
        base_url=settings.ollama_host,
    )

    # -- Step 2: configure the chunker ---------------------------------------
    # SentenceSplitter respects sentence boundaries when possible. The naive
    # alternative (splitting at exactly N chars) cuts mid-sentence and degrades
    # retrieval quality — chunks become harder to embed cleanly because they
    # start/end mid-thought.
    LIxSettings.node_parser = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # -- Step 3: load the documents ------------------------------------------
    # SimpleDirectoryReader is llama-index's "give it a folder, get Documents".
    # It picks up .txt, .pdf, .docx, .md by default. For this lab we only have
    # .txt synthetic notes, so we restrict the extension list — defensive habit
    # so a stray DOCX in the folder doesn't accidentally enter the index.
    documents = SimpleDirectoryReader(
        input_dir=str(settings.notes_dir),
        required_exts=[".txt"],
    ).load_data()

    if not documents:
        raise RuntimeError(
            f"No .txt files found in {settings.notes_dir}. "
            "Run scripts/seed_data.py or place notes in this directory."
        )

    # -- Step 4: prep the vector store ---------------------------------------
    # ChromaDB persists to disk under settings.persist_dir. PersistentClient
    # is the file-backed flavor; in-memory is fine for tests but we want
    # `make ingest` to leave behind a real index that subsequent commands
    # can reuse.
    chroma_client = chromadb.PersistentClient(path=str(settings.persist_dir))
    # get_or_create_collection: idempotent — re-running ingest reuses the
    # same collection, so previous embeddings are not duplicated. If you
    # want a clean rebuild, delete data/chroma_db/ (or `make clean`).
    collection = chroma_client.get_or_create_collection(name="oncology_notes")

    # ChromaVectorStore is llama-index's adapter on top of the chroma collection.
    # llama-index talks to it through the VectorStore Protocol — the upshot is
    # that swapping ChromaDB for Qdrant later is a one-line change here.
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # -- Step 5: build the index --------------------------------------------
    # This is where the actual work happens: each document is chunked, each
    # chunk embedded, each (chunk_text, vector, metadata) tuple stored.
    # `show_progress=True` prints a tqdm bar — handy when ingesting hundreds
    # of notes; for 8 it's overkill but harmless.
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    return index


def load_index(settings: Settings) -> VectorStoreIndex:
    """Reopen a previously-built ChromaDB index without re-ingesting.

    The other commands (extract, eval) call this. Splitting `build_index`
    from `load_index` is what makes the ingest-once / query-many pattern
    cheap.
    """
    LIxSettings.embed_model = OllamaEmbedding(
        model_name=settings.embed_model,
        base_url=settings.ollama_host,
    )

    chroma_client = chromadb.PersistentClient(path=str(settings.persist_dir))
    collection = chroma_client.get_or_create_collection(name="oncology_notes")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # from_vector_store reads existing embeddings out of Chroma instead of
    # generating new ones. If the collection is empty, retrieval will return
    # nothing — that's the signal you forgot to run `make ingest` first.
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def list_notes(notes_dir: Path) -> list[Path]:
    """Convenience: return note paths in a stable, sorted order.

    Used by the CLI for human-readable output ("processed 8 notes from data/...")
    and by tests that want to assert on the corpus directly.
    """
    return sorted(notes_dir.glob("*.txt"))
