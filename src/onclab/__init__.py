"""onclab — Local oncology RAG testbed.

Layout map (so you can navigate this package without git grep):

    config.py        — env-driven settings; one place to change model names, paths, thresholds
    llm_client.py    — thin Ollama wrapper; honors MOCK_LLM=1 for unit tests
    phoenix_setup.py — observability instrumentation (run once at app startup)
    ingest.py        — load notes -> chunk -> embed -> store in ChromaDB
    rag.py           — build the retrieval-augmented query engine
    extract.py       — structured entity extraction over RAG results
    cli.py           — typer commands: ingest / extract / eval

The ordering is also the data-flow order. If you read these files top-to-bottom,
you walk the pipeline from "raw text on disk" to "structured oncology entities".
"""

__version__ = "0.1.0"
