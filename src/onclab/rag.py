"""Build the retrieval-augmented query engine.

Mental model for this file:
    The query engine is the function `(query: str) -> Response`.
    Internally it does: embed the query -> retrieve top-k chunks ->
    stuff chunks into a prompt template -> call the LLM -> return the
    text plus references to the source chunks. Each of those steps is
    a swappable component in llama-index, which is what makes RAG
    architectures testable layer by layer.

Why we expose retrieval-only AND query-engine paths:
    The strategy guide is explicit: the retrieval layer is *deterministic*
    and should be tested with classical assertions ("did the right chunks
    surface?"). The generation layer is non-deterministic and needs
    behavioral / LLM-as-judge metrics. Keeping the retriever accessible
    on its own lets test_retrieval.py poke just that layer without paying
    the LLM-call cost.
"""

from __future__ import annotations

from llama_index.core import (
    Settings as LIxSettings,
)
from llama_index.core import (
    VectorStoreIndex,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama as LIxOllama

from .config import Settings


def configure_llm(settings: Settings) -> None:
    """Wire llama-index's global LLM to our Ollama instance.

    Called once before building the query engine. Same global-state
    tradeoff as in ingest.py — lives in one place to keep call sites clean.
    """
    LIxSettings.llm = LIxOllama(
        model=settings.llm_model,
        base_url=settings.ollama_host,
        # request_timeout matters: a 14B model on a 4070 Ti Super takes
        # 5–15 seconds for a few-hundred-token completion. Default of 30s
        # is fine; we set it explicitly to make the value visible.
        request_timeout=120.0,
    )


def build_retriever(index: VectorStoreIndex, settings: Settings) -> VectorIndexRetriever:
    """Return a retriever that surfaces top-k chunks for any query.

    The retriever is the deterministic half of the RAG pipeline. Given
    the same query and the same index, it returns the same chunks every
    time (modulo embedding-model nondeterminism, which for a frozen model
    is none). This is what lets test_retrieval.py write classical asserts.
    """
    return VectorIndexRetriever(index=index, similarity_top_k=settings.top_k)


def build_query_engine(index: VectorStoreIndex, settings: Settings) -> RetrieverQueryEngine:
    """End-to-end query engine: retrieve + LLM generation.

    The default RetrieverQueryEngine assembles a prompt of the form
    "Given context: {chunks}\n\nAnswer: {query}". You can customize
    the template; for v1 the default is fine — the goal at this stage
    is to get a working pipeline you can poke with eval, not to tune
    prompts. Prompt tuning is a *consequence* of eval results, not the
    starting point.
    """
    configure_llm(settings)
    retriever = build_retriever(index, settings)

    # streaming=False is the right default here: DeepEval and the CLI both
    # consume the full response string. Streaming matters for interactive
    # UX, not for batch eval.
    return RetrieverQueryEngine.from_args(retriever=retriever, streaming=False)


def retrieve_only(index: VectorStoreIndex, query: str, settings: Settings) -> list[str]:
    """Helper for tests: return raw chunk texts without calling the LLM.

    test_retrieval.py uses this to assert "for query X, chunk text containing
    Y must appear in the top_k results". That's the layer where a regression
    is caught before it propagates into a wrong generated answer.
    """
    retriever = build_retriever(index, settings)
    nodes = retriever.retrieve(query)
    # Each node has a `.text` (chunk content) and `.metadata` (file path,
    # chunk index, etc.). For assertion purposes we just want the text.
    return [node.text for node in nodes]
