"""Shared top-1 retrieval compare helpers for mock and live embedding A/B.

Mock callers: ``tests/eval/test_embedding_drift.py`` (deterministic ``MockEmbedding``
indexes from ``tests/conftest.py``).

Live callers: ``tests/eval/test_live_embedding_ab.py`` and
``scripts/embedding_live_ab_compare.py`` (``OllamaEmbedding`` over in-memory Chroma;
no persisted ``data/chroma_db/``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.request import urlopen

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.schema import TextNode

    from .config import Settings

EMBEDDING_LIVE_AB_REPORT_REL_PATH = Path("reports/embedding_live_ab_report.json")

DRIFT_QUERIES = (
    "What stage is the lung cancer patient?",
    "Which patient has glioblastoma?",
    "What is the capital of France?",
)

_CANDIDATE_PREFERENCES = ("mxbai-embed-large", "all-minilm")


def top1_node_id(
    index: VectorStoreIndex,
    query: str,
    settings: Settings,
    *,
    embed_model: BaseEmbedding | None = None,
) -> str:
    """Return the node id of the highest-scoring retrieved chunk."""
    from llama_index.core import Settings as LIxSettings

    from onclab.rag import build_retriever

    prev_embed = LIxSettings.embed_model
    if embed_model is not None:
        LIxSettings.embed_model = embed_model
    try:
        nodes = build_retriever(index, settings).retrieve(query)
    finally:
        if embed_model is not None:
            LIxSettings.embed_model = prev_embed

    assert nodes, f"No retrieval results for query {query!r}"
    return nodes[0].node.id_


def compare_retrieval_top1(
    baseline_index: VectorStoreIndex,
    candidate_index: VectorStoreIndex,
    queries: tuple[str, ...],
    settings: Settings,
    *,
    baseline_embed_model: BaseEmbedding | None = None,
    candidate_embed_model: BaseEmbedding | None = None,
) -> list[tuple[str, str, str, bool]]:
    """Side-by-side top-1 ids: (query, baseline_id, candidate_id, agree)."""
    rows: list[tuple[str, str, str, bool]] = []
    for query in queries:
        baseline_id = top1_node_id(
            baseline_index,
            query,
            settings,
            embed_model=baseline_embed_model,
        )
        candidate_id = top1_node_id(
            candidate_index,
            query,
            settings,
            embed_model=candidate_embed_model,
        )
        rows.append((query, baseline_id, candidate_id, baseline_id == candidate_id))
    return rows


def agreement_rate(rows: list[tuple[str, str, str, bool]]) -> float:
    """Fraction of queries where baseline and candidate agree on top-1 chunk id."""
    if not rows:
        return 1.0
    agreements = sum(1 for _, _, _, agree in rows if agree)
    return agreements / len(rows)


def _normalize_ollama_host(host: str) -> str:
    return host.rstrip("/")


def ollama_reachable(ollama_host: str) -> bool:
    """Return True when Ollama responds at ``/api/tags``."""
    url = f"{_normalize_ollama_host(ollama_host)}/api/tags"
    try:
        with urlopen(url, timeout=5) as resp:
            return resp.status == 200
    except (OSError, URLError, TimeoutError, ValueError):
        return False


def list_ollama_model_names(ollama_host: str) -> list[str]:
    """Return model names reported by Ollama ``/api/tags``."""
    url = f"{_normalize_ollama_host(ollama_host)}/api/tags"
    with urlopen(url, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    models = payload.get("models") or []
    return [str(entry.get("name", "")) for entry in models if entry.get("name")]


def _model_base(name: str) -> str:
    return name.split(":", 1)[0]


def pick_live_candidate_embed_model(
    baseline_model: str,
    ollama_host: str,
) -> tuple[str | None, str]:
    """Pick a second Ollama embed model for live A/B.

    Prefers ``mxbai-embed-large``, then ``all-minilm``, then any distinct embed
    model, then any distinct pulled model. Returns ``(model_name, selection_note)``.
    """
    available = list_ollama_model_names(ollama_host)
    baseline_base = _model_base(baseline_model)

    for pref in _CANDIDATE_PREFERENCES:
        for name in available:
            if _model_base(name) == pref and _model_base(name) != baseline_base:
                return name, f"selected preferred model {pref}"

    for name in available:
        base = _model_base(name)
        if base != baseline_base and "embed" in base.lower():
            return name, f"fallback embed model {base}"

    for name in available:
        base = _model_base(name)
        if base != baseline_base:
            return name, f"fallback model {base} (preferred embed models unavailable)"

    return None, "no distinct candidate model available on Ollama"


def load_synthetic_note_nodes(settings: Settings) -> list[TextNode]:
    """Load one TextNode per synthetic note under ``settings.notes_dir``."""
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.schema import TextNode

    note_paths = sorted(settings.notes_dir.glob("*.txt"))
    if not note_paths:
        raise FileNotFoundError(f"No .txt files found in {settings.notes_dir}")

    documents = SimpleDirectoryReader(
        input_files=[str(path) for path in note_paths],
    ).load_data()

    return [
        TextNode(text=doc.text, metadata=doc.metadata, id_=doc.doc_id)
        for doc in documents
    ]


def build_ollama_embedding_index(
    settings: Settings,
    embed_model_name: str,
    *,
    collection_name: str,
    nodes: list[TextNode] | None = None,
) -> VectorStoreIndex:
    """Build an ephemeral Chroma-backed index with ``OllamaEmbedding``."""
    import chromadb
    from llama_index.core import Settings as LIxSettings
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    embed_model = OllamaEmbedding(
        model_name=embed_model_name,
        base_url=settings.ollama_host,
    )
    LIxSettings.embed_model = embed_model
    if nodes is None:
        nodes = load_synthetic_note_nodes(settings)

    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=False,
    )


def build_embedding_live_ab_report(
    *,
    baseline_model: str,
    candidate_model: str,
    rows: list[tuple[str, str, str, bool]],
    ollama_reachable_flag: bool,
    candidate_selection_note: str = "",
) -> dict[str, Any]:
    """Build JSON payload for ``reports/embedding_live_ab_report.json``."""
    query_rows = [
        {
            "query": query,
            "baseline_top_id": baseline_id,
            "candidate_top_id": candidate_id,
            "agree": agree,
        }
        for query, baseline_id, candidate_id, agree in rows
    ]
    return {
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "candidate_selection_note": candidate_selection_note,
        "queries": query_rows,
        "agreement_rate": agreement_rate(rows),
        "queries_compared": len(rows),
        "ollama_reachable": ollama_reachable_flag,
    }


def default_embedding_live_ab_report_path(repo_root: Path) -> Path:
    return repo_root / EMBEDDING_LIVE_AB_REPORT_REL_PATH


def write_embedding_live_ab_report(
    repo_root: Path,
    payload: dict[str, Any],
    *,
    path: Path | None = None,
) -> Path:
    """Write live embedding A/B JSON under ``reports/``."""
    out = path or default_embedding_live_ab_report_path(repo_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def run_live_embedding_ab(
    settings: Settings,
    *,
    queries: tuple[str, ...] = DRIFT_QUERIES,
    baseline_model: str | None = None,
    candidate_model: str | None = None,
    nodes: list[TextNode] | None = None,
) -> dict[str, Any]:
    """Run live Ollama embedder A/B over in-memory indexes; return report payload."""
    from llama_index.embeddings.ollama import OllamaEmbedding

    reachable = ollama_reachable(settings.ollama_host)
    baseline = baseline_model or settings.embed_model
    selection_note = ""
    candidate = candidate_model
    if candidate is None:
        candidate, selection_note = pick_live_candidate_embed_model(baseline, settings.ollama_host)
    if candidate is None:
        raise RuntimeError(
            f"No distinct candidate embed model on Ollama (baseline={baseline!r}). "
            "Pull mxbai-embed-large or all-minilm."
        )

    if nodes is None:
        nodes = load_synthetic_note_nodes(settings)

    baseline_index = build_ollama_embedding_index(
        settings,
        baseline,
        collection_name="embedding_live_ab_baseline",
        nodes=nodes,
    )
    candidate_index = build_ollama_embedding_index(
        settings,
        candidate,
        collection_name="embedding_live_ab_candidate",
        nodes=nodes,
    )

    baseline_embed = OllamaEmbedding(model_name=baseline, base_url=settings.ollama_host)
    candidate_embed = OllamaEmbedding(model_name=candidate, base_url=settings.ollama_host)
    rows = compare_retrieval_top1(
        baseline_index,
        candidate_index,
        queries,
        settings,
        baseline_embed_model=baseline_embed,
        candidate_embed_model=candidate_embed,
    )

    return build_embedding_live_ab_report(
        baseline_model=baseline,
        candidate_model=candidate,
        rows=rows,
        ollama_reachable_flag=reachable,
        candidate_selection_note=selection_note,
    )
