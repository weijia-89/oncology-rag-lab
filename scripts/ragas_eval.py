#!/usr/bin/env python3
"""Run Ragas-style RAG eval and write reports/eval_report.json.

Uses DeepEval native metrics (see src/onclab/ragas_eval.py). MOCK_LLM=1 runs
lexical overlap proxies without Ollama judge calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from onclab.config import load_settings  # noqa: E402
from onclab.ragas_eval import run_rag_eval  # noqa: E402


def _build_eval_index(settings):
    import chromadb
    from llama_index.core import Settings as LIxSettings
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.schema import TextNode
    from llama_index.vector_stores.chroma import ChromaVectorStore

    class _BagOfWordsEmbedding(BaseEmbedding):
        embed_dim: int = 384

        def _bucket(self, token: str) -> int:
            return int.from_bytes(token.encode(), "big") % self.embed_dim

        def _vectorize(self, text: str) -> list[float]:
            vec = [0.0] * self.embed_dim
            for token in text.lower().split():
                vec[self._bucket(token)] += 1.0
            norm = sum(x * x for x in vec) ** 0.5
            if norm:
                vec = [x / norm for x in vec]
            return vec

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._vectorize(query)

        def _get_text_embedding(self, text: str) -> list[float]:
            return self._vectorize(text)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._vectorize(query)

    note_paths = sorted(settings.notes_dir.glob("*.txt"))
    documents = SimpleDirectoryReader(input_files=[str(p) for p in note_paths]).load_data()
    nodes = [TextNode(text=doc.text, metadata=doc.metadata, id_=doc.doc_id) for doc in documents]

    LIxSettings.embed_model = _BagOfWordsEmbedding()
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(name="ragas_eval_notes")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes, storage_context=storage_context, show_progress=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG eval and write eval_report.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "reports" / "eval_report.json",
        help="Path for JSON report (default: reports/eval_report.json)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=_REPO_ROOT / "reports" / "ragas_baseline.json",
        help="Optional baseline report for regression pass_rate",
    )
    args = parser.parse_args()

    settings = load_settings(
        notes_dir=_REPO_ROOT / "data" / "synthetic_notes",
        persist_dir=_REPO_ROOT / "data" / "chroma_db",
    )
    index = _build_eval_index(settings)
    report = run_rag_eval(
        _REPO_ROOT,
        eval_index=index,
        output_path=args.output,
        baseline_path=args.baseline if args.baseline.exists() else None,
        settings=settings,
    )
    print(f"wrote {args.output} pass_rate={report['pass_rate']:.3f}")
    return 0 if report["regression_status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
