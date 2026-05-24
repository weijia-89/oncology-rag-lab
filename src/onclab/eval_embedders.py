"""CI-friendly in-memory retrieval embedders for eval harnesses.

Rag eval and pytest eval fixtures share these helpers so bag-of-words
hashing/normalization cannot drift between `make ragas-eval*` and eval tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core.embeddings import BaseEmbedding

if TYPE_CHECKING:
    from onclab.config import Settings

BAG_OF_WORDS_EMBEDDING_MODEL = "bag-of-words-inmemory-v1"


# sdk-review F1: single source for bag-of-words embedder shared by ragas_eval CLI and pytest
class BagOfWordsEmbedding(BaseEmbedding):
    """Deterministic keyword overlap embedder for CI-friendly retrieval tests."""

    embed_dim: int = 384

    def _bucket(self, token: str) -> int:
        # Stable across processes — unlike built-in hash().
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


def load_eval_note_nodes(settings: Settings):
    """Load one TextNode per synthetic note — shared by eval index builders."""
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


def build_in_memory_eval_index(
    settings: Settings,
    embed_model: BaseEmbedding,
    *,
    collection_name: str,
    nodes=None,
):
    """Build an ephemeral Chroma-backed index with a caller-supplied embedder."""
    import chromadb
    from llama_index.core import Settings as LIxSettings
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    LIxSettings.embed_model = embed_model
    if nodes is None:
        nodes = load_eval_note_nodes(settings)

    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=False,
    )
