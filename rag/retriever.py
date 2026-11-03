"""ChromaDB retrieval over the SOP knowledge base built in Notebook 4."""
from __future__ import annotations

from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

from .config import INTENT_TO_DOC, Settings


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    similarity: float  # 1 - cosine distance (higher is better)

    @property
    def source(self) -> str:
        return f"{self.doc_id}.md"


class Retriever:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.embedder = SentenceTransformer(cfg.embedding_model)
        client = chromadb.PersistentClient(path=cfg.chroma_path)
        self.collection = client.get_collection(cfg.collection_name)

    @property
    def size(self) -> int:
        return self.collection.count()

    def _query(self, text: str, n: int, doc_id: str | None = None) -> list[Chunk]:
        emb = self.embedder.encode([text]).tolist()
        kwargs = {"query_embeddings": emb, "n_results": n}
        if doc_id:
            kwargs["where"] = {"doc_id": doc_id}
        res = self.collection.query(**kwargs)
        chunks = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            chunks.append(
                Chunk(cid, (meta or {}).get("doc_id", "unknown"), doc, 1.0 - float(dist))
            )
        return chunks

    @staticmethod
    def build_search_string(intent: str | None, query: str) -> str:
        """Map the router's JSON intent to a vector-search string (Task 4.3.1)."""
        if not intent:
            return query
        return f"{intent.replace('_', ' ')}: {query}"

    def retrieve(self, query: str, intent: str | None, k: int) -> tuple[str, list[Chunk]]:
        search = self.build_search_string(intent, query)
        mapped_doc = INTENT_TO_DOC.get(intent or "")
        picked: list[Chunk] = []

        # 1) Prefer chunks from the SOP the intent maps to.
        if mapped_doc:
            try:
                picked += self._query(search, max(1, k - 1), doc_id=mapped_doc)
            except Exception:  # doc not present in the index
                pass
        # 2) Fill the rest with the best global matches.
        seen = {c.chunk_id for c in picked}
        for c in self._query(search, k + 2):
            if c.chunk_id not in seen and len(picked) < k:
                picked.append(c)
                seen.add(c.chunk_id)
        picked.sort(key=lambda c: c.similarity, reverse=True)
        return search, picked[:k]
