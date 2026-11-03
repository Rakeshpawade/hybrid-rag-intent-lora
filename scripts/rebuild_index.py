"""Rebuild the ChromaDB index from retrieval_corpus.json.

Usage: python scripts/rebuild_index.py [path/to/retrieval_corpus.json]

Expected JSON: list of {"chunk_id"|"id": str, "doc_id": str, "text": str}
(adjust the key lookups below if your notebook exported different names).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from rag.config import ROOT, settings  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "retrieval_corpus.json")
with open(src, encoding="utf-8") as f:
    corpus = json.load(f)

ids, docs, metas = [], [], []
for i, row in enumerate(corpus):
    ids.append(str(row.get("chunk_id") or row.get("id") or f"chunk_{i}"))
    docs.append(row.get("text") or row.get("content") or row.get("chunk"))
    metas.append({"doc_id": str(row.get("doc_id") or row.get("source") or "unknown")})

embedder = SentenceTransformer(settings.embedding_model)
client = chromadb.PersistentClient(path=settings.chroma_path)
try:
    client.delete_collection(settings.collection_name)
except Exception:
    pass
col = client.create_collection(settings.collection_name, metadata={"hnsw:space": "cosine"})
col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embedder.encode(docs).tolist())
print(f"Indexed {col.count()} chunks into '{settings.collection_name}' at {settings.chroma_path}")
