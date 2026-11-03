"""Hybrid RAG pipeline: fine-tuned intent router -> ChromaDB retrieval -> grounded answer."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from .config import Settings, settings
from .llm import LocalLLM
from .retriever import Chunk, Retriever

ROUTER_SYSTEM = "You are an intent routing assistant. Respond ONLY in valid JSON."

ANSWER_SYSTEM = (
    "You are a friendly customer support agent for an online company. "
    "Answer the customer using the policy excerpts given in their message. "
    "Be concise (2-4 sentences), polite and practical. "
    "You cannot see individual accounts or live order data, so when asked about a "
    "specific order, explain the steps from the policy for checking it. "
    "Only if the excerpts are completely unrelated to the question, say you don't have "
    "that information and suggest contacting a human agent. "
    "Never invent policies, numbers or dates."
)


@dataclass
class RAGResult:
    query: str
    intent: str | None = None
    category: str | None = None
    router_raw: str = ""
    json_valid: bool = False
    search_string: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    timings: dict = field(default_factory=dict)


class HybridRAG:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self.retriever = Retriever(cfg)
        self.llm = LocalLLM(cfg)

    # ------------------------------------------------------------ info for UI
    def info(self) -> dict:
        return {
            "Base model": self.cfg.base_model_id,
            "Intent router": (
                "LoRA fine-tuned adapter" if self.llm.adapter_loaded else "Base model (no adapter)"
            ),
            "Adapter path": self.cfg.adapter_path if self.llm.adapter_loaded else "-",
            "Device": self.llm.device + (" (4-bit)" if self.llm.quantized else ""),
            "Embedding model": self.cfg.embedding_model.split("/")[-1],
            "Vector DB": f"ChromaDB / {self.cfg.collection_name}",
            "Indexed chunks": self.retriever.size,
        }

    # ------------------------------------------------------------ step 1: route
    def route(self, query: str) -> tuple[str | None, str | None, str, bool]:
        raw = self.llm.generate(
            [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": query},
            ],
            max_new_tokens=self.cfg.router_max_new_tokens,
            use_adapter=True,
        )
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                intent = str(data.get("intent", "")).strip().lower() or None
                category = str(data.get("category", "")).strip().upper() or None
                return intent, category, raw, intent is not None
            except json.JSONDecodeError:
                pass
        return None, None, raw, False  # fall back to raw-query retrieval

    # ---------------------------------------------- steps 1+2: route and retrieve
    def prepare(self, query: str, top_k: int | None = None) -> RAGResult:
        res = RAGResult(query=query)
        t0 = time.perf_counter()
        res.intent, res.category, res.router_raw, res.json_valid = self.route(query)
        t1 = time.perf_counter()
        res.search_string, res.chunks = self.retriever.retrieve(
            query, res.intent, top_k or self.cfg.top_k
        )
        t2 = time.perf_counter()
        res.timings = {"router_s": t1 - t0, "retrieval_s": t2 - t1}
        return res

    # ------------------------------------------------------- step 3: generate
    def _answer_messages(self, res: RAGResult, history: list[dict]) -> list[dict]:
        context = "\n\n".join(
            f"[{i}] ({c.source})\n{c.text}" for i, c in enumerate(res.chunks, 1)
        )
        msgs = [{"role": "system", "content": ANSWER_SYSTEM}]
        keep = self.cfg.history_turns * 2
        msgs += [{"role": m["role"], "content": m["content"]} for m in history[-keep:]]
        msgs.append({
            "role": "user",
            "content": (
                f"Policy excerpts:\n{context}\n\n"
                f"Customer question: {res.query}\n\n"
                "Write a helpful reply based on the excerpts."
            ),
        })
        return msgs

    def stream_answer(self, res: RAGResult, history: list[dict]) -> Iterator[str]:
        start = time.perf_counter()
        yield from self.llm.stream(
            self._answer_messages(res, history),
            max_new_tokens=self.cfg.answer_max_new_tokens,
            use_adapter=False,
        )
        res.timings["generation_s"] = time.perf_counter() - start
