"""Grounded answer generation with numbered citations."""
from __future__ import annotations

from .llm import chat
from .retriever import Hit

_SYSTEM = """You answer questions using ONLY the provided context passages.

Rules:
- Cite passages inline with their bracketed numbers, e.g. "BM25 saturates term \
frequency [2]." Every factual claim needs at least one citation.
- If the context does not contain the answer, say so explicitly instead of \
guessing. Do not use outside knowledge.
- Be concise and technical."""


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {h.chunk.doc_id}, section: {h.chunk.heading})\n{h.chunk.text}"
        for i, h in enumerate(hits)
    )


def answer(query: str, hits: list[Hit]) -> str:
    context = format_context(hits)
    return chat(_SYSTEM, f"Context passages:\n\n{context}\n\nQuestion: {query}")
