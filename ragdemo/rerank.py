"""LLM-based listwise reranking (the "RankGPT" pattern).

Why rerank at all: the first-stage retrievers (BM25, bi-encoder embeddings)
score query and document *independently*, which is what makes them fast enough
to scan the whole corpus — but they can't model fine-grained query-document
interaction. A reranker reads query and candidate *together* and re-orders a
small candidate set much more accurately. Classic choice is a cross-encoder;
here we use an LLM listwise prompt to keep the demo dependency-free (no torch).
Same two-stage architecture either way: cheap recall first, expensive
precision on the top-N only.
"""
from __future__ import annotations

from .llm import chat_json
from .retriever import Hit

_SYSTEM = """You are a search result reranker. Given a query and numbered \
passages, rank the passage numbers from most to least relevant to the query. \
Respond with JSON: {"ranking": [most_relevant_number, ..., least_relevant_number]}. \
Include every passage number exactly once."""


def llm_rerank(query: str, hits: list[Hit]) -> list[Hit]:
    if len(hits) <= 1:
        return hits
    passages = "\n\n".join(
        f"[{i + 1}] ({h.chunk.doc_id})\n{h.chunk.text}" for i, h in enumerate(hits)
    )
    result = chat_json(_SYSTEM, f"Query: {query}\n\nPassages:\n{passages}")
    ranking = result.get("ranking", [])
    seen: set[int] = set()
    order: list[int] = []
    for num in ranking:
        idx = int(num) - 1
        if 0 <= idx < len(hits) and idx not in seen:
            order.append(idx)
            seen.add(idx)
    order.extend(i for i in range(len(hits)) if i not in seen)  # fallback: keep original
    return [hits[i] for i in order]
