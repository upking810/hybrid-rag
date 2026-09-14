"""Agentic retrieval loop: plan -> retrieve -> judge -> (re-search) -> answer.

What makes this "agentic" rather than plain RAG: the LLM controls the
retrieval process instead of being a passive consumer of one retrieval pass.

  1. PLAN   — rewrite/decompose the user question into 1-3 search queries
              (fixes vague, compound, or conversational questions).
  2. RETRIEVE — hybrid search per query, RRF-fuse the pooled results.
  3. JUDGE  — the LLM checks whether the gathered evidence can answer the
              question; if not, it proposes a *new* query targeting the gap.
  4. LOOP   — repeat up to MAX_AGENT_ROUNDS, then generate a cited answer
              from everything gathered.

This is a minimal ReAct-style loop where the only tool is `search`. The same
skeleton extends to multiple tools (SQL, web, APIs) — which is exactly the
planning / tool-execution / agentic-loop shape of production agentic search.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .fusion import rrf
from .generate import answer
from .llm import chat_json
from .retriever import Hit, Retriever

_PLAN_SYSTEM = """You turn a user question into effective search queries for a \
technical knowledge base about information retrieval and RAG. Decompose \
compound questions; expand vague ones with the likely technical terms. \
Respond with JSON: {"queries": ["...", ...]} (1 to 3 queries)."""

_JUDGE_SYSTEM = """You judge whether retrieved passages contain enough \
information to answer the user's question. Respond with JSON:
{"sufficient": true/false, "missing": "what information is still missing", \
"next_query": "a NEW search query targeting the missing information"}.
Set sufficient=true only if the passages actually contain the answer. \
The next_query must differ from previous queries."""


@dataclass
class AgentTrace:
    rounds: list[dict] = field(default_factory=list)
    final_hits: list[Hit] = field(default_factory=list)
    answer: str = ""


def _retrieve_pooled(retriever: Retriever, queries: list[str]) -> list[Hit]:
    """Run hybrid search per query and RRF-fuse the pooled rankings."""
    rankings = []
    for q in queries:
        hits = retriever.search_hybrid(q, config.CANDIDATES_PER_RETRIEVER)
        chunk_ids = {c.chunk_id: i for i, c in enumerate(retriever.chunks)}
        rankings.append([chunk_ids[h.chunk.chunk_id] for h in hits])
    fused = rrf(rankings)
    return [Hit(retriever.chunks[i], s) for i, s in fused[: config.FINAL_TOP_K + 3]]


def agentic_answer(retriever: Retriever, question: str, verbose: bool = True) -> AgentTrace:
    trace = AgentTrace()
    plan = chat_json(_PLAN_SYSTEM, question)
    queries: list[str] = plan.get("queries") or [question]
    asked: list[str] = []
    evidence: dict[str, Hit] = {}  # chunk_id -> Hit, accumulated across rounds

    for round_no in range(1, config.MAX_AGENT_ROUNDS + 1):
        asked.extend(queries)
        hits = _retrieve_pooled(retriever, queries)
        for h in hits:
            evidence.setdefault(h.chunk.chunk_id, h)

        pooled = list(evidence.values())[: config.FINAL_TOP_K + 3]
        passages = "\n\n".join(f"({h.chunk.doc_id}) {h.chunk.text}" for h in pooled)
        verdict = chat_json(
            _JUDGE_SYSTEM,
            f"Question: {question}\nPrevious queries: {asked}\n\nPassages:\n{passages}",
        )
        trace.rounds.append({"queries": queries, "verdict": verdict,
                             "new_chunks": [h.chunk.chunk_id for h in hits]})
        if verbose:
            print(f"  round {round_no}: queries={queries}")
            print(f"    sufficient={verdict.get('sufficient')} "
                  f"missing={verdict.get('missing', '')!r}")

        if verdict.get("sufficient") or round_no == config.MAX_AGENT_ROUNDS:
            break
        next_query = verdict.get("next_query") or question
        queries = [next_query]

    trace.final_hits = list(evidence.values())[: config.FINAL_TOP_K]
    trace.answer = answer(question, trace.final_hits)
    return trace
