"""Retrieval evaluation over a hand-labeled golden set.

Metrics (document-level, since golden labels point at source documents):
  - hit@k : fraction of questions where at least one relevant document
            appears among the top-k retrieved chunks' source docs.
  - MRR   : mean reciprocal rank of the first relevant document.

Running all three modes side by side is the point: it shows *measured*
evidence for why hybrid retrieval is the default in production systems —
BM25 wins on exact-term queries, vectors win on paraphrased ones, and RRF
fusion gets close to the best of both on every query.
"""
from __future__ import annotations

import json

from . import config
from .retriever import Retriever

MODES = ["bm25", "vector", "hybrid"]
K = 5


def load_golden() -> list[dict]:
    lines = config.GOLDEN_QA_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def evaluate(verbose_misses: bool = True) -> None:
    golden = load_golden()
    retriever = Retriever()
    print(f"Evaluating {len(golden)} questions, k={K}\n")
    results: dict[str, dict[str, float]] = {}

    for mode in MODES:
        hits_at_k = 0
        rr_sum = 0.0
        misses: list[str] = []
        for item in golden:
            relevant = set(item["relevant_docs"])
            retrieved_docs: list[str] = []
            for h in retriever.search(item["question"], mode=mode, top_k=K):
                if h.chunk.doc_id not in retrieved_docs:
                    retrieved_docs.append(h.chunk.doc_id)
            first_rank = next(
                (r for r, d in enumerate(retrieved_docs, 1) if d in relevant), None
            )
            if first_rank is not None:
                hits_at_k += 1
                rr_sum += 1.0 / first_rank
            else:
                misses.append(item["question"])
        results[mode] = {"hit@k": hits_at_k / len(golden), "mrr": rr_sum / len(golden)}
        if verbose_misses and misses:
            print(f"[{mode}] missed: " + "; ".join(m[:60] for m in misses))

    print(f"\n{'mode':<8} {'hit@' + str(K):>8} {'MRR':>8}")
    for mode in MODES:
        r = results[mode]
        print(f"{mode:<8} {r['hit@k']:>8.2f} {r['mrr']:>8.2f}")
