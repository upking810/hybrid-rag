"""Reciprocal Rank Fusion (RRF).

Why fuse by rank instead of by score: BM25 scores and cosine similarities live
on incomparable scales (BM25 is unbounded and corpus-dependent; cosine is in
[-1, 1]), so any weighted-score combination needs fragile per-query
normalization. RRF sidesteps this by only using each system's *ranking*:

    rrf_score(d) = sum over rankers of 1 / (k + rank_of_d)

k (default 60, from the original Cormack et al. paper) damps the influence of
top ranks so one ranker's #1 can't single-handedly dominate. This is what
Elasticsearch and OpenSearch ship as their default hybrid-search combiner.
"""
from __future__ import annotations

from collections import defaultdict

from . import config


def rrf(rank_lists: list[list[int]], k: int = config.RRF_K) -> list[tuple[int, float]]:
    """Fuse ranked lists of doc indices. Returns [(doc_index, rrf_score), ...] sorted."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rank_lists:
        for rank, doc_idx in enumerate(ranking, start=1):
            scores[doc_idx] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
