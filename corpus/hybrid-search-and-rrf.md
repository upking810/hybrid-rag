# Hybrid Search and Reciprocal Rank Fusion

## Why hybrid retrieval is the production default

Lexical (BM25) and dense (embedding) retrieval fail in complementary ways. BM25 nails exact identifiers, rare terms, and codes but misses paraphrases; embeddings understand paraphrase and intent but blur exact tokens and out-of-vocabulary strings. Running both and combining the results consistently beats either alone across query mixes — which is why Elasticsearch, OpenSearch, Azure AI Search, and essentially every serious RAG stack default to hybrid retrieval.

## The score combination problem

The naive combination — a weighted sum of BM25 score and cosine similarity — has a fundamental flaw: the scores live on incomparable scales. BM25 is unbounded and depends on corpus statistics and query length; cosine similarity lives in [-1, 1] and its useful range varies by embedding model. Making a weighted sum meaningful requires per-query score normalization (min-max or z-score over the result list), which is fragile: one outlier score reshapes the whole distribution, and the right weight varies by query type.

## Reciprocal rank fusion

Reciprocal Rank Fusion (RRF) sidesteps score calibration entirely by using only each system's ranking. Each result list contributes 1 / (k + rank) for every document it contains, and documents are sorted by their summed contributions. The constant k (60 in the original Cormack et al. paper and in most implementations) dampens the top ranks: without it, a single #1 placement would dominate everything below it.

RRF's virtues are robustness and zero tuning — no normalization, no learned weights, works with any number of rankers. Its cost is that it discards score magnitude: a #1 result that is dramatically better than #2 counts the same as one that is marginally better. Both Elasticsearch and OpenSearch ship RRF as the standard hybrid combiner; OpenSearch also offers score-based normalization pipelines for teams willing to tune them.

## Beyond two rankers

Fusion generalizes: multiple query rewrites, multiple embedding models, sparse learned representations (like SPLADE), and per-field retrievers can all be fused with the same RRF machinery. A related pattern in agentic retrieval is fusing the result lists from several sub-queries generated from one user question (as in RAG-Fusion). When the candidate set after fusion still needs precision, a reranker is applied on top — fusion optimizes recall of the candidate pool, reranking optimizes the final order.
