# Approximate Nearest Neighbor Search and HNSW

## Why approximate

Exact nearest-neighbor search is a brute-force scan: compute the distance from the query to every vector. At small scale (thousands of vectors) this is perfectly fine and even preferable — it is simple and gives perfect recall. At millions to billions of vectors it is too slow, so production systems use approximate nearest neighbor (ANN) indexes that trade a little recall for orders-of-magnitude lower latency. The recall/latency/memory triangle is the central trade-off of every ANN method.

## HNSW in one picture

HNSW (Hierarchical Navigable Small World) is the most widely deployed ANN index — it is the default in Milvus, Elasticsearch/OpenSearch k-NN, Qdrant, and pgvector. It is a multi-layer proximity graph, best understood as a skip-list made of graphs: the top layers contain few nodes with long-range links; the bottom layer contains every vector with short-range links to its near neighbors.

A search starts at an entry point in the top layer, greedily walks toward the query (always moving to the neighbor closest to the query), drops down a layer when it can't improve, and repeats until it reaches the bottom layer, where a best-first search with a candidate beam collects the nearest neighbors. The upper layers give logarithmic-like routing across the space; the bottom layer gives local precision.

## The parameters that matter

M: the maximum number of links per node. Higher M means a denser graph — better recall and connectivity, more memory (the graph links often cost as much memory as the vectors themselves) and slower construction. Typical values are 16–64.

ef_construction: the beam width used while building the graph. Higher values produce a better-quality graph at the cost of slower indexing.

ef_search (ef): the beam width at query time and the main runtime knob. Raising ef_search directly buys recall with latency — the engine explores more candidates before settling. Tuning usually means fixing a recall target (say 95% recall@10) and finding the smallest ef_search that reaches it.

## Operational properties

HNSW supports incremental inserts (good for streaming updates) but deletes are awkward — most systems tombstone deleted vectors and rebuild periodically. The whole graph must generally live in RAM, which is why memory-constrained deployments turn to quantization or disk-based indexes. Filtered search (vector search with a metadata predicate) is a known hard spot: pre-filtering can strand the graph walk, so engines implement filter-aware traversal or fall back to brute force on small filtered sets.
