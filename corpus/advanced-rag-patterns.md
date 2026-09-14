# Advanced RAG Patterns

## Query-side transformations

HyDE (Hypothetical Document Embeddings) asks the LLM to write a hypothetical answer to the query and embeds *that* instead of the query — answer-shaped text lands closer to real answer passages than question-shaped text does. RAG-Fusion generates several rewrites of the query, retrieves for each, and merges the lists with reciprocal rank fusion. Step-back prompting first asks a more general question to fetch background context before the specific one. All exploit the same asymmetry: queries are short and oddly distributed; a cheap LLM call reshapes them toward the document distribution.

## GraphRAG and structured knowledge

GraphRAG (popularized by Microsoft) builds a knowledge graph from the corpus at ingestion time — an LLM extracts entities and relations, then communities are detected and summarized hierarchically. Queries that defeat chunk-level retrieval — "what are the main themes across these documents", or multi-hop questions traversing relationships — are answered from community summaries and graph neighborhoods instead of isolated chunks. The cost is a heavy, LLM-intensive indexing pass and a harder freshness story. It complements rather than replaces vector RAG: local factual questions still go to chunks.

## Late interaction: ColBERT

ColBERT keeps a vector *per token* of each document instead of one vector per chunk. At query time each query token finds its best-matching document token (MaxSim) and the scores sum. This preserves fine-grained matching that single-vector bi-encoders average away, at roughly cross-encoder-level quality but with precomputable document representations — the price is index size (many vectors per document) and specialized infrastructure (e.g. the PLAID engine). Sparse learned retrieval (SPLADE) is the other middle path: a transformer expands documents into weighted term sets served by ordinary inverted indexes.

## Multimodal retrieval

Multimodal embedding models (the CLIP lineage and successors) map images and text into a shared vector space, enabling text-to-image and image-to-image search with the same ANN machinery. Document-image retrieval models like ColPali embed page screenshots directly — sidestepping brittle PDF parsing by treating layout, tables, and figures visually. Production multimodal RAG typically stores per-modality embeddings plus extracted text, fuses across them, and increasingly feeds retrieved images straight into vision-language models for generation.
