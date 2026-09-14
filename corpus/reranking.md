# Reranking

## Two-stage retrieval architecture

Production search is almost always staged: a cheap first stage scans the whole corpus and returns a candidate set with high recall (hundreds of candidates), then an expensive second stage re-scores just those candidates for precision. The stages have different jobs — first-stage retrieval must not miss the answer; the reranker must put it on top. Metrics follow: recall@100 for the first stage, nDCG@10 or MRR for the final ranking.

## Bi-encoder vs cross-encoder

A bi-encoder encodes query and document independently, enabling precomputation and ANN lookup — that is what makes first-stage dense retrieval scale. A cross-encoder concatenates query and document into one input and lets the transformer attend across them jointly, producing a single relevance score. Joint attention captures interactions a bi-encoder structurally cannot (which entity the query asks about, negation, constraint satisfaction), so cross-encoders are markedly more accurate — but each query-document pair costs a full model forward pass, so they can only run on a small candidate set. Typical latency budget: rerank 20–100 candidates.

## Reranker options in practice

Common choices: open-weight cross-encoders (the BGE reranker family, MiniLM-based rerankers), managed APIs (Cohere Rerank), and late-interaction models like ColBERT, which precompute per-token document vectors and score with a cheap MaxSim operator — a middle point between bi- and cross-encoders in both cost and quality. More recently, listwise LLM reranking ("RankGPT" style) prompts an LLM with the query and numbered passages and asks for a relevance ordering; it is expensive per query but strong zero-shot, and convenient when an LLM is already in the stack.

## When reranking pays off

Reranking helps most when the first stage retrieves the right documents but ranks them poorly — which shows up as high recall@50 with mediocre nDCG@10. It cannot recover documents the first stage never surfaced, so fix recall first (better chunking, hybrid retrieval, query rewriting) before spending on a reranker. In RAG specifically, reranking matters doubly: the generator's context window is limited, and models attend poorly to the middle of long contexts, so putting the truly relevant chunks in the first few slots measurably improves answer quality.
