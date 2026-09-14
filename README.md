# hybrid-rag: a from-scratch hybrid retrieval + agentic RAG demo

A deliberately small RAG system built without frameworks — every stage is
~50 lines of readable Python, because the goal is to be able to explain and
defend each design decision, not to glue black boxes together.

The corpus is self-referential: 14 short technical notes about information
retrieval and RAG. So the demo answers questions about exactly the concepts it
implements — ask it `"What is reciprocal rank fusion?"` and it retrieves the
note about the algorithm it just used.

## Architecture

```
                      offline (index time)
  corpus/*.md ──> heading-aware chunking ──> chunks
                                              ├──> hand-rolled inverted index + BM25
                                              └──> embeddings (OpenAI) ──> numpy matrix

                      online (query time)
  question ──> [agentic: LLM query planning / decomposition]
           ──> BM25 top-20 ─┐
           ──> kNN  top-20 ─┴──> RRF fusion ──> [optional LLM listwise rerank]
           ──> top-5 chunks ──> [agentic: LLM sufficiency judge ──> re-search loop]
           ──> grounded generation with numbered citations
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # put your OPENAI_API_KEY in .env

python main.py index        # chunk + embed the corpus (one-time, ~$0.01)
python main.py ask "Why can't you add BM25 scores to cosine similarities?"
python main.py ask "What is HNSW?" --mode bm25          # lexical only
python main.py ask "What is HNSW?" --mode vector        # dense only
python main.py ask "How should I chunk documents?" --rerank
python main.py ask "Compare how HNSW and IVF trade memory for recall" --agentic
python main.py eval         # bm25 vs vector vs hybrid on 20 golden questions
```

## Design decisions and trade-offs

| Decision | Choice | Why / trade-off |
|---|---|---|
| BM25 | hand-rolled inverted index ([bm25.py](ragdemo/bm25.py)) | postings lists + saturated TF + length norm in ~60 lines; k1/b explained in-code |
| Vector search | exact brute-force dot product | perfect recall, microseconds at ~100 chunks; ANN (HNSW) only pays for itself at ≫10⁵ vectors |
| Fusion | RRF, k=60 ([fusion.py](ragdemo/fusion.py)) | BM25 and cosine scores are on incomparable scales; rank-based fusion needs no normalization or tuning — same default as Elasticsearch/OpenSearch |
| Chunking | heading-aware, ~180 words, 40-word overlap ([chunking.py](ragdemo/chunking.py)) | headings are semantic boundaries; heading path prepended to each chunk feeds both BM25 and the embedding |
| Rerank | LLM listwise (RankGPT-style), optional | keeps the demo torch-free; production would use a cross-encoder — same two-stage recall→precision architecture |
| Agentic loop | plan → retrieve → judge → re-search, ≤3 rounds ([agent.py](ragdemo/agent.py)) | minimal ReAct loop with search as the only tool; judge step is the pragmatic form of CRAG-style self-correction |
| Evaluation | doc-level hit@5 + MRR over 20 hand-labeled questions ([evaluate.py](ragdemo/evaluate.py)) | measures the claim "hybrid beats either alone" instead of asserting it |

## What this deliberately leaves out (and where it would go at scale)

- **ANN indexing** — swap the numpy scan for HNSW (this corpus: pointless; 10M vectors: mandatory). Planned next step: back the same interface with an OpenSearch instance running BM25 + k-NN in one engine.
- **Cross-encoder reranker** — replace the LLM rerank with a BGE-style cross-encoder for latency and cost.
- **Ingestion pipeline** — real systems need incremental re-indexing, deduplication, metadata/ACL filtering, freshness handling.
- **Generation-side eval** — faithfulness / answer-relevance judging (RAGAS-style) on top of the retrieval metrics.
