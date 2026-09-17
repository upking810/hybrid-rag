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

## Results

All outputs below are real runs against the 58-chunk index (2026-09-14).

### Retrieval evaluation (20 golden questions, k=5)

```
mode        hit@5      MRR
bm25         0.95     0.90
vector       1.00     0.97
hybrid       1.00     0.97
```

Two honest observations, which matter more than the numbers:

- **BM25's one miss is a textbook vocabulary-mismatch case** (see below): the
  question shares almost no tokens with the relevant document.
- **Hybrid ties vector instead of beating it.** At 58 clean, well-written
  chunks with well-formed questions, dense retrieval saturates. The regime
  where hybrid earns its keep is larger/noisier corpora and queries containing
  exact identifiers, error codes, or rare tokens — where embeddings blur and
  BM25 provides the floor. Eval conclusions are a function of the query
  distribution; a single aggregate number hides this.

### Vocabulary mismatch, demonstrated

`"How do I measure whether my retriever is finding the right documents?"` —
no lexical overlap with `rag-evaluation.md` (which says recall@k, MRR, nDCG,
golden set):

```
retrieved (bm25):                          retrieved (vector):
  [1] vector-embeddings.md      ✗            [1] rag-evaluation.md         ✓
  [2] vector-embeddings.md      ✗            [2] agentic-search.md
  [3] reranking.md                           [3] reranking.md
  ...top-5 misses rag-evaluation.md          ...relevant doc ranked #1
```

BM25 chases the surface words ("finding", "documents") into the wrong
documents; the embedding matches the intent.

### Agentic loop on a compound question

`"Compare how HNSW and IVF-PQ trade memory for recall, and which suits a
billion-vector corpus"` with `--agentic`:

```
round 1: queries=['HNSW vs IVF-PQ memory recall comparison',
                  'HNSW performance on billion-vector corpus',
                  'IVF-PQ suitability for large-scale vector databases']
  sufficient=False missing='...quantitative metrics or examples...'
round 2: queries=['HNSW and IVF-PQ memory recall trade-offs for billion-vector datasets']
  sufficient=False ...
round 3: ... (max rounds reached, answering from accumulated evidence)
```

The planner decomposed the comparison into three targeted sub-queries
(retrieval pooled both `ann-search-hnsw.md` and `ivf-and-pq.md` — a single
query embedding tends to favor one side of a comparison). The judge kept
demanding quantitative benchmarks the corpus doesn't contain, so the
`MAX_AGENT_ROUNDS` bound kicked in and the system answered from the gathered
evidence anyway — the loop-budget safeguard doing exactly its job. The final
answer correctly attributes RAM-residency to HNSW and recommends IVF-PQ for
billion-scale, with per-claim citations.

### Reranking

`"When is it worth adding a reranker to a RAG system?"` with `--rerank`: the
LLM listwise reranker promoted the two directly-on-point chunks
(`Reranking > When reranking pays off`, `RAG Failure Modes > Lost in the
middle`) into the top slots, and the answer cites both: reranking pays when
recall@50 is high but nDCG@10 is mediocre, and matters doubly in RAG because
models attend poorly to the middle of long contexts.

## What this deliberately leaves out (and where it would go at scale)

- **ANN indexing** — swap the numpy scan for HNSW (this corpus: pointless; 10M vectors: mandatory). Planned next step: back the same interface with an OpenSearch instance running BM25 + k-NN in one engine.
- **Cross-encoder reranker** — replace the LLM rerank with a BGE-style cross-encoder for latency and cost.
- **Ingestion pipeline** — real systems need incremental re-indexing, deduplication, metadata/ACL filtering, freshness handling.
- **Generation-side eval** — faithfulness / answer-relevance judging (RAGAS-style) on top of the retrieval metrics.
