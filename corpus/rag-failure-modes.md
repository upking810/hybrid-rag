# RAG Failure Modes

## Retrieval-side failures

The most common failure class: the evidence never reaches the prompt. Causes include vocabulary mismatch (pure lexical retrieval and a paraphrased query), bad chunking (the answer is split across two chunks, or buried in an oversized chunk whose embedding averaged it away), embedding domain mismatch (general-purpose model on specialized jargon), and missing-from-corpus (the honest case — the system should say so rather than improvise). Compound questions fail structurally: one query vector cannot represent two information needs, which is what query decomposition addresses.

## Lost in the middle

LLMs attend unevenly over long contexts: information at the beginning and end of the prompt is used far more reliably than information in the middle. Stuffing 20 retrieved chunks into the context can therefore *reduce* answer quality — the right passage drowns at position 11. Mitigations: retrieve fewer, better chunks (rerank, then cut aggressively), and order context so the strongest evidence sits first or last. This is a key reason reranking improves RAG even when recall is already fine.

## Hallucination despite retrieval

Grounding reduces but does not eliminate fabrication. Typical patterns: the model blends parametric memory with context (answers from what it "knows" when context is thin), over-generalizes from a partially relevant passage, or fabricates citations — citing passage [3] for a claim passage [3] does not make. Counters: strict grounding instructions, requiring per-claim citations, an explicit "not in the provided documents" escape hatch, and a post-generation faithfulness check that flags unsupported claims. Retrieval quality is itself an anti-hallucination lever: models fabricate most when context is almost-but-not-quite relevant.

## Staleness and conflict

The index is a cache of the world and goes stale: documents change after embedding, deleted pages linger as orphaned chunks, and near-duplicate versions of the same document retrieve together, wasting context and sometimes contradicting each other. Freshness pipelines (incremental re-ingestion, tombstoning), deduplication at index time, and timestamp metadata for recency-aware ranking address this. When sources genuinely conflict, surfacing both with citations beats silently picking one.

## Security: prompt injection via retrieved content

RAG turns your corpus into part of the prompt, so a document containing "ignore your instructions and..." becomes an injection vector — retrieval is an attack surface, not just a quality problem. Mitigations include treating retrieved text strictly as data (clear delimiter framing and instructions that context is never to be obeyed), sanitizing ingested content, and restricting what downstream tools an answer can trigger. None are complete; defense in depth is the current state of the art.
