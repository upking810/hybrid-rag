# The RAG Pipeline End to End

## What RAG is and why it exists

Retrieval-Augmented Generation grounds an LLM's answer in retrieved evidence instead of parametric memory. It addresses the three structural weaknesses of a bare LLM: knowledge cutoff (the model can't know yesterday's documents), private data (it was never trained on your wiki), and hallucination (without evidence it fabricates plausibly). RAG is also the cheap alternative to fine-tuning for knowledge injection — updating the index is instant and reversible; fine-tuning is neither, and is better suited to teaching style or format than facts.

## The offline path: ingestion and indexing

Ingestion runs ahead of query time: load source documents (HTML, PDF, wikis, tickets), clean and parse them, chunk them with a structure-aware splitter, embed each chunk, and write chunks plus vectors into an index — alongside a lexical index over the same chunks if the system is hybrid. Metadata (source, timestamp, access control labels, section path) is stored with each chunk; it powers filtering, freshness, permission trimming, and citations later. Index freshness is an operational concern: sources change, so pipelines re-ingest incrementally and old chunks must be invalidated.

## The online path: query to answer

At query time the canonical flow is: (1) optionally rewrite the user question — expand acronyms, resolve conversational references, decompose compound questions; (2) retrieve candidates, typically hybrid BM25 + vector fused with RRF; (3) rerank the candidates and keep the top few; (4) assemble a prompt containing the question and the numbered chunks; (5) generate an answer constrained to the provided context, with inline citations; (6) optionally verify — check that claims are supported, or that the answer actually addresses the question.

## Grounding and citations

The generation prompt does real safety work: instruct the model to use only the provided context, to cite passage numbers for each claim, and to say "not found in the provided documents" rather than guess. Citations are not cosmetic — they enable spot-checking, build user trust, and turn hallucination from an invisible failure into a visible one (a claim with no citation is a flag). Systems that skip this discipline ship confident nonsense.

## Where quality is won

In practice, most RAG quality problems are retrieval problems: the right passage never reached the prompt. The highest-leverage improvements, in rough order, are better chunking, hybrid retrieval, query rewriting, and reranking — before touching the generator or its prompt. This is why serious teams instrument retrieval metrics separately from answer metrics: you cannot fix what you attribute to the wrong stage.
