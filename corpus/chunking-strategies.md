# Chunking Strategies for RAG

## Why chunk at all

Documents are chunked before embedding for three reasons. Embedding fidelity: squeezing a 20-page document into one vector averages away its topics — a query about one section matches poorly. Context economy: the generator has a limited (and expensively priced) context window, so retrieval should return focused passages, not whole files. Attribution: fine-grained chunks let the system cite the specific passage that supports a claim.

## The size trade-off

Chunk size is a retrieval-precision vs context-completeness dial. Small chunks (100–200 words) embed crisply and match queries precisely, but may cut the answer off mid-thought and force the generator to work from fragments. Large chunks (500+ words) preserve context but dilute the embedding and waste context-window budget on irrelevant sentences. Common practice lands between 150 and 500 words with 10–20% overlap between adjacent chunks, so a sentence split at a boundary survives intact in at least one chunk.

## Structure-aware chunking

Splitting on character count alone cuts sentences and tables in half. Better chunkers respect document structure: split on markdown or HTML headings first, then paragraphs, then sentences, only falling back to hard cuts for pathological cases (recursive splitting). Prepending the heading path to each chunk's text ("Installation > Docker > GPU support") is a cheap, high-leverage trick: it injects lexical and semantic context that both BM25 and the embedding model exploit. Tables, code blocks, and lists deserve special handling — keep them atomic where possible.

## Decoupling retrieval granularity from generation context

The chunk you match on need not be the text you hand the generator. Small-to-big (parent-document) retrieval embeds small chunks for precise matching but returns the enclosing section to the LLM. Sentence-window retrieval matches a single sentence and expands to its neighbors. Contextual retrieval (an Anthropic technique) prepends an LLM-written one-line summary situating each chunk within its document before embedding, reducing failed retrievals from context-free chunks. All of these attack the same problem: matching wants small, self-similar units; generation wants coherent, complete context.
