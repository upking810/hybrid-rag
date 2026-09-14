# Vector Embeddings and Dense Retrieval

## What embeddings are

An embedding model maps a piece of text to a fixed-length vector (typically 384–3072 dimensions) such that semantically similar texts land close together in the vector space. Similarity is usually measured by cosine similarity; if vectors are L2-normalized, cosine similarity equals the dot product. Dense retrieval means embedding both the corpus (offline, at indexing time) and the query (online), then finding the nearest corpus vectors to the query vector.

## Why dense retrieval fixes vocabulary mismatch

Lexical search can only match tokens that literally appear. Dense retrieval matches meaning: "how do I make my search engine understand synonyms and paraphrases" retrieves a document about semantic similarity even if they share no keywords, because the embedding model was trained so that paraphrases map to nearby vectors. This is the single biggest reason to add vector search to a keyword system: users rarely phrase questions with the same words the documents use.

## Bi-encoder architecture and its trade-off

Standard dense retrieval uses a bi-encoder: query and document are encoded independently into single vectors, and relevance is just their dot product. Independence is what makes it scalable — all document vectors are precomputed, and query time is one model call plus a nearest-neighbor lookup. The cost is expressiveness: compressing a whole passage into one vector loses detail, and the model cannot attend across query and document tokens jointly. That is why a cross-encoder reranker on the top candidates usually improves quality substantially.

## Weaknesses of pure vector search

Embeddings are lossy. They tend to fail on: exact identifiers and rare proper nouns (a version number or error code has little semantic content to embed); negation and fine-grained constraints; out-of-domain vocabulary the model never saw; and very long documents squeezed into one vector (mitigated by chunking). Embedding models also have a training-data cutoff and domain bias — legal or biomedical text may embed poorly under a general-purpose model. In practice these failure modes are complementary to BM25's, which is the argument for hybrid retrieval.

## Practical knobs

Key choices: which model (quality vs cost vs dimension), whether to normalize (yes, almost always), how to chunk long texts, and whether to prefix queries and passages differently (some models are trained with "query:" / "passage:" prefixes). Embedding drift matters operationally: changing the embedding model requires re-embedding the entire corpus, so index rebuild cost is part of model choice.
