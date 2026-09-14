# Inverted Index and BM25

## What an inverted index is

An inverted index is the core data structure of full-text search engines such as Lucene, Elasticsearch, and OpenSearch. Instead of mapping documents to the words they contain (a "forward index"), it maps each term to a postings list: the set of documents containing that term, usually together with the term frequency and positions. Answering a keyword query then becomes intersecting or unioning a few postings lists instead of scanning every document, which is why full-text search over billions of documents can respond in milliseconds.

Postings lists are kept sorted by document id so they can be intersected with skip pointers, and are heavily compressed (delta encoding plus variable-byte or frame-of-reference encoding). Lucene stores the index in immutable segments that are periodically merged in the background; a delete is a tombstone until the merge rewrites the segment.

## TF-IDF intuition

Lexical ranking starts from two signals. Term frequency (TF): a document mentioning the query term more often is more likely to be about it. Inverse document frequency (IDF): a term that appears in few documents (like "hnsw") is far more informative than one that appears everywhere (like "system"). TF-IDF multiplies the two, scoring rare-term matches higher.

## BM25: the two fixes over raw TF-IDF

BM25 (Okapi BM25) is the default relevance function in Elasticsearch and OpenSearch. It refines TF-IDF with two ideas, each with a tunable parameter.

First, term-frequency saturation, controlled by k1 (typically 1.2–2.0). The contribution of TF is tf * (k1 + 1) / (tf + k1 * norm), which grows quickly for the first few occurrences and then flattens out: seeing a term 50 times is not ten times more relevant than seeing it 5 times. Lower k1 saturates faster.

Second, document length normalization, controlled by b (typically 0.75). Long documents naturally contain more term occurrences, so BM25 divides by a factor that scales with document length relative to the corpus average length. b=1 means full normalization, b=0 turns it off.

## Where lexical search wins and loses

BM25 is exact-match: it excels at rare tokens, product codes, error strings, function names, and identifiers ("ERR_CONN_RESET", "k1 parameter") where embeddings often blur distinctions. It fails on vocabulary mismatch: a query phrased with synonyms or paraphrases ("laptop won't turn on" vs a document saying "notebook fails to boot") shares no tokens with the answer. That gap is precisely what dense vector retrieval addresses, and why production systems run both in a hybrid setup.
