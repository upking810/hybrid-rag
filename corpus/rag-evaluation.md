# Evaluating RAG Systems

## Evaluate the stages separately

A RAG system has two failure surfaces — retrieval and generation — and conflating them makes debugging impossible. If the answer is wrong, was the right passage never retrieved (retrieval miss), retrieved but ranked below the cutoff (ranking miss), or present in context yet ignored or contradicted by the model (generation miss)? Stage-wise metrics answer that question; a single end-to-end score does not.

## Retrieval metrics

Retrieval evaluation needs a golden set: questions labeled with their relevant documents or passages. Core metrics: recall@k (is a relevant item in the top k — the metric that bounds everything downstream), hit rate, MRR (mean reciprocal rank — how high does the first relevant item appear), and nDCG (rank-position-weighted gain with graded relevance). For RAG, recall@k at the k you actually pass to the generator is the number to watch: a relevant document at rank 15 might as well not exist if only 5 chunks enter the prompt.

Building the golden set is the real work. Start small — even 20–50 hand-labeled questions expose most regressions — and grow it from real user queries and observed failures. Synthetic question generation (an LLM writes questions per chunk) scales the set cheaply but inherits the LLM's phrasing bias, so keep a hand-written core.

## Generation metrics

Given retrieved context, the generation side is judged on: faithfulness / groundedness (is every claim supported by the context — the anti-hallucination metric), answer relevance (does it address the question), and completeness. These are graded by human review or LLM-as-judge; frameworks like RAGAS package faithfulness, answer relevance, context precision, and context recall as LLM-judged scores. LLM judges are cheap and correlate reasonably with humans but have known biases (verbosity, position, self-preference), so calibrate them against a hand-graded sample before trusting trends.

## Evaluation as regression infrastructure

The point of all this is not a one-time score but a harness: every change to chunking, embedding model, fusion weights, or prompts runs against the golden set before shipping, exactly like a test suite. Retrieval changes that look like obvious wins routinely regress a query class — a harness catches it, intuition does not. Online, the same discipline continues with A/B tests, user feedback signals (thumbs, reformulation rate, abandonment), and sampled human audits.
