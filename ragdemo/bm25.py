"""Hand-rolled inverted index + BM25 (Okapi) scoring.

Deliberately written from scratch instead of using a library: the point of the
demo is being able to explain every step. The index is a classic postings-list
inverted index: term -> [(doc_index, term_frequency), ...].

BM25 in one breath: TF-IDF with two fixes — term frequency saturates (k1
controls how fast: seeing a term 20x is not 20x more relevant than 1x), and
scores are normalized by document length (b controls how much: long documents
shouldn't win just by containing more words).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        doc_tokens = [tokenize(d) for d in documents]
        self.doc_lens = [len(t) for t in doc_tokens]
        self.n_docs = len(doc_tokens)
        self.avgdl = sum(self.doc_lens) / max(self.n_docs, 1)

        # Inverted index: postings[term] = [(doc_index, tf), ...]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, tokens in enumerate(doc_tokens):
            for term, tf in Counter(tokens).items():
                self.postings[term].append((i, tf))

        # BM25+-style smoothed IDF (always positive, unlike the classic formula
        # which can go negative for terms present in >50% of documents).
        self.idf = {
            term: math.log(1 + (self.n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self.postings.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return [(doc_index, score), ...] sorted by descending BM25 score."""
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for doc_idx, tf in self.postings[term]:
                dl_norm = 1 - self.b + self.b * self.doc_lens[doc_idx] / self.avgdl
                scores[doc_idx] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * dl_norm)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]
