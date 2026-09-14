"""Index building/loading and the three retrieval modes: bm25 | vector | hybrid."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from . import config, embeddings
from .bm25 import BM25Index
from .chunking import Chunk, chunk_corpus
from .fusion import rrf

CHUNKS_PATH = config.INDEX_DIR / "chunks.json"
VECTORS_PATH = config.INDEX_DIR / "embeddings.npy"


@dataclass
class Hit:
    chunk: Chunk
    score: float


def build_index() -> None:
    """Chunk the corpus, embed every chunk, persist both to data/index/."""
    chunks = chunk_corpus(config.CORPUS_DIR)
    if not chunks:
        raise SystemExit(f"No markdown files found in {config.CORPUS_DIR}")
    print(f"Chunked corpus: {len(chunks)} chunks from "
          f"{len({c.doc_id for c in chunks})} documents")
    vectors = embeddings.embed_texts([c.text for c in chunks])
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(
        json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    np.save(VECTORS_PATH, vectors)
    print(f"Saved index to {config.INDEX_DIR} "
          f"(embeddings: {vectors.shape[0]} x {vectors.shape[1]})")


class Retriever:
    def __init__(self):
        if not CHUNKS_PATH.exists():
            raise SystemExit("Index not found. Run `python main.py index` first.")
        raw = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**c) for c in raw]
        self.vectors: np.ndarray = np.load(VECTORS_PATH)
        # BM25 is rebuilt in memory at load time — cheap at this scale. At real
        # scale both indexes live in the engine (e.g. Lucene segments + HNSW).
        self.bm25 = BM25Index([c.text for c in self.chunks])

    def _bm25_indices(self, query: str, top_k: int) -> list[tuple[int, float]]:
        return self.bm25.search(query, top_k)

    def _vector_indices(self, query: str, top_k: int) -> list[tuple[int, float]]:
        # Exact brute-force scan: at ~100 chunks this is microseconds. ANN
        # structures like HNSW only earn their complexity at >> 1e5 vectors.
        q = embeddings.embed_query(query)
        sims = self.vectors @ q
        top = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in top]

    def search_bm25(self, query: str, top_k: int) -> list[Hit]:
        return [Hit(self.chunks[i], s) for i, s in self._bm25_indices(query, top_k)]

    def search_vector(self, query: str, top_k: int) -> list[Hit]:
        return [Hit(self.chunks[i], s) for i, s in self._vector_indices(query, top_k)]

    def search_hybrid(self, query: str, top_k: int) -> list[Hit]:
        n = config.CANDIDATES_PER_RETRIEVER
        bm25_ranking = [i for i, _ in self._bm25_indices(query, n)]
        vec_ranking = [i for i, _ in self._vector_indices(query, n)]
        fused = rrf([bm25_ranking, vec_ranking])
        return [Hit(self.chunks[i], s) for i, s in fused[:top_k]]

    def search(self, query: str, mode: str = "hybrid",
               top_k: int = config.FINAL_TOP_K) -> list[Hit]:
        fn = {"bm25": self.search_bm25, "vector": self.search_vector,
              "hybrid": self.search_hybrid}[mode]
        return fn(query, top_k)
