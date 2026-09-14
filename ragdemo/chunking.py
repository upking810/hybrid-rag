"""Heading-aware markdown chunking with overlap.

Strategy: split on markdown headings first (a heading is a natural semantic
boundary), then greedily pack paragraphs into chunks of ~CHUNK_TARGET_WORDS,
hard-capped at CHUNK_MAX_WORDS. Adjacent chunks within a section share
CHUNK_OVERLAP_WORDS of trailing context so a sentence cut at a boundary is
still retrievable from at least one chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from . import config


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str      # source filename, e.g. "hybrid-search-and-rrf.md"
    heading: str     # nearest heading path, e.g. "Hybrid Search > Why fuse ranks"
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a markdown doc into (heading_path, body) sections."""
    sections: list[tuple[str, str]] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            path = " > ".join(h for _, h in heading_stack) or "(intro)"
            sections.append((path, body))
        current_lines.clear()

    for line in markdown.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, m.group(2).strip()))
        else:
            current_lines.append(line)
    flush()
    return sections


def _pack_paragraphs(paragraphs: list[str]) -> list[str]:
    """Greedily pack paragraphs into chunks near the target size, with overlap."""
    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for para in paragraphs:
        words = len(para.split())
        if buf and buf_words + words > config.CHUNK_MAX_WORDS:
            chunks.append("\n\n".join(buf))
            overlap = " ".join(" ".join(buf).split()[-config.CHUNK_OVERLAP_WORDS:])
            buf = [f"(...) {overlap}"] if overlap else []
            buf_words = len(overlap.split())
        buf.append(para)
        buf_words += words
        if buf_words >= config.CHUNK_TARGET_WORDS:
            chunks.append("\n\n".join(buf))
            overlap = " ".join(" ".join(buf).split()[-config.CHUNK_OVERLAP_WORDS:])
            buf = [f"(...) {overlap}"] if overlap else []
            buf_words = len(overlap.split())
    if buf and buf_words > config.CHUNK_OVERLAP_WORDS:  # don't emit an overlap-only tail
        chunks.append("\n\n".join(buf))
    return chunks


def chunk_document(path: Path) -> list[Chunk]:
    doc_id = path.name
    text = path.read_text(encoding="utf-8")
    chunks: list[Chunk] = []
    for heading, body in _split_sections(text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for i, piece in enumerate(_pack_paragraphs(paragraphs)):
            chunk_id = f"{doc_id}::{heading}::{i}"
            # Prepend the heading path: it carries strong lexical signal for BM25
            # and disambiguating context for the embedding model.
            chunks.append(Chunk(chunk_id, doc_id, heading, f"[{heading}]\n{piece}"))
    return chunks


def chunk_corpus(corpus_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        chunks.extend(chunk_document(path))
    return chunks
