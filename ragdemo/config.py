"""Central configuration. Everything overridable via environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
GOLDEN_QA_PATH = DATA_DIR / "golden_qa.jsonl"

CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "text-embedding-3-small")

# Chunking: target/max size in words, overlap carried between adjacent chunks.
CHUNK_TARGET_WORDS = 180
CHUNK_MAX_WORDS = 260
CHUNK_OVERLAP_WORDS = 40

# Retrieval
CANDIDATES_PER_RETRIEVER = 20  # top-k pulled from BM25 and from vector search before fusion
RRF_K = 60                     # standard RRF constant
FINAL_TOP_K = 5                # chunks handed to the LLM

# Agentic loop
MAX_AGENT_ROUNDS = 3
