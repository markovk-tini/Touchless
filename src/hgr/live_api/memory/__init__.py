"""Iris memory — episodic (interactions over time) and semantic (extracted
facts like person→email) storage with cheap embedding-based recall. Lets
Iris answer "what did I email Sarah about last week?" and resolve "send
the usual update to the team" using prior turns instead of starting from
zero each time.

Layout:
  store.py     - SQLite schema + CRUD (offline-testable)
  embedder.py  - pluggable embedder (OpenAI + FakeEmbedder for tests)
  extractor.py - pull semantic facts (people, recipients, files) from a turn
  manager.py   - MemoryManager: record(turn) + recall(goal) -> context

Author: Konstantin Markov
"""
from .embedder import Embedder, FakeEmbedder, OpenAIEmbedder, default_embedder
from .extractor import extract_facts
from .llm_extractor import extract_facts_from_conversation
from .manager import MemoryManager, default_memory_path
from .store import EpisodicRow, MemoryStore, SemanticRow

__all__ = [
    "Embedder", "FakeEmbedder", "OpenAIEmbedder", "default_embedder",
    "extract_facts", "extract_facts_from_conversation",
    "MemoryManager", "default_memory_path",
    "MemoryStore", "EpisodicRow", "SemanticRow",
]
