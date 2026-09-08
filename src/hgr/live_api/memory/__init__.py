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
from .patterns import find_repeated_patterns, find_tool_sequences
from .project_memory import (
    IndexReport,
    ProjectChunk,
    ProjectMemoryStore,
    chunk_text,
    default_project_memory_path,
)
from .store import (
    EpisodicRow,
    MemoryStore,
    SemanticRow,
    SOURCE_KIND_CONVERSATION,
    SOURCE_KIND_LEGACY,
    SOURCE_KIND_PLANNER_STEP,
    SOURCE_KIND_USER_SAID,
    add_columns_if_missing,
    consolidate_facts,
)

__all__ = [
    "Embedder", "FakeEmbedder", "OpenAIEmbedder", "default_embedder",
    "extract_facts", "extract_facts_from_conversation",
    "MemoryManager", "default_memory_path",
    "MemoryStore", "EpisodicRow", "SemanticRow",
    "ProjectMemoryStore", "ProjectChunk", "IndexReport",
    "chunk_text", "default_project_memory_path",
    "find_repeated_patterns", "find_tool_sequences",
    "add_columns_if_missing", "consolidate_facts",
    "SOURCE_KIND_USER_SAID", "SOURCE_KIND_CONVERSATION",
    "SOURCE_KIND_PLANNER_STEP", "SOURCE_KIND_LEGACY",
]
