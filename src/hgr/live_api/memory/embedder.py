"""Pluggable text embedder. Two concrete implementations:

  - OpenAIEmbedder: `text-embedding-3-small` over HTTPS. Cheap
    (~$0.02/M tokens). The production default.
  - FakeEmbedder: deterministic hash-based vector. Used by tests so the
    memory layer is fully exercisable offline.

The Embedder protocol exposes one method: `embed(text) -> list[float]`.
Anything that satisfies that protocol works.

Author: Konstantin Markov
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import urllib.error
import urllib.request
from typing import List, Protocol


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]: ...


# ---- OpenAI ---------------------------------------------------------------
class OpenAIEmbedder:
    """OpenAI text-embedding-3-small. Returns a 1536-float vector."""

    API_URL = "https://api.openai.com/v1/embeddings"
    DEFAULT_MODEL = "text-embedding-3-small"
    DIM = 1536

    def __init__(self, model: str = "", timeout: float = 15.0) -> None:
        self._model = model or os.environ.get("TOUCHLESS_EMBED_MODEL", self.DEFAULT_MODEL)
        self._timeout = timeout

    @staticmethod
    def configured() -> bool:
        return bool((os.environ.get("OPENAI_API_KEY") or "").strip())

    def embed(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return [0.0] * self.DIM
        key = os.environ["OPENAI_API_KEY"]
        body = json.dumps({"input": text[:8000], "model": self._model}).encode("utf-8")
        req = urllib.request.Request(
            self.API_URL, data=body,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return list((payload.get("data") or [{}])[0].get("embedding") or [])


# ---- Fake (tests / offline) ----------------------------------------------
class FakeEmbedder:
    """Hash-based deterministic embedder. Same text -> same vector;
    related text -> some cosine overlap (because we hash word-level
    tokens into a sparse vector instead of the whole string). Useful
    for offline tests that need recall to work semantically.

    Default dim is 64 — small, fast, and enough to discriminate the
    handful of distinct phrases a test cares about."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        for tok in (text or "").lower().split():
            h = hashlib.md5(tok.encode("utf-8")).digest()
            # Use 4 bytes -> uint -> index, next 4 bytes -> sign
            idx = struct.unpack("<I", h[:4])[0] % self._dim
            sign = -1.0 if (h[4] & 1) else 1.0
            vec[idx] += sign
        # L2-normalize so cosine_sim collapses to a dot product.
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# ---- factory --------------------------------------------------------------
def default_embedder() -> Embedder:
    """Use OpenAI when configured, else fall back to the fake. Production
    code paths should rely on the OpenAI one (we ship with OPENAI_API_KEY
    expected); the fallback exists so headless dev/CI doesn't crash."""
    if OpenAIEmbedder.configured():
        return OpenAIEmbedder()
    return FakeEmbedder()


# ---- vector ops -----------------------------------------------------------
def cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0:
        return 0.0
    return dot / denom
