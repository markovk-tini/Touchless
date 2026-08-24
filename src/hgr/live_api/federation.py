"""Cross-device federation skeleton.

Phase-10 subscription tier. Iris on PC, mobile, and watch
eventually all share the same memory + entity_graph + standing
orders via a Touchless cloud sync layer. The subscription proxy
hosts that endpoint.

This module is the SKELETON: contracts + an in-memory fake +
the local exporter / importer / merge logic, so other modules
can integrate against a stable interface today even though the
actual cloud endpoint ships later.

Sync model: at-least-once eventual-consistency with monotonic
clocks per device. Each artifact (memory fact, entity, standing
order, prompt variant outcome) carries:
  * device_id     — UUID of the device that authored it
  * version       — incrementing per artifact per device
  * updated_at    — wall-clock at last edit
  * tombstone     — soft-delete marker

On sync:
  * Pull remote artifacts since last_known_remote_version
  * Local applies remote artifacts whose updated_at > local's
    (last-writer-wins, tie broken by device_id sort)
  * Push local artifacts whose updated_at > last_pushed
  * Track per-artifact-kind progress in `sync_state` so partial
    failures resume cleanly

Public:
  * `register_provider(kind, provider)` — register a per-kind
    sync provider (e.g. memory.store, entity_graph).
  * `sync_once()` — best-effort pull + push. Returns
    SyncReport with counts.
  * `set_remote(remote_endpoint)` — point at the cloud (None
    disables sync).

The cloud endpoint is not yet implemented; in tests the in-memory
fake `InMemoryRemote` stands in.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol


_DEVICE_ID_ENV_KEY = "TOUCHLESS_DEVICE_ID"


def _resolve_device_id() -> str:
    import os
    env = os.environ.get(_DEVICE_ID_ENV_KEY, "").strip()
    if env:
        return env
    return f"dev-{uuid.uuid4().hex[:12]}"


# ---- artifact contract --------------------------------------------

@dataclass
class FederatedArtifact:
    artifact_id: str               # globally unique
    kind: str                       # 'memory' | 'entity' | 'order' | ...
    device_id: str
    version: int
    updated_at: float
    payload: Dict[str, Any] = field(default_factory=dict)
    tombstone: bool = False

    def serialize(self) -> str:
        return json.dumps({
            "id": self.artifact_id,
            "kind": self.kind,
            "device_id": self.device_id,
            "version": self.version,
            "updated_at": self.updated_at,
            "payload": self.payload,
            "tombstone": self.tombstone,
        }, sort_keys=True, default=str)

    @classmethod
    def deserialize(cls, blob: str) -> "FederatedArtifact":
        d = json.loads(blob)
        return cls(
            artifact_id=d["id"], kind=d["kind"],
            device_id=d["device_id"],
            version=int(d["version"]),
            updated_at=float(d["updated_at"]),
            payload=d.get("payload", {}) or {},
            tombstone=bool(d.get("tombstone")))


class FederationProvider(Protocol):
    """Per-kind sync provider. Each subsystem (memory, entity_graph,
    standing_orders) implements this so the engine can drive sync."""

    def export(self, since: float) -> List[FederatedArtifact]:
        ...

    def apply(self, artifact: FederatedArtifact) -> bool:
        ...


@dataclass
class SyncReport:
    pushed: int = 0
    pulled: int = 0
    applied: int = 0
    conflicts: int = 0
    failed: int = 0
    duration_ms: float = 0.0


# ---- remote interface ----------------------------------------------

class FederationRemote(Protocol):
    """Cloud-side interface. Tests use InMemoryRemote."""

    def push(self, artifacts: List[FederatedArtifact]
             ) -> int:
        ...

    def pull(self, since_per_device: Dict[str, int]
             ) -> List[FederatedArtifact]:
        ...


class InMemoryRemote:
    """Test / dev double. Holds artifacts in a list; pull returns
    everything past the per-device version watermark."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._artifacts: List[FederatedArtifact] = []

    def push(self, artifacts: List[FederatedArtifact]) -> int:
        with self._lock:
            for a in artifacts:
                # Replace existing entry with same id, else append.
                idx = next(
                    (i for i, x in enumerate(self._artifacts)
                     if x.artifact_id == a.artifact_id),
                    -1)
                if idx >= 0:
                    self._artifacts[idx] = a
                else:
                    self._artifacts.append(a)
            return len(artifacts)

    def pull(self,
             since_per_device: Dict[str, int]
             ) -> List[FederatedArtifact]:
        with self._lock:
            out: List[FederatedArtifact] = []
            for a in self._artifacts:
                last = since_per_device.get(a.device_id, 0)
                if a.version > last:
                    out.append(a)
            return out

    def all(self) -> List[FederatedArtifact]:
        with self._lock:
            return list(self._artifacts)

    def clear(self) -> None:
        with self._lock:
            self._artifacts.clear()


# ---- engine --------------------------------------------------------

class FederationEngine:
    """Orchestrates pull + apply + push. Tracks per-kind / per-device
    high-water marks so re-sync is incremental."""

    def __init__(self,
                 *,
                 device_id: Optional[str] = None,
                 remote: Optional[FederationRemote] = None
                 ) -> None:
        self._device_id = device_id or _resolve_device_id()
        self._remote = remote
        self._providers: Dict[str, FederationProvider] = {}
        self._last_push_ts: float = 0.0
        # Per-device-id high-water mark for pulls.
        self._pulled_versions: Dict[str, int] = {}
        self._lock = threading.RLock()

    @property
    def device_id(self) -> str:
        return self._device_id

    def set_remote(self,
                   remote: Optional[FederationRemote]) -> None:
        with self._lock:
            self._remote = remote

    def register_provider(self, kind: str,
                          provider: FederationProvider) -> None:
        with self._lock:
            self._providers[kind] = provider

    def sync_once(self) -> SyncReport:
        report = SyncReport()
        if self._remote is None:
            return report
        start = time.time()
        # Pull first so we don't push conflicts.
        try:
            with self._lock:
                pulled = self._remote.pull(
                    dict(self._pulled_versions))
        except Exception:
            pulled = []
        report.pulled = len(pulled)
        for art in pulled:
            try:
                provider = self._providers.get(art.kind)
                if provider is None:
                    continue
                ok = provider.apply(art)
                if ok:
                    report.applied += 1
                with self._lock:
                    cur = self._pulled_versions.get(
                        art.device_id, 0)
                    if art.version > cur:
                        self._pulled_versions[art.device_id] = (
                            art.version)
            except Exception:
                report.failed += 1
        # Push local exports per provider.
        with self._lock:
            since = self._last_push_ts
            providers = dict(self._providers)
        outgoing: List[FederatedArtifact] = []
        for kind, prov in providers.items():
            try:
                outgoing.extend(prov.export(since))
            except Exception:
                report.failed += 1
        if outgoing:
            try:
                report.pushed = self._remote.push(outgoing)
            except Exception:
                report.failed += 1
        with self._lock:
            self._last_push_ts = time.time()
        report.duration_ms = (time.time() - start) * 1000.0
        return report

    def reset_state(self) -> None:
        """Admin: clear watermarks so a full re-sync runs next time."""
        with self._lock:
            self._last_push_ts = 0.0
            self._pulled_versions.clear()


# ---- singleton ------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton: Optional[FederationEngine] = None


def global_engine() -> FederationEngine:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = FederationEngine()
        return _singleton


def reset_global() -> None:
    global _singleton
    with _singleton_lock:
        _singleton = None
