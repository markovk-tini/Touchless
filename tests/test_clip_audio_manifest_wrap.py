"""Probe: verify _parse_ffmpeg_clip_audio_manifest's chained
wall_start_mtime logic holds when the segment-wrap count is bumped
from 8 (v1) to 32 (v2 buffer scaling) and segments get rewritten by
ffmpeg's ring rotation.

Why this matters:
* v2 plan bumped clip_max_buffer_seconds 65 -> 305 → wrap_count
  ceil(305/10)+1 = 32.
* The export uses wall_start_mtime / wall_end_mtime to pick which
  segments overlap the clip window. If the chain drifts at higher
  wrap counts the 2 m / 5 m clips end up extracting from the wrong
  wall window.
* The user reported 2 m / 5 m clips with audio sync wildly off.
  We need to know whether the manifest parser is the cause or
  whether it's something downstream.

What this probe does:
* Replicates _parse_ffmpeg_clip_audio_manifest's algorithm exactly
  against synthetic temp files.
* Simulates four scenarios:
    A. 8 segments, no wrap (v1 baseline)
    B. 32 segments, no wrap (v2, buffer not yet full)
    C. 32 segments, partial wrap (3 segments rewritten)
    D. 32 segments, full wrap (every slot rewritten at least once)
* For each entry checks: wall_start_mtime is contiguous with the
  previous entry's wall_end_mtime, the chained values are close
  to the segment's REAL wall start, and the export's window-
  selection math picks the right subset for 60 s / 2 m / 5 m
  clip windows.
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


# ---------- Replica of _parse_ffmpeg_clip_audio_manifest ----------

def parse_audio_manifest(list_path: Path, cache_dir: Path) -> list[dict]:
    """1-to-1 algorithmic copy of MainWindow._parse_ffmpeg_clip_audio_manifest
    so the probe runs without importing the heavy app module."""
    if not list_path.exists():
        return []
    entries: list[dict] = []
    with list_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            raw_path = (row[0] or "").strip()
            try:
                start_time = float(row[1])
                end_time = float(row[2])
            except Exception:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = cache_dir / path
            if not path.exists() or path.stat().st_size <= 0:
                continue
            try:
                wall_end_mtime = float(path.stat().st_mtime)
            except Exception:
                wall_end_mtime = 0.0
            file_duration = max(0.0, end_time - start_time)
            entries.append({
                "path": path,
                "start_time": start_time,
                "end_time": end_time,
                "file_duration": file_duration,
                "wall_end_mtime": wall_end_mtime,
                "wall_start_mtime": (
                    wall_end_mtime - file_duration
                    if wall_end_mtime > 0 else 0.0
                ),
            })
    latest_by_path: dict[str, dict] = {}
    for entry in entries:
        key = str(entry["path"])
        prior = latest_by_path.get(key)
        if prior is None or entry["end_time"] > prior["end_time"]:
            latest_by_path[key] = entry
    ordered = sorted(latest_by_path.values(), key=lambda e: e["end_time"])
    prev_wall_end = 0.0
    for entry in ordered:
        wall_end = float(entry.get("wall_end_mtime", 0.0) or 0.0)
        file_duration = float(entry.get("file_duration", 0.0) or 0.0)
        if wall_end <= 0:
            continue
        if prev_wall_end > 0:
            entry["wall_start_mtime"] = prev_wall_end
        else:
            entry["wall_start_mtime"] = wall_end - file_duration
        prev_wall_end = wall_end
    return ordered


# ---------- Synthetic-scenario builder ----------

def build_scenario(
    cache_dir: Path,
    *,
    wrap_count: int,
    segment_seconds: float,
    cache_started_wall: float,
    elapsed_wall: float,
    rate_factor: float = 1.0,
) -> Path:
    """Build a temp cache dir + CSV manifest that mimics what ffmpeg
    would have produced for the given session age.

    `elapsed_wall` = how long the cache has been recording in WALL
    seconds. `rate_factor` < 1.0 simulates the WASAPI under-delivery
    we've seen in real sessions (e.g. 0.95). With drift, segments
    take segment_seconds / rate_factor wall seconds to fill.

    Returns the manifest CSV path."""
    list_path = cache_dir / "audio_segments.csv"
    wall_per_segment = segment_seconds / rate_factor
    # How many segment-completion events fired so far. ffmpeg writes
    # one CSV row per completion.
    num_completions = int(elapsed_wall / wall_per_segment)
    rows = []
    for completion_idx in range(num_completions):
        slot = completion_idx % wrap_count
        path = cache_dir / f"audio_seg_{slot:03d}.aac"
        # Each completion advances the segment muxer's relative
        # time by segment_seconds — that's what shows up in the CSV.
        start_time = completion_idx * segment_seconds
        end_time = (completion_idx + 1) * segment_seconds
        # The file mtime is the wall time the muxer finished
        # writing this segment.
        wall_end_for_this_completion = cache_started_wall + (completion_idx + 1) * wall_per_segment
        # Write 1 byte so the parser's st_size > 0 check passes.
        path.write_bytes(b"\x00")
        os.utime(path, (wall_end_for_this_completion, wall_end_for_this_completion))
        rows.append((str(path), f"{start_time:.6f}", f"{end_time:.6f}"))
    with list_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row)
    return list_path


def real_wall_start_for_slot_last_write(
    *, slot: int, wrap_count: int, segment_seconds: float,
    cache_started_wall: float, elapsed_wall: float, rate_factor: float = 1.0,
) -> float:
    """The TRUE wall_start of the LAST-written content for a given
    slot, given the scenario parameters. Used to compute the
    expected chained wall_start values."""
    wall_per_segment = segment_seconds / rate_factor
    num_completions = int(elapsed_wall / wall_per_segment)
    # Find the LATEST completion that landed in this slot.
    latest_completion = None
    for completion_idx in range(num_completions):
        if completion_idx % wrap_count == slot:
            latest_completion = completion_idx
    if latest_completion is None:
        return -1.0
    # Wall start = wall time the PREVIOUS segment finished (or 0
    # for the very first segment).
    if latest_completion == 0:
        return cache_started_wall
    return cache_started_wall + latest_completion * wall_per_segment


# ---------- Probe runner ----------

def run_scenario(
    name: str, *, wrap_count: int, elapsed_wall: float, rate_factor: float = 1.0,
) -> dict:
    segment_seconds = 10.0
    cache_started_wall = 1_700_000_000.0  # epoch-ish, doesn't matter
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        list_path = build_scenario(
            cache_dir,
            wrap_count=wrap_count,
            segment_seconds=segment_seconds,
            cache_started_wall=cache_started_wall,
            elapsed_wall=elapsed_wall,
            rate_factor=rate_factor,
        )
        parsed = parse_audio_manifest(list_path, cache_dir)
        # Compute expected wall_starts from the scenario parameters
        # and compare to what the parser produced.
        errors = []
        for entry in parsed:
            path = entry["path"]
            slot = int(path.stem.split("_")[-1])
            expected_wall_start = real_wall_start_for_slot_last_write(
                slot=slot,
                wrap_count=wrap_count,
                segment_seconds=segment_seconds,
                cache_started_wall=cache_started_wall,
                elapsed_wall=elapsed_wall,
                rate_factor=rate_factor,
            )
            actual_wall_start = entry["wall_start_mtime"]
            err = actual_wall_start - expected_wall_start
            entry["_expected_wall_start"] = expected_wall_start
            entry["_actual_wall_start"] = actual_wall_start
            entry["_error_seconds"] = err
            if abs(err) > 0.5:  # half-second tolerance
                errors.append((path.name, err, expected_wall_start, actual_wall_start))
        # Simulate the export's window-selection for 60 s / 2 m / 5 m
        # clips ending at "now" (cache_started_wall + elapsed_wall).
        now_wall = cache_started_wall + elapsed_wall
        window_results = {}
        for clip_seconds in (60, 120, 300):
            window_start = now_wall - clip_seconds - 0.5  # slop
            window_end = now_wall + 0.5
            selected = [
                e for e in parsed
                if not (e["wall_end_mtime"] < window_start or e["wall_start_mtime"] > window_end)
            ]
            if selected:
                first = selected[0]
                last = selected[-1]
                # Cap selected coverage to the requested window so a
                # 60s clip with 10 selected segments isn't reported as
                # "100s coverage" (each segment is ~10.5s but the
                # actual clip output is clamped to clip_seconds).
                full_span = last["wall_end_mtime"] - first["wall_start_mtime"]
                covered = min(full_span, float(clip_seconds))
                expected_min_coverage = min(clip_seconds, elapsed_wall) * 0.95
                window_results[clip_seconds] = {
                    "n_selected": len(selected),
                    "first_wall_start": first["wall_start_mtime"] - cache_started_wall,
                    "last_wall_end": last["wall_end_mtime"] - cache_started_wall,
                    "covered_seconds": covered,
                    "coverage_ok": covered >= expected_min_coverage,
                }
            else:
                window_results[clip_seconds] = {"n_selected": 0, "coverage_ok": False}
        return {
            "name": name,
            "wrap_count": wrap_count,
            "elapsed_wall": elapsed_wall,
            "rate_factor": rate_factor,
            "n_segments_parsed": len(parsed),
            "max_chain_error_seconds": max((abs(e["_error_seconds"]) for e in parsed), default=0.0),
            "errors": errors,
            "window_results": window_results,
        }


def test_audio_manifest_wrap_scenarios():
    """Pytest-style runner that prints + asserts per-scenario
    results. The print output is what matters for the user; the
    asserts just gate CI-style runs."""
    scenarios = [
        # name, wrap_count, elapsed_wall, rate_factor
        ("A v1 baseline: 8 wrap, 70s elapsed",     8, 70, 1.0),
        ("B v1 baseline + drift: 8 wrap, 70s @95%", 8, 70, 0.95),
        ("C v2 new: 32 wrap, 70s (no wrap yet)",   32, 70, 1.0),
        ("D v2 new: 32 wrap, 200s (no wrap yet)",  32, 200, 1.0),
        ("E v2 new: 32 wrap, 320s (just about to wrap)", 32, 320, 1.0),
        ("F v2 new: 32 wrap, 400s (~8 segs rewritten)", 32, 400, 1.0),
        ("G v2 + drift: 32 wrap, 400s @95%",       32, 400, 0.95),
        ("H v2 long: 32 wrap, 600s (~28 rewrites)", 32, 600, 1.0),
        ("I v2 long + drift: 32 wrap, 600s @95%",  32, 600, 0.95),
    ]
    print()
    print("=" * 78)
    print("CLIP AUDIO MANIFEST WRAP PROBE")
    print("=" * 78)
    overall_failures = []
    for name, wrap, elapsed, rate in scenarios:
        result = run_scenario(name, wrap_count=wrap, elapsed_wall=elapsed, rate_factor=rate)
        print(f"\n{name}")
        print(f"  segments parsed:  {result['n_segments_parsed']}")
        print(f"  max chain error:  {result['max_chain_error_seconds']:.3f} s")
        for clip_s, info in result["window_results"].items():
            print(f"  clip {clip_s:3d} s window: n_selected={info['n_selected']:>2}, "
                  f"covered={info.get('covered_seconds', 0):.1f}s, "
                  f"coverage_ok={info['coverage_ok']}")
        if result["errors"]:
            print(f"  [!] CHAIN ERRORS: {len(result['errors'])}")
            for fname, err, exp, act in result["errors"][:5]:
                print(f"      {fname}: error={err:+.3f}s (expected={exp:.3f}, got={act:.3f})")
            overall_failures.append(name)
        else:
            print(f"  [ok] chain consistent (all entries within 0.5s tolerance)")
        # Window-selection failures
        for clip_s, info in result["window_results"].items():
            if not info["coverage_ok"] and clip_s <= int(elapsed * 0.95):
                overall_failures.append(f"{name} -> {clip_s}s clip coverage")
    print()
    print("=" * 78)
    if overall_failures:
        print(f"FAILURES: {len(overall_failures)}")
        for f in overall_failures:
            print(f"  - {f}")
    else:
        print("ALL SCENARIOS PASSED")
    print("=" * 78)
    # Soft assert — print is the actual report
    assert not overall_failures, f"Manifest wrap scenarios failed: {overall_failures}"


if __name__ == "__main__":
    test_audio_manifest_wrap_scenarios()
