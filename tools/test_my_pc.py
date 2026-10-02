"""Test My PC — probes the current machine and reports what
performance to expect from each Touchless mode (Default / Lite / GPU).

Run:
    python tools/test_my_pc.py

Reports:
    * System hardware (CPU, RAM, GPU, camera)
    * Per-mode measured inference cost (median + p95)
    * Expected sustained fps under a 60-fps camera cap
    * Whether GPU mode actually engages DirectML or falls back to CPU
    * Recommended mode for this hardware

This is a thin CLI shim over ``hgr.diagnostics.test_my_pc`` — the same
module the in-app Settings > Test My PC button uses.
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path


# Ensure the repo root is on sys.path so `import hgr...` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def main() -> int:
    from hgr.diagnostics.test_my_pc import (
        MODES,
        benchmark_mode,
        probe_system,
        recommend,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=60,
                        help="warmup frames per mode (default 60)")
    parser.add_argument("--measure", type=int, default=300,
                        help="measurement frames per mode (default 300)")
    parser.add_argument("--cam-width", type=int, default=640,
                        help="synthetic camera frame width (default 640)")
    parser.add_argument("--cam-height", type=int, default=480,
                        help="synthetic camera frame height (default 480)")
    args = parser.parse_args()

    print("=" * 70)
    print(" Touchless — Test My PC")
    print("=" * 70)
    print("")

    sys_info = probe_system()
    cpu = sys_info.get("cpu", {})
    ram = sys_info.get("ram", {})
    gpu = sys_info.get("gpu", {})
    cam = sys_info.get("camera", {})

    print("SYSTEM")
    print(f"  OS:      {sys_info.get('os', platform.system())}  "
          f"({sys_info.get('os_version', platform.version())})")
    print(f"  CPU:     {cpu.get('name', 'unknown')}")
    if "physical_cores" in cpu:
        print(f"           {cpu['physical_cores']} physical / "
              f"{cpu.get('logical_cores', '?')} logical cores")
    if "current_mhz" in cpu:
        maxf = cpu.get("max_mhz")
        print(f"           {cpu['current_mhz']} MHz current"
              + (f" / {maxf} MHz max" if maxf else ""))
    if ram:
        print(f"  RAM:     {ram.get('total_gb', '?')} GB total, "
              f"{ram.get('available_gb', '?')} GB available")
    print(f"  GPU:     {gpu.get('name', 'unknown')}")
    if gpu.get("vram_mb"):
        print(f"           {gpu['vram_mb']} MB VRAM")
    dml = gpu.get("directml_available")
    if dml is True:
        print(f"           DirectML: available (ONNX providers: "
              f"{', '.join(gpu.get('onnxruntime_providers', []))})")
    elif dml is False:
        print("           DirectML: NOT available — GPU mode will fall back to CPU MediaPipe")
    if cam.get("devices"):
        print(f"  Cameras: {', '.join(cam['devices'])}")
    print("")

    print("BENCHMARK (synthetic frames)")
    print(f"  {args.cam_width}x{args.cam_height} input,  "
          f"{args.warmup} warmup + {args.measure} measurement frames")
    print("")

    results: list = []
    for m in MODES:
        label = m["label"]

        def _cb(phase, done, total, _label=label):
            if phase == "build" and done == 0:
                print(f"  [{_label}] building detector...", flush=True)
            elif phase == "warmup" and done == total:
                print(f"  [{_label}] warmup {total} frames done", flush=True)
            elif phase == "measure" and done == total:
                print(f"  [{_label}] measuring {total} frames done", flush=True)

        try:
            r = benchmark_mode(
                label,
                model_complexity=m["model_complexity"],
                max_process_width=m["max_process_width"],
                prefer_gpu=m["prefer_gpu"],
                cap_width=args.cam_width,
                cap_height=args.cam_height,
                warmup=args.warmup,
                measure=args.measure,
                progress_cb=_cb,
            )
        except Exception as exc:
            r = {"label": label, "error": f"benchmark threw: {exc!r}"}
        results.append(r)

    print("")
    print("RESULTS")
    for r in results:
        if "error" in r:
            print(f"  {r['label']}:  ERROR — {r['error']}")
            continue
        print(
            f"  {r['label']:<8}  build={r['build_ms']}ms  "
            f"median={r['median_ms']}ms  p95={r['p95_ms']}ms  p99={r['p99_ms']}ms  "
            f"backend={r['backend']}  ceiling@60cam={r['fps_ceiling_at_60cam']}fps"
        )
    print("")

    print("SUMMARY & RECOMMENDATION")
    rec, reason, lines = recommend(results, gpu)
    for ln in lines:
        print(ln)
    print("")
    print(f"  RECOMMENDED MODE: **{rec}** — {reason}")
    print("")

    print("NOTES")
    print("  * Camera fps limits your visible fps regardless of engine speed.")
    print("    If your camera caps at 30 fps (e.g. Kiyo Pro under HDR), no mode")
    print("    can exceed 30 fps — engine ceiling above 30 just means more")
    print("    headroom for other work.")
    if dml is False:
        print("  * GPU mode fell back to CPU. Install a newer graphics driver or")
        print("    check onnxruntime-directml is installed.")
    print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
