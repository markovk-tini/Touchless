# Clip V2 Stable Checkpoint — 2026-06-26

Git tag: `clip-v2-stable-2026-06-26`
Commit: `453071f` (clip V2: bump cache + export video quality)

## Why this checkpoint exists

After ~30 iterations of debugging the clip recording feature, this
commit is the FIRST that meets the user's full quality bar across
all three clip durations (60s, 2-min, 5-min):

- **Timing**: sub-300 ms drift across the entire clip duration
- **Audio quality**: mic is clean (no garbling), sys + mic mixed correctly
- **Video quality**: visibly sharp (less fuzzy than the original cache)
- **Full duration**: clips deliver their requested length (60/120/300s)
  with at most 2s safety-net shrink in worst-case audio-short scenarios

If a future change regresses ANY of these properties, revert to this
tag immediately:

```
git checkout clip-v2-stable-2026-06-26
```

Or cherry-pick specific commits FROM this baseline if forward progress
breaks.

## The architecture that made it work

**V2 (OBS-canonical) per-stream audio cache.** Replaces the V1 single
ffmpeg-amix subprocess with TWO independent encoder subprocesses (one
per audio source: sys WASAPI loopback + mic WASAPI input). Each writes
its own AAC segment ring + CSV manifest. At export time, the muxer
reads both streams, applies per-stream `atempo` rate compensation +
`atrim` to the clip wall window, then `amix`es into the final mp4.

Verified by deep-research workflow `whzz8fl3p` (24/25 claims confirmed
3-0 against OBS Studio + WASAPI Win32 + FFmpeg docs).

## Critical commits in the stable chain

| commit | what it does |
|---|---|
| `d5bb651` | V2 dual-stream architecture (cache + export refactor) |
| `a6896f3` | Per-stream atempo rate compensation (`sys_rate`/`mic_rate`) |
| `6872dfb` | Anchor V2 audio to video's wall window (5s-early fix) |
| `3381ef6` | Mic callback mode (root-cause fix for garbling) + 256k AAC + export HQ video |
| `1bb0bee` | Fix audio-vs-video wall-window divergence when safety net fires |
| `d0a56d0` | Cap safety net at 2s + per-stream in-progress segment detection |
| `453071f` | Bump cache CRF 23→20 + export CRF 18→17 (this commit) |

## Key configuration (settings.json fields + env vars)

| knob | default | what it does |
|---|---|---|
| `clip_capture_system_audio` | True | Capture WASAPI loopback in clips |
| `clip_capture_microphone` | True | Capture mic in clips |
| `clip_export_trim_video_to_audio` | True | Safety net: trim video tail if audio coverage shorter (max 2s) |
| `HGR_CLIP_AUDIO_V2` | "1" | V2 architecture on (set "0" to fall back to V1 single-amix) |
| `HGR_CLIP_MIC_V2_CALLBACK` | "1" | Mic in callback mode (set "0" for polling — KNOWN TO GARBLE on UVC mics) |
| `HGR_CLIP_MIC_AAC_BITRATE` | "256k" | Cache mic AAC bitrate (set "192k" to revert) |
| `HGR_CLIP_EXPORT_HQ` | "1" | Export uses slow preset + CRF 17 (set "0" for fast preset + CRF 23) |
| `HGR_CLIP_CACHE_HQ` | "1" | Cache uses CRF 20 (set "0" for CRF 23) |
| `HGR_CLIP_TRIM_VIDEO` | "1" | Safety net enabled (set "0" to disable) |

## What the V2 architecture does NOT do (intentional)

- **No per-packet QPC timestamps from IAudioCaptureClient::GetBuffer.**
  Research recommended this but our mtime-based per-segment timestamps
  are sufficient for sub-300ms sync.
- **No in-memory packet deque.** OBS uses memory; we use disk segments.
  Trade-off: ~60MB extra RAM avoided vs durable cache that survives
  ffmpeg restart.
- **V1 single-amix code path still present** as `HGR_CLIP_AUDIO_V2=0`
  fallback. Can be removed in a future commit once V2 is fully validated
  across all user hardware.

## Diagnostic logs to watch in a healthy clip export

```
[clip-audio-v2] V2 cache running: sys=yes mic=yes sys_anchor=... mic_anchor=...
[wasapi-bridge] WasapiSysLoopbackV2: real=... silence=...
[wasapi-bridge] WasapiMicInputV2 (callback): ... written, queue depth=0, peak=0-1, chunks_dropped=0
[clip-export-v2] in-progress sys: <name> size=NB est_dur=N.NNs extends coverage by N.NNs
[clip-export-v2] in-progress mic: <name> size=NB est_dur=N.NNs extends coverage by N.NNs
[clip-export-v2] sys_segs=N mic_segs=N clip_dur=N.NNs sys_rate=N.NNNNN mic_rate=N.NNNNN
[clip-export-v2] FINAL chain: clip_wall=[...] clip_dur=N.NNs sys_atempo=N.NNNNN mic_atempo=skipped|...
[clip-export] HQ encoder branch active: <h264_nvenc|libx264|...>
[clip-export] ffmpeg rc=0 ...
```

If `mic_atempo=skipped`, mic was within 0.5% of nominal (no compensation
needed). `sys_atempo` typically lands at 0.97-0.99 due to silence-padded
sys bridge.

## Known remaining quirks (not blockers)

- After a 5-min export, the app has been observed to intermittently
  crash or fail-export with AVERROR_EOF. Tracked in a separate fix
  commit. Workaround: retry the clip.
- Cache disk usage grew with the CRF 20 bump (~30-50% per segment).
  Bounded by `segment_wrap` so total cache disk is still bounded.

## How to verify this baseline is intact

```
# 1. Restart app, wait 30s for cache to fill.
# 2. Voice-clip 60s, 2-min, 5-min in sequence.
# 3. Confirm each saved clip has:
#    - Audio and video in sync end-to-end (no drift)
#    - Mic audio is CLEAR (no static/garbling)
#    - Video is sharp (less fuzzy than original)
#    - Duration matches requested (within 2s)
```
