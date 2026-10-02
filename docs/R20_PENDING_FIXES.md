# r20 blockers - BOTH FIXED in r20e (2026-09-25)

Kept as the record of what was wrong and why the fix looks the way it
does. Both blockers below were fixed in commit `b67647a`.

**Blocker 1 (dark camera in Lite/GPU mode):** fixed by
`GestureWorker._unstick_inherited_short_shutter`, called inside the
mode-skip branch of `_apply_default_capture_tuning` before it returns.
Writing the tests first caught two flaws in the first draft: it would
have written to a camera already on Auto at a short exposure in a bright
room, and it would have refused a camera stuck under Media Foundation,
where 1.0 means manual but rounds to the DirectShow Auto value. Both are
now covered by `tests/test_unstick_inherited_exposure.py`.

**Blocker 2 (silent clipping switch-off):** the reason is now a persisted
config field, `clip_cache_seeded_off_note`, because the Settings panel is
built during startup before the seeding runs and an in-memory reason
could never reach the tooltip. The disabled pill and its modal fallback
now say Touchless turned clipping off to keep the PC fast, and screen
recording refuses at the entry point rather than after a monitor picker
and a three-second countdown. Covered by
`tests/test_clip_disabled_is_honest.py`.

**Still open from the same trace**, deliberately not taken here and
recorded in OPEN_ISSUES 1.7: the gesture wheel still shows the four clip
and record items at full opacity and announces success before the refusal
reaches the user. Fixing that means touching wheel rendering, which
CLAUDE.md flags as high risk, so it wants its own change.

---

## Original write-up (2026-09-24)

Status: the r20 work through commit 5d6a38f is committed and green on the
reference rig, but it is NOT ready to hand to the field machine. Two
blockers are described below. Both were found by tracing the landed code
against that machine's debug bundle, and both were confirmed by reading
the source, not inferred.

Resume here.

---

## BLOCKER 2: the clip-recorder seeding is silent and its explanation is dead code

`_seed_clip_cache_default_once` correctly switches the always-on recorder
off on a large-desktop / small-VRAM machine, which is the single biggest
CPU saving of the round. But it does so invisibly and it breaks six
user-facing things:

- the instant-clip gesture (default left-hand TWO held 1.0 s)
- gesture wheel: Clip 30 Sec, Clip 1 Min
- gesture wheel: Screen Record, Screen Record Custom
- voice: 'clip that', 'clip the last minute', 'save clip', stop recording

Screenshots are unaffected.

Three specific defects:

1. **The explanatory tooltip can never appear.** `_build_general_clip_section`
   (main_window.py:11641) reads `_clip_cache_seeded_off_reason` while the
   settings panel is built during `MainWindow.__init__`, which runs long
   before the seeder sets that attribute at engine start. Verified by
   character offset: the read is at ~531873, the write at ~1376196. On
   later launches the seeding latch returns early so the attribute is
   never set at all.
2. **The wheel still offers the dead items at full opacity** and the engine
   announces success ('Saving last 30 seconds') before the main window
   discovers clipping is off.
3. **Screen Record makes the user complete a monitor picker and a 3-2-1
   countdown first**, because the gate sits inside `_start_screen_recording`
   downstream of both.

Fix direction: set the reason from persisted state rather than from the
seeding run, so the tooltip works on every launch; grey out or hide the
clip and record wheel items when `clip_cache_enabled` is False; and move
the screen-record gate above the picker and countdown.

---

## BLOCKER 1: un-stick an inherited dark exposure in EVERY mode

Original note:

### Proposed fix: un-stick an inherited dark exposure in EVERY mode

## The problem, restated precisely

`_apply_default_capture_tuning` (noop_engine.py:6924) returns early at
line 6986:

    if self._wants_ffmpeg_cap() or self._low_fps_active:
        ... "[r49-short-shutter] skip: reason=..." ...
        return

`_wants_ffmpeg_cap()` is True whenever lite_mode or gpu_mode is set. The
r53 kick that returns a stuck camera to Auto lives at line 7684, INSIDE
the block that early-return skips. So on any machine whose saved mode is
Lite or GPU, a camera inherited at Exposure=-6/Manual is never un-stuck.

Proof it is not theoretical: all four r20 smoke launches on the dev rig
(gpu_mode=True) logged
    [r49-short-shutter] skip: reason=wants_ffmpeg (exposure hint is dead in this mode)
so the dark-camera fix has never executed, on either rig.

## Why the fix is safe in principle

The kick is directional. `_dshow_auto_exposure_on` only ever hands the
driver back to Auto. It cannot darken a picture, cannot latch a manual
value, and cannot reduce brightness. The thing §3.3 / §3.8 / §3.9 warn
about is the opposite direction: writing a manual short shutter based on
a sticky user flag. This adds no such write.

## The change

Hoist ONLY the un-stick decision above the early return. Do not hoist the
rest of the tuning block, which is genuinely mode-specific.

Conditions, all required, matching the existing 2.4 gate exactly:
  1. classify_camera_shutter_hint(display_name) is True
     (a positively identified generic UVC, not None, not False)
  2. the driver's current exposure reads short (< -2.5) or clearly manual
  3. short shutter is NOT wanted: camera_force_short_shutter is False and
     HGR_FORCE_SHORT_SHUTTER / HGR_FFMPEG_SHORT_SHUTTER are not set
Then: `_dshow_auto_exposure_on(cap, log_tag="[unstick-inherited]")` and
`_note_driver_write("Exposure")`.

Condition 3 is the one the existing kick does not need but this does,
because in ffmpeg modes the preflight may legitimately WANT a short
shutter; we must not fight it.

## What must NOT change

- The expression at line 7684 stays exactly
  `_should_kick = _armed or _known_generic_needs_kick` (checkpoint 2.4).
- No new write is added on the Kiyo Pro path: its Qt name is
  "USB Video Device", which the classifier scores None, so condition 1
  fails and nothing fires. Verified by
  tests/test_camera_name_reaches_classifier.py.
- classifier-driven short shutter stays behind
  HGR_CLASSIFIER_AUTOSHUTTER=1.

## Tests to add

- the un-stick fires for a generic-UVC name reading -6 in GPU mode
- it does NOT fire when the name scores None (the dev rig's Kiyo Pro)
- it does NOT fire when the name scores False (a premium camera)
- it does NOT fire when camera_force_short_shutter is True (Boost on)
- it does NOT fire when the driver reads a normal exposure
- source guard: the un-stick call site index is LESS than the index of
  the `if self._wants_ffmpeg_cap() or self._low_fps_active:` early return
- source guard: line 7684's expression is unchanged

## Verification on the field machine

The single log line that proves it ran:
    [unstick-inherited] ... auto-exposure ON via backend=DSHOW (0.75) ok=True
and at exit:
    [camera-controls] NOT restoring (would re-latch Manual): ... Exposure -6/Manual -> ...
