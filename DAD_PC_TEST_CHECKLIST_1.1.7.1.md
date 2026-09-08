# Touchless 1.1.7.1 — Dad's PC Test Checklist

**Purpose:** Verify the camera-darkness hotfix on a different rig before shipping to Subodh publicly.

---

## Step 1 — Baseline capture (BEFORE installing 1.1.7.1)

While dad's PC still has the current v1.1.7 installed:

1. Launch Touchless. Let it fully start.
2. Note: **is the preview visibly darker than the room?**
   - [ ] Yes, preview looks dim / MediaPipe struggles → **this rig has the bug** (great test case)
   - [ ] No, preview looks normal → this rig may not be affected (still worth testing for regression)
3. Grab the log:
   ```
   %TEMP%\Touchless\
   ```
   Look for the most recent `.log` file. Search inside it for:
   ```
   [r49-short-shutter] evaluate: ...
   [r53-shutter-off-kick] ...
   ```
   Save those lines to compare later.
4. Copy the log to a USB stick or shared folder — labelled `dad_v1.1.7_baseline.log`.
5. Close Touchless completely.

---

## Step 2 — Install 1.1.7.1

1. Copy `Touchless_Installer.exe` (~2.4 GB) to dad's PC via USB stick or shared drive.
2. Double-click the installer. It's signed as "Konstantin Markov" — SmartScreen should NOT complain.
3. Complete the install. Let Touchless auto-launch at the end.

---

## Step 3 — Post-install verification

1. **Visual check** (the most important):
   - Is the preview brightness identical to Windows Camera app on the same machine?
   - Toggle the engine OFF → preview should look the same as before
   - Toggle the engine ON → preview should still look the same (this is the fix)
   - [ ] Engine OFF and Engine ON brightness match → **PASS**
   - [ ] Engine ON is still darker → FAIL, do NOT ship to Subodh yet

2. **Log check**:
   Open the most recent log at `%TEMP%\Touchless\`. Search for:
   ```
   [r53-shutter-off-kick]
   ```
   You should see something like:
   ```
   [r53-shutter-off-kick] armed=... should_kick=... user_chose=False verdict=... post_auto=... post_exp=...
   ```
   - If `should_kick=False` → **the new gate is protecting his camera** ✅
   - If `should_kick=True` → the gate let the writes through; that's fine ONLY if the classifier flagged his camera as generic (verdict=True) or he had FSS on before

3. **Look for the Layer 3 rollback log line** (only if his camera name matches the classifier):
   ```
   [r49-short-shutter] auto-hint luma verify: samples=... median=... too_dark=...
   ```
   - `too_dark=False` → hint stayed, hint helps his camera
   - `too_dark=True` → auto-rolled back, will see `[r49-short-shutter] AUTO-ROLLBACK: ...`

4. **GPU Mode test**:
   - Switch to GPU Mode in Settings → look for:
     ```
     [ffmpeg_capture] 60 fps luma check passed: median=... (threshold=50.0)
     ```
   - If median > 50 → Kiyo-Pro-style pass (60 fps stays)
   - If a downshift happens: `60 fps opened but median luma X < threshold 50.0 — driver likely cut shutter...`
   - Either outcome is fine; both are the fix working correctly

5. **Copy the log** to USB/shared drive — labelled `dad_v1.1.7.1_post.log`.

---

## Step 4 — Compare and decide

Diff the two `[r53-shutter-off-kick]` and `[r49-short-shutter]` lines from the two logs.

Decision tree:
- **Preview brightness identical + logs show `should_kick=False` on both** → hotfix works, ship it to Subodh.
- **Preview brightness IMPROVED after install** → hotfix confirmed on a real bug case, ship it with high confidence.
- **Preview brightness REGRESSED after install** → do NOT ship. Investigate first.

---

## Notes for Konstantin

- Dad's camera name matters. In the log, look for `display_name='...'` in the `[r49-short-shutter] evaluate:` line. If it says something like `Realtek`, `USB2.0 Camera`, `HP HD Camera`, `Chicony`, `Sonix` — those are exactly the drivers the fix targets.
- If dad's camera is a Logitech C920 / Kiyo / Brio — classifier will return `verdict=False` (premium), skip both the hint AND the kick. Behavior identical to what you saw on your Kiyo Pro.
- If dad's camera falls through both lists (`verdict=None`), it's the same class as Subodh's HP HD Camera → this is the primary test case for the gate fix.
