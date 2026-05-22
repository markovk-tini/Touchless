# Touchless to-do

Active backlog. Bug reports and architectural notes still live in `OPEN_ISSUES.md`; this file holds the planned-work / polish tickets.

---

## v1.1.0b7 polish bundle — new-user / Store-launch readiness

Small QoL bundle. Each item is independently shippable; pick any subset for a given build.

### 4.1.1 First-run "no camera detected" state
**What:** When `cv2.VideoCapture` returns no working device on launch (no camera physically connected, OR Windows Privacy → Camera permission denied for the app), the home page currently shows a black panel with no instruction. New users assume the app is broken.

**Plan:** Add an explicit overlay on the live-view widget when no camera is opened. Copy:
> *"No camera detected. Open Windows Settings → Privacy & security → Camera and allow Touchless, or plug in a webcam, then click Refresh."*
With a "Refresh" button that re-runs camera enumeration. Same pattern for missing microphone on the first voice command.

**Files:** [main_window.py](src/hgr/app/ui/main_window.py) live-view widget, [camera_utils.py](src/hgr/app/camera/camera_utils.py) for the no-device signal.

### 4.1.2 Privacy / local-only banner + Settings → About entry
**What:** Microsoft Store cert reviewers fail apps that silently start the camera/mic without telling the user. Store listing has a privacy-policy URL field but ALSO requires an in-app disclosure.

**Plan:** Two surfaces:
- One-time first-run dialog before the engine starts the camera. Copy: *"Touchless processes camera and microphone data locally on this PC. Nothing leaves your device unless you connect Spotify or enable optional grammar correction (off by default)."* Buttons: "Got it" (latches a `privacy_disclosure_shown` flag) and "Read full policy" (link to project website).
- Permanent Settings → About entry showing the same statement so Store reviewers can find it on demand.

**Files:** [main_window.py](src/hgr/app/ui/main_window.py) startup sequence, [app_config.py](src/hgr/config/app_config.py) for the latch flag.

### 4.1.3 Status text audit (user-readable wording)
**What:** Engine status strings (`mouse_control_text`, `_status`, `voice_status_text`, drawing status) read like dev logs: *"show open hand mouse pose"*, *"mouse waiting for hand"*, *"hold left hand three to turn mouse mode off"*. New users find them terse and confusing.

**Plan:** One pass through every status-string-emitting site. Rewrite to imperative-friendly second-person English. Examples:
- `"show open hand mouse pose"` → *"Open your right palm to start the cursor"*
- `"mouse waiting for hand"` → *"Lost your hand — show it to the camera"*
- `"hold left hand three to turn mouse mode off"` → *"Hold left-hand three to turn mouse mode off"*

Same pass on voice/drawing. Keep the underlying status keys (`"ready"`, `"drag"`, etc.) — they're internal — but the user-facing `control_text` is what shows in pills and the action history.

**Files:** [mouse_gesture.py](src/hgr/debug/mouse_gesture.py), [noop_engine.py](src/hgr/app/integration/noop_engine.py), [voice_status_overlay.py](src/hgr/gesture/ui/voice_status_overlay.py).

### 4.1.4 Auto-update success toast
**What:** After the v1.0.6 → v1.0.7+ auto-update mechanism completes, the new version launches without any signal that an update happened. User just sees a fresh window.

**Plan:** On startup, compare last-saved version-launched flag to current `__version__`. If they differ, show a one-time toast at the bottom-right: *"Updated to vX.Y.Z. View what's new"* (clickable to open release notes, or just dismissable). Persist via `config.last_launched_version`.

**Files:** [app_config.py](src/hgr/config/app_config.py), [main_window.py](src/hgr/app/ui/main_window.py) startup, version source in package metadata.

### 4.1.5 Phone-camera disconnect graceful fallback
**What:** If the phone (paired phone-camera source) navigates away from the Touchless tab or loses Wi-Fi mid-session, the app currently sits on the last frame for several seconds before recovery. New users assume "the app froze" and force-quit.

**Plan:**
- Detect a stale phone-camera frame (no new frame for ~3 s while phone-camera is the active source).
- Show a corner toast: *"Phone disconnected — switching to local camera."*
- Auto-fallback to the previously-selected local camera if one is available; otherwise show the no-camera overlay from 4.1.1.
- When the phone reconnects, optionally re-prompt to switch back.

**Files:** [phone_camera/](src/hgr/debug/phone_camera/), camera source switching in main worker.

### 4.1.6 Tutorial practice-target aim assist
**What:** The pinch-click in the tutorial's mouse practice arena requires landing precisely on each dot. With a slightly trembly cursor that's been smoothed across an absolute-mapping box, beginners miss often, get frustrated, and conclude "mouse mode doesn't work."

**Plan:** Increase the practice-target hitbox by 30–40% beyond the visible dot radius — the visible target stays the same size so the user knows where to aim, but the registration zone is more forgiving. Tutorial-only, not in live mode where Windows' own click-target heuristics already help.

**Files:** [tutorial_window.py](src/hgr/app/ui/tutorial_window.py) `MousePracticeWidget.register_click()` — bump the distance threshold for each target.

### 4.1.7 Settings descriptions audit (kept in mind)
**Status:** noted, not yet ticketed. Several settings already use `_build_expandable_note(brief, detailed)`; coverage gap is in the older rows. Action item: audit every settings row and ensure each has either an expandable note OR a substantive tooltip.

### 4.1.8 Per-user pinch sensitivity slider (kept in mind)
**Status:** noted. `_PINCH_THRESHOLD = 0.42` ([mouse_gesture.py](src/hgr/debug/mouse_gesture.py)) is hand-size-normalized so it works across camera distances, but users with shorter thumbs or larger hands may want to adjust. One config field + one Mouse Control settings row + plumbing through `MouseGestureTracker.__init__`. Defer until real users report difficulty.

### 4.1.9 Error-log readability (kept in mind)
**Status:** noted. When errors do hit the log file, prefer phrasing that a non-technical user can act on. Example: instead of `"spotify play got HTTP 404 body=..."` log `"Spotify play failed: no active device. Open Spotify and start a track to retry. (HTTP 404)"` — keep the diagnostic suffix in parentheses for our debugging, but lead with the human-readable cause.

---

## v1.2 polish backlog (post-1.1.0)

### 4.2.1 Store-install detection (skip in-app updater nag for Store users)
**What:** [release_checker.py](src/hgr/app/updater/updater.py) pops the "Update available" dialog on every launch regardless of how the user installed the app. For users who installed from the Microsoft Store, the Store has its own update channel — our prompt becomes a duplicate. Result: the user sees two update flows for the same version bump, and our updater wastes ~133 MB of bandwidth re-downloading a payload the Store already shipped.

**Plan:** One-time runtime check at updater startup that detects whether the process is running inside a packaged identity (Store install) vs. an unpackaged classic install (website). Use the Windows `GetCurrentPackageFullName` kernel32 API:
- Returns `APPMODEL_ERROR_NO_PACKAGE` (15700) → unpackaged → website install → run the updater as today.
- Returns any other value (typically `ERROR_INSUFFICIENT_BUFFER` 122 with a non-zero length) → we have a package identity → Store install → short-circuit the GitHub poll.

EXE-direct Store submissions still get a package identity assigned at install time (Microsoft wraps registration around our EXE), so this check works even though we don't ship MSIX.

**Code sketch:**
```python
import ctypes

def _is_store_install() -> bool:
    try:
        kernel32 = ctypes.windll.kernel32
        length = ctypes.c_uint32(0)
        result = kernel32.GetCurrentPackageFullName(ctypes.byref(length), None)
        return result != 15700  # APPMODEL_ERROR_NO_PACKAGE → not packaged
    except Exception:
        return False  # be conservative: if check fails, run updater normally
```

Then early-return from the update-check entry point when `_is_store_install()` is True.

**Files:** [src/hgr/app/updater/updater.py](src/hgr/app/updater/updater.py).

**Why deferred:** not required for Store cert — Microsoft accepts both channels coexisting, and the idempotent extract-over-install path means duplicate updates don't break anything. Pure UX polish.

---

## Website polish — needs assets / bigger blocks

Code-side polish (SVG icons, hover lift + glow, scroll fade-in, OG meta tags, button microinteractions, footer beef) shipped already in [hgr-download-page/index.html](hgr-download-page/index.html). The items below need actual content / asset production before they can land.

### Web.1 Hero visual (the highest-impact missing piece)
**What:** The hero is text-only. Modern app sites lead with the product *in motion*.

**Plan:** Produce a 6-10 s muted, looping MP4 (≤ 1.5 MB at 720p, h264) showing one clean Touchless flow — e.g. swipe → mouse-mode pinch → drawing in mid-air. Drop into the hero as a `<video autoplay muted loop playsinline>` next to or under the headline copy. If filming is blocked, fall back to a stylized illustration (Touchless logo + dotted hand-landmark mesh on a dark gradient).

**Files:** [hgr-download-page/index.html](hgr-download-page/index.html) hero `<section>`. Asset goes in `hgr-download-page/hero-loop.mp4` (or `.webm`).

### Web.2 Real Open Graph share image (1200×630)
**What:** Currently `og:image` points at a placeholder `og-image.png` that doesn't exist. Pasting the URL into Discord / Slack / Twitter shows a broken preview.

**Plan:** Design a 1200×630 PNG with the Touchless logo, the headline ("Control your PC with hand gestures and voice"), and a dark gradient matching the site theme. Same image fits the Twitter card.

**Files:** Create `hgr-download-page/og-image.png` and `favicon.png`. The meta tags already reference both.

### Web.3 "See it in action" demo section
**What:** Trust signal between feature grid and closing CTA. Even a single embedded looped video — different from the hero — turns "interesting concept" into "I can see it works."

**Plan:** New `<section class="demo">` with either a self-hosted MP4 / WebM or a YouTube embed. Caption beneath: one-line describing what's happening ("Voice command opening Notepad, then dictating a few words live").

**Files:** [hgr-download-page/index.html](hgr-download-page/index.html). Asset `hgr-download-page/demo.mp4` (or YouTube URL).

### Web.4 Testimonial / quote strip (kept in mind)
**Status:** even one or two real beta-tester quotes with first-name attribution would fill the awkward gap between features and the closing CTA. Defer until enough testers have shipped feedback worth quoting.

### Web.5 Custom logo mark (kept in mind)
**Status:** the "TL" text-in-rounded-square placeholder works but is the most obvious "made by one person" signal. Replace with a real SVG mark (stylized hand, or the wave/dot motif from the app icon at [touchless_icon.png](assets/icons/touchless_icon.png)).

<!-- Author: Konstantin Markov -->
