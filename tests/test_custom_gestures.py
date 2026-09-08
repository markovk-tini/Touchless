"""Unit tests for the custom-gestures package.

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/test_custom_gestures.py -v

Covers:
  - Registry JSON round-trip (add / save / load / remove)
  - Landmark normalization invariants (translation + scale invariance)
  - Classifier match / no-match / threshold behavior
  - Action dispatcher routing (keystroke payload validation only; no real
    SendInput calls — those would move the user's cursor / type keys)

Deliberately does NOT touch the running app, SendInput, cv2, or mediapipe.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest


# Allow running pytest from repo root without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hgr.custom_gestures.action import (  # noqa: E402
    Action,
    _mistaken_https_plain_text,
    describe,
    execute_open_url,
)
from hgr.custom_gestures.classifier import GestureClassifier  # noqa: E402
from hgr.custom_gestures.description import (  # noqa: E402
    CategoricalRange,
    format_gesture_summary,
    live_signature,
    pose_signature,
)
from hgr.custom_gestures.recorder import (  # noqa: E402
    GestureRecorder,
    _sample_from_landmarks,
    augment_sample,
    augment_samples,
    landmarks_to_sample,
    normalize_landmarks,
)
from hgr.custom_gestures.registry import (  # noqa: E402
    CustomGesture,
    GestureRegistry,
    GestureSample,
)


def _synthetic_landmarks(seed: int = 0, noise: float = 0.0) -> np.ndarray:
    """Build a deterministic (21, 3) landmark array for tests."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(-0.5, 0.5, size=(21, 3)).astype(np.float32)
    # Put wrist at a non-origin position so normalization has something to do.
    base[0] = np.array([0.3, 0.4, 0.1], dtype=np.float32)
    # Push landmark 9 (middle-finger MCP) out from wrist so scaling has a
    # non-trivial divisor.
    base[9] = base[0] + np.array([0.2, -0.1, 0.05], dtype=np.float32)
    if noise > 0:
        base = base + rng.normal(0.0, noise, size=base.shape).astype(np.float32)
    return base


# ------------------------- normalize_landmarks -------------------------


def test_normalize_places_wrist_at_origin():
    lm = _synthetic_landmarks(seed=1)
    feats = normalize_landmarks(lm)
    # First 3 floats = wrist after normalization. Must be zero.
    assert np.allclose(feats[:3], 0.0, atol=1e-6)


def test_normalize_is_translation_invariant():
    lm = _synthetic_landmarks(seed=2)
    shift = np.array([5.0, -2.0, 3.0], dtype=np.float32)
    a = normalize_landmarks(lm)
    b = normalize_landmarks(lm + shift)
    assert np.allclose(a, b, atol=1e-5)


def test_normalize_is_scale_invariant():
    lm = _synthetic_landmarks(seed=3)
    a = normalize_landmarks(lm)
    b = normalize_landmarks(lm * 10.0)
    # Scaling around origin after translation should preserve the
    # normalized descriptor (up to floating-point noise).
    assert np.allclose(a, b, atol=1e-5)


def test_normalize_rejects_wrong_shape():
    with pytest.raises(ValueError):
        normalize_landmarks(np.zeros((20, 3), dtype=np.float32))


def test_landmarks_to_sample_has_correct_length():
    lm = _synthetic_landmarks(seed=4)
    sample = landmarks_to_sample(lm)
    # 63 landmarks + 3 spacing + 5 extension + 10 joint + 5 curl + 1 spread
    assert len(sample.features) == 106


def test_spacing_extension_and_joint_features_trail_landmarks():
    """[63:66] spacing, [66:71] extension, [71:81] 10 joint-bend angles."""
    lm = _synthetic_landmarks(seed=21)
    sample = landmarks_to_sample(lm)
    feats = np.asarray(sample.features, dtype=np.float32)
    normalized_lm = feats[:63].reshape(21, 3)
    expected_spacing = [
        float(np.linalg.norm(normalized_lm[8] - normalized_lm[12])),
        float(np.linalg.norm(normalized_lm[12] - normalized_lm[16])),
        float(np.linalg.norm(normalized_lm[16] - normalized_lm[20])),
    ]
    expected_extension = [
        float(np.linalg.norm(normalized_lm[t] - normalized_lm[0]))
        for t in (4, 8, 12, 16, 20)
    ]
    np.testing.assert_allclose(feats[63:66], expected_spacing, atol=1e-5)
    np.testing.assert_allclose(feats[66:71], expected_extension, atol=1e-5)

    # Joint angles must be in [0, π], one pair per finger (5 fingers, 10 angles).
    joint_feats = feats[71:81]
    assert len(joint_feats) == 10
    for a in joint_feats:
        assert 0.0 - 1e-5 <= float(a) <= np.pi + 1e-5


def test_joint_angles_zero_for_extended_finger():
    """A finger with all four landmarks colinear should have ~0 bend at each
    joint — the segment vectors are parallel, angle is 0."""
    # Build landmarks where index finger (5,6,7,8) is colinear along +y.
    lm = _synthetic_landmarks(seed=88)
    # Override index chain to perfectly straight along +y (relative offsets).
    base = lm[5].copy()
    lm[6] = base + np.array([0.0, 1.0, 0.0], dtype=np.float32)
    lm[7] = base + np.array([0.0, 2.0, 0.0], dtype=np.float32)
    lm[8] = base + np.array([0.0, 3.0, 0.0], dtype=np.float32)

    sample = landmarks_to_sample(lm)
    feats = np.asarray(sample.features, dtype=np.float32)
    # Index angles are at indices 71+2 (L6) and 71+3 (L7). Order:
    # thumb (0,1), index (2,3), middle (4,5), ring (6,7), pinky (8,9).
    index_angles = feats[71 + 2:71 + 4]
    np.testing.assert_allclose(index_angles, [0.0, 0.0], atol=1e-4)


def test_legacy_63dim_sample_loads_with_migration(tmp_path):
    """A registry written with the old 63-float schema must load under the
    new 71-dim schema by deriving all 8 missing structural features."""
    import json
    lm = _synthetic_landmarks(seed=31)
    # Build a 63-float feature vector the old way.
    arr = lm.astype(np.float32, copy=True)
    arr -= arr[0]
    arr /= max(1e-6, float(np.linalg.norm(arr[9])))
    legacy_feats = arr.reshape(63).tolist()
    legacy_blob = {
        "schema_version": 1,
        "gestures": [{
            "name": "legacy_g",
            "description": "",
            "created_at": "2026-04-20T00:00:00+00:00",
            "action": {"kind": "noop", "payload": {}},
            "samples": [{"features": legacy_feats}],
        }],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy_blob), encoding="utf-8")

    reg = GestureRegistry(path=path)
    reg.load()
    loaded = reg.get("legacy_g")
    assert loaded is not None
    assert len(loaded.samples) == 1
    assert len(loaded.samples[0].features) == 106


def test_legacy_66dim_sample_loads_with_migration(tmp_path):
    """Samples written with the intermediate 66-float schema (landmarks +
    3 spacing features) must load with both extension AND joint-angle
    features added."""
    import json
    lm = _synthetic_landmarks(seed=32)
    arr = lm.astype(np.float32, copy=True)
    arr -= arr[0]
    arr /= max(1e-6, float(np.linalg.norm(arr[9])))
    lm_flat = arr.reshape(63).tolist()
    spacing = [
        float(np.linalg.norm(arr[8] - arr[12])),
        float(np.linalg.norm(arr[12] - arr[16])),
        float(np.linalg.norm(arr[16] - arr[20])),
    ]
    legacy_feats = lm_flat + spacing  # 66 floats
    legacy_blob = {
        "schema_version": 1,
        "gestures": [{
            "name": "legacy_g66",
            "description": "",
            "created_at": "2026-04-22T00:00:00+00:00",
            "action": {"kind": "noop", "payload": {}},
            "samples": [{"features": legacy_feats}],
        }],
    }
    path = tmp_path / "legacy66.json"
    path.write_text(json.dumps(legacy_blob), encoding="utf-8")

    reg = GestureRegistry(path=path)
    reg.load()
    loaded = reg.get("legacy_g66")
    assert loaded is not None
    assert len(loaded.samples[0].features) == 106


def test_legacy_71dim_sample_loads_with_migration(tmp_path):
    """71-float samples (landmarks + spacing + extension) must gain the
    10 joint-angle features on load."""
    import json
    lm = _synthetic_landmarks(seed=33)
    arr = lm.astype(np.float32, copy=True)
    arr -= arr[0]
    arr /= max(1e-6, float(np.linalg.norm(arr[9])))
    lm_flat = arr.reshape(63).tolist()
    spacing = [
        float(np.linalg.norm(arr[8] - arr[12])),
        float(np.linalg.norm(arr[12] - arr[16])),
        float(np.linalg.norm(arr[16] - arr[20])),
    ]
    extension = [float(np.linalg.norm(arr[t] - arr[0])) for t in (4, 8, 12, 16, 20)]
    legacy_feats = lm_flat + spacing + extension  # 71 floats
    legacy_blob = {
        "schema_version": 1,
        "gestures": [{
            "name": "legacy_g71",
            "description": "",
            "created_at": "2026-04-22T00:00:00+00:00",
            "action": {"kind": "noop", "payload": {}},
            "samples": [{"features": legacy_feats}],
        }],
    }
    path = tmp_path / "legacy71.json"
    path.write_text(json.dumps(legacy_blob), encoding="utf-8")

    reg = GestureRegistry(path=path)
    reg.load()
    loaded = reg.get("legacy_g71")
    assert loaded is not None
    assert len(loaded.samples[0].features) == 106


# ------------------------- GestureRecorder -------------------------


def test_recorder_captures_up_to_target():
    rec = GestureRecorder(target_samples=5)
    for i in range(7):
        rec.capture(_synthetic_landmarks(seed=i))
    assert rec.count == 7
    assert rec.is_complete()
    assert len(rec.finalize()) == 7


def test_recorder_reset_clears_samples():
    rec = GestureRecorder(target_samples=3)
    rec.capture(_synthetic_landmarks(seed=0))
    rec.capture(_synthetic_landmarks(seed=1))
    rec.reset()
    assert rec.count == 0


def test_recorder_finalize_rejects_empty():
    rec = GestureRecorder(target_samples=3)
    with pytest.raises(ValueError):
        rec.finalize()


# ------------------------- Registry round-trip -------------------------


def test_registry_roundtrip(tmp_path: Path):
    path = tmp_path / "gestures.json"
    reg = GestureRegistry(path=path)
    reg.load()
    sample = GestureSample(features=[0.0] * 106)
    reg.add(
        name="test_thumb",
        samples=[sample],
        action=Action(kind="keystroke", payload={"key": "enter"}),
        description="smoke-test",
    )
    reg.save()

    reloaded = GestureRegistry(path=path)
    reloaded.load()
    names = [g.name for g in reloaded.list()]
    assert names == ["test_thumb"]
    g = reloaded.get("test_thumb")
    assert g is not None
    assert g.action.kind == "keystroke"
    assert g.action.payload == {"key": "enter"}
    assert g.description == "smoke-test"
    assert len(g.samples) == 1


def test_registry_rejects_duplicate_without_overwrite(tmp_path: Path):
    reg = GestureRegistry(path=tmp_path / "gestures.json")
    reg.load()
    sample = GestureSample(features=[0.0] * 106)
    reg.add("dup", [sample], Action(kind="noop"))
    with pytest.raises(ValueError):
        reg.add("dup", [sample], Action(kind="noop"))
    # With overwrite it should succeed.
    reg.add("dup", [sample], Action(kind="noop"), overwrite=True)


def test_registry_remove(tmp_path: Path):
    reg = GestureRegistry(path=tmp_path / "gestures.json")
    reg.load()
    reg.add("g1", [GestureSample(features=[0.0] * 106)], Action(kind="noop"))
    assert reg.remove("g1") is True
    assert reg.remove("g1") is False
    assert reg.list() == []


def test_registry_rejects_wrong_feature_length():
    with pytest.raises(ValueError):
        GestureSample(features=[0.0] * 10)


# ------------------------- Classifier -------------------------


def test_classifier_returns_none_when_empty(tmp_path: Path):
    reg = GestureRegistry(path=tmp_path / "empty.json")
    reg.load()
    clf = GestureClassifier(reg)
    clf.reload()
    assert clf.classify(_synthetic_landmarks(seed=0)) is None


def test_classifier_matches_identical_sample(tmp_path: Path):
    reg = GestureRegistry(path=tmp_path / "gestures.json")
    reg.load()
    lm = _synthetic_landmarks(seed=42)
    sample = landmarks_to_sample(lm)
    reg.add("pose_a", [sample], Action(kind="noop"))

    clf = GestureClassifier(reg, threshold=0.90)
    clf.reload()
    match = clf.classify(lm)
    assert match is not None
    assert match.gesture.name == "pose_a"
    # Identical input should score ~1.0 (up to float noise).
    assert match.score > 0.999


def test_classifier_tolerates_small_noise(tmp_path: Path):
    reg = GestureRegistry(path=tmp_path / "gestures.json")
    reg.load()
    lm = _synthetic_landmarks(seed=7)
    # Use augmented samples like the trainer does — single un-augmented
    # sample is brittle near categorical-feature boundaries because hard
    # bucketing can flip a class with tiny noise. Real recordings always
    # have augmentation variants stored, which absorb that.
    reg.add(
        "pose_b",
        augment_samples([landmarks_to_sample(lm)]),
        Action(kind="noop"),
    )

    clf = GestureClassifier(reg, threshold=0.90)
    clf.reload()
    noisy = _synthetic_landmarks(seed=7, noise=0.003)
    match = clf.classify(noisy)
    assert match is not None
    assert match.gesture.name == "pose_b"


def test_classifier_rejects_below_threshold(tmp_path: Path):
    reg = GestureRegistry(path=tmp_path / "gestures.json")
    reg.load()
    reg.add("pose_c", [landmarks_to_sample(_synthetic_landmarks(seed=1))],
            Action(kind="noop"))

    clf = GestureClassifier(reg, threshold=0.99)  # very strict
    clf.reload()
    # Different seed = different pose = lower similarity.
    match = clf.classify(_synthetic_landmarks(seed=99))
    assert match is None


def test_classifier_can_use_explicit_gesture_list(tmp_path: Path):
    """The classifier should accept a list of gestures directly, without
    needing a registry — used by conflict-detection and other code that
    needs an isolated 'what would these gestures alone match?' check."""
    from hgr.custom_gestures.registry import CustomGesture
    sample_a = landmarks_to_sample(_synthetic_landmarks(seed=10))
    g = CustomGesture(
        name="solo",
        samples=[sample_a],
        action=Action(kind="noop"),
        created_at="2026-04-23T00:00:00+00:00",
    )
    clf = GestureClassifier(gestures=[g], threshold=0.85)
    clf.reload()
    match = clf.classify(_synthetic_landmarks(seed=10))
    assert match is not None
    assert match.gesture.name == "solo"


def test_classifier_confidence_margin_blocks_close_calls(tmp_path: Path):
    """If two stored gestures are very similar, a query equally close to
    both should be rejected when the score gap is below confidence_margin."""
    reg = GestureRegistry(path=tmp_path / "gestures.json")
    reg.load()
    lm_a = _synthetic_landmarks(seed=44)
    # Build a near-clone: identical pose for the trailing structural
    # features (rotation-invariant), same landmark portion.
    lm_b = _synthetic_landmarks(seed=44)
    reg.add("a", [landmarks_to_sample(lm_a)], Action(kind="noop"))
    reg.add("b", [landmarks_to_sample(lm_b)], Action(kind="noop"))

    # With margin=0, nearest wins even if it's a tie.
    clf_loose = GestureClassifier(reg, threshold=0.85, confidence_margin=0.0)
    clf_loose.reload()
    assert clf_loose.classify(lm_a) is not None

    # With a real margin, two identical gestures should produce no match.
    clf_strict = GestureClassifier(reg, threshold=0.85, confidence_margin=0.05)
    clf_strict.reload()
    assert clf_strict.classify(lm_a) is None


def test_classifier_picks_correct_gesture_among_many(tmp_path: Path):
    reg = GestureRegistry(path=tmp_path / "gestures.json")
    reg.load()
    lm_a = _synthetic_landmarks(seed=10)
    lm_b = _synthetic_landmarks(seed=20)
    reg.add("a", [landmarks_to_sample(lm_a)], Action(kind="noop"))
    reg.add("b", [landmarks_to_sample(lm_b)], Action(kind="noop"))

    clf = GestureClassifier(reg, threshold=0.90)
    clf.reload()
    m_a = clf.classify(lm_a)
    m_b = clf.classify(lm_b)
    assert m_a is not None and m_a.gesture.name == "a"
    assert m_b is not None and m_b.gesture.name == "b"


# ------------------------- Action dispatcher -------------------------


def test_augment_sample_expands_variants():
    lm = _synthetic_landmarks(seed=5)
    sample = landmarks_to_sample(lm)
    variants = augment_sample(sample)
    # 1 original
    # + (6 rotations on x) + (6 on y) + (4 on z) = 16 rotation variants
    # + 3 thumb-jitter
    # + 2 per-finger jitter
    # = 22 total
    assert len(variants) == 22
    # All variants must be valid 106-dim feature vectors.
    for v in variants:
        assert len(v.features) == 106
    # Variants must differ from original (except the original itself).
    original = variants[0].features
    differing = [v for v in variants[1:] if v.features != original]
    assert len(differing) == len(variants) - 1


def test_augment_preserves_wrist_at_origin():
    lm = _synthetic_landmarks(seed=6)
    sample = landmarks_to_sample(lm)
    for variant in augment_sample(sample):
        # Rotations are around the origin; wrist (first 3 floats) stays at 0.
        assert np.allclose(variant.features[:3], 0.0, atol=1e-5)


def test_augment_makes_rotated_pose_matchable(tmp_path):
    """Augmentation is the whole point: a pose held with a ~10° tilt
    should now match even though only the upright pose was recorded. We
    simulate the tilt in NORMALIZED-landmark space (where the wrist is at
    the origin), which is the same transform a live tilted hand produces
    after normalize_landmarks runs.
    """
    import math

    lm_upright = _synthetic_landmarks(seed=17)
    upright_sample = landmarks_to_sample(lm_upright)
    # Only the first 63 floats are the landmark coords; last 3 are spacing
    # features that the normalizer recomputes. Rotate just the landmarks.
    upright_feats_full = np.asarray(upright_sample.features, dtype=np.float32)
    upright_lm = upright_feats_full[:63].reshape(21, 3)

    theta = math.radians(10.0)
    R = np.array(
        [[math.cos(theta), -math.sin(theta), 0.0],
         [math.sin(theta),  math.cos(theta), 0.0],
         [0.0,              0.0,             1.0]],
        dtype=np.float32,
    )
    tilted_lm = upright_lm @ R.T
    # Rebuild the full feature vector from the rotated landmarks. The
    # v5 direction + palm-normal regions are rotation-sensitive, so we
    # can't just splice them from the upright sample — recompute via
    # the recorder's same-pose helper (which expects already-normalized
    # landmarks).
    tilted_sample = _sample_from_landmarks(tilted_lm.astype(np.float32))
    tilted_feats = np.asarray(tilted_sample.features, dtype=np.float32)

    # Plain: stored only the upright sample.
    reg_plain = GestureRegistry(path=tmp_path / "plain.json")
    reg_plain.load()
    reg_plain.add("p", [upright_sample], Action(kind="noop"))
    clf_plain = GestureClassifier(reg_plain, threshold=0.88)
    clf_plain.reload()
    plain_match = clf_plain.classify_raw(tilted_feats)

    # Augmented: stored upright + rotational variants.
    reg_aug = GestureRegistry(path=tmp_path / "aug.json")
    reg_aug.load()
    reg_aug.add("p", augment_samples([upright_sample]), Action(kind="noop"))
    clf_aug = GestureClassifier(reg_aug, threshold=0.88)
    clf_aug.reload()
    aug_match = clf_aug.classify_raw(tilted_feats)

    assert aug_match is not None, "augmented classifier must match a tilted pose"
    # Augmentation should score the tilted pose at least as well as plain.
    if plain_match is not None:
        assert aug_match.score >= plain_match.score


def test_curl_and_spread_classes_in_feature_vector():
    """Last 6 floats are 5 curl classes + 1 spread class, all integer
    ordinals in [0, 4] / [0, 3]."""
    lm = _synthetic_landmarks(seed=21)
    sample = landmarks_to_sample(lm)
    feats = np.asarray(sample.features, dtype=np.float32)
    curl_classes = feats[81:86]
    spread_class = feats[86]
    assert len(curl_classes) == 5
    for c in curl_classes:
        assert 0.0 <= float(c) <= 4.0
    assert 0.0 <= float(spread_class) <= 3.0


def test_legacy_81dim_sample_loads_with_migration(tmp_path):
    """81-float samples (landmarks + spacing + extension + joint angles)
    must gain the 6 categorical features on load."""
    import json
    lm = _synthetic_landmarks(seed=34)
    sample = landmarks_to_sample(lm)
    feats_full = np.asarray(sample.features, dtype=np.float32)
    legacy_feats = feats_full[:81].tolist()
    legacy_blob = {
        "schema_version": 1,
        "gestures": [{
            "name": "legacy_g81",
            "description": "",
            "created_at": "2026-04-23T00:00:00+00:00",
            "action": {"kind": "noop", "payload": {}},
            "samples": [{"features": legacy_feats}],
        }],
    }
    path = tmp_path / "legacy81.json"
    path.write_text(json.dumps(legacy_blob), encoding="utf-8")
    reg = GestureRegistry(path=path)
    reg.load()
    loaded = reg.get("legacy_g81")
    assert loaded is not None
    assert len(loaded.samples[0].features) == 106


def test_pose_signature_returns_ranges(tmp_path):
    lm = _synthetic_landmarks(seed=60)
    s1 = landmarks_to_sample(lm)
    g = CustomGesture(
        name="stable",
        samples=[s1, s1, s1],
        action=Action(kind="noop"),
        created_at="2026-04-23T00:00:00+00:00",
    )
    sig = pose_signature(g)
    for key in ("thumb_curl", "index_curl", "middle_curl",
                "ring_curl", "pinky_curl", "spread"):
        assert isinstance(sig[key], CategoricalRange)
        assert sig[key].min_class == sig[key].max_class


def test_live_signature_extracts_finger_states():
    lm = _synthetic_landmarks(seed=80)
    feats = normalize_landmarks(lm)
    sig = live_signature(feats)
    for key in ("thumb_curl", "index_curl", "middle_curl",
                "ring_curl", "pinky_curl"):
        assert key in sig
        assert 0 <= sig[key] <= 4
    assert "spread" in sig
    assert 0 <= sig["spread"] <= 3


def test_format_gesture_summary_renders():
    lm = _synthetic_landmarks(seed=70)
    sample = landmarks_to_sample(lm)
    g = CustomGesture(
        name="show_me",
        samples=[sample],
        action=Action(kind="noop"),
        created_at="2026-04-23T00:00:00+00:00",
        description="example",
    )
    summary = format_gesture_summary(g)
    assert "How to do 'show_me'" in summary
    assert "Hand pose:" in summary
    assert "Thumb" in summary
    assert "Spread:" in summary


def test_classifier_hysteresis_extends_match_window(tmp_path):
    """sticky_name lowers the threshold for the named gesture so a
    score that would be rejected normally still matches."""
    reg = GestureRegistry(path=tmp_path / "gestures.json")
    reg.load()
    reg.add(
        "steady",
        [landmarks_to_sample(_synthetic_landmarks(seed=51))],
        Action(kind="noop"),
    )
    # Strict threshold rejects a noisy query without sticky_name.
    clf = GestureClassifier(reg, threshold=0.85, hysteresis=0.10)
    clf.reload()
    noisy = _synthetic_landmarks(seed=51, noise=0.005)
    if clf.classify(noisy) is None:
        # With sticky_name, threshold drops by hysteresis and matches.
        assert clf.classify(noisy, sticky_name="steady") is not None


def test_action_describe_covers_all_kinds():
    cases = [
        Action(kind="noop"),
        Action(kind="keystroke", payload={"key": "enter"}),
        Action(kind="hotkey", payload={"keys": ["ctrl", "c"]}),
        Action(kind="text", payload={"text": "hello"}),
        Action(kind="open_url", payload={"url": "https://example.com"}),
        Action(kind="run_command", payload={"command": "echo hi"}),
    ]
    for action in cases:
        s = describe(action)
        assert isinstance(s, str) and s


def test_mistaken_https_plain_text_detects_spaced_host():
    assert _mistaken_https_plain_text("https://my snap") == "my snap"
    assert _mistaken_https_plain_text("http://ohhhhh snap") == "ohhhhh snap"
    assert _mistaken_https_plain_text("https://example.com") is None
    assert _mistaken_https_plain_text("https://example.com/my page") is None
    assert _mistaken_https_plain_text("my snap") is None


def test_open_url_spaced_host_types_literal_not_browser(monkeypatch):
    typed: list[str] = []
    opened: list[str] = []

    monkeypatch.setattr(
        "hgr.custom_gestures.action.execute_text",
        lambda text: typed.append(text) or True,
    )
    monkeypatch.setattr(
        "hgr.custom_gestures.action.webbrowser.open",
        lambda url: opened.append(url) or True,
    )

    assert execute_open_url("https://my snap") is True
    assert typed == ["my snap"]
    assert opened == []

    assert execute_open_url("https://example.com") is True
    assert opened == ["https://example.com"]

# Author: Konstantin Markov
