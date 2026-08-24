from __future__ import annotations

import importlib.util
import unittest

# noop_engine pulls in the heavy gesture stack (cv2/PySide6/mediapipe). Skip the
# whole module when those aren't importable, mirroring tests/test_low_fps_mode.py.
_HAS_DEPS = (
    importlib.util.find_spec("cv2") is not None
    and importlib.util.find_spec("PySide6") is not None
)

if _HAS_DEPS:
    from hgr.app.integration.noop_engine import _reconcile_final_edit, _seam_normalize

# whisper_stream only needs numpy + sounddevice (no cv2/PySide6).
_HAS_WHISPER = importlib.util.find_spec("sounddevice") is not None
if _HAS_WHISPER:
    try:
        from hgr.voice.whisper_stream import _adaptive_silence_threshold, _RMS_SILENCE_THRESHOLD
    except Exception:  # pragma: no cover - import-time backend issues
        _HAS_WHISPER = False


def _apply(screen_committed: str, backspace: int, to_type: str) -> str:
    """Simulate the editor: backspace from the end of the live text, then insert."""
    kept = screen_committed[: len(screen_committed) - backspace] if backspace else screen_committed
    return kept + to_type


@unittest.skipUnless(_HAS_DEPS, "noop_engine dependencies (cv2/PySide6) unavailable")
class ReconcileFinalEditTest(unittest.TestCase):
    """Streaming hypothesis -> final reconciliation: the load-bearing math that
    must never duplicate words, never backspace past this utterance's own live
    text, and must collapse to the old commit-only behaviour when no hypotheses
    were typed."""

    def _check_bounds(self, committed: str, committed_chars: int, final_text: str):
        backspace, to_type, common = _reconcile_final_edit(committed, committed_chars, final_text)
        # Safety invariant: never delete more than this utterance's own live text.
        self.assertGreaterEqual(backspace, 0)
        self.assertLessEqual(backspace, committed_chars)
        # common_chars is the length of the agreed prefix that stays on screen.
        self.assertEqual(common, committed_chars - backspace)
        return backspace, to_type, common

    def test_commit_only_no_hypotheses_types_whole_final(self):
        # committed == "" is the HGR_DICTATION_HYPOTHESES=0 / short-utterance path.
        backspace, to_type, common = self._check_bounds("", 0, "hello world")
        self.assertEqual((backspace, to_type, common), (0, "hello world ", 0))
        self.assertEqual(_apply("", backspace, to_type), "hello world ")

    def test_clean_extension_types_only_the_new_tail(self):
        committed = "the quick brown"
        backspace, to_type, _ = self._check_bounds(committed, len(committed), "the quick brown fox")
        self.assertEqual(backspace, 0)
        self.assertEqual(to_type, " fox ")
        self.assertEqual(_apply(committed, backspace, to_type), "the quick brown fox ")

    def test_casing_divergence_does_not_duplicate(self):
        # Greedy live "the", beam final "The": case-insensitive match keeps the
        # live prefix (no re-type) and appends only the new tail.
        committed = "the quick brown"
        backspace, to_type, _ = self._check_bounds(committed, len(committed), "The quick brown fox")
        self.assertEqual(backspace, 0)
        self.assertEqual(to_type, " fox ")
        screen = _apply(committed, backspace, to_type)
        self.assertEqual(screen, "the quick brown fox ")
        self.assertNotIn("brownThe", screen)  # the duplication bug this fixes

    def test_word_divergence_backspaces_only_the_wrong_tail(self):
        committed = "the quick braun"  # greedy misheard "brown"
        backspace, to_type, common = self._check_bounds(committed, len(committed), "the quick brown fox")
        self.assertEqual((backspace, to_type), (6, " brown fox "))  # removes " braun"
        self.assertEqual(common, len("the quick"))
        self.assertEqual(_apply(committed, backspace, to_type), "the quick brown fox ")

    def test_final_shorter_than_live_backspaces_the_extra(self):
        # Hallucination strip can shorten the final below the live text.
        committed = "send the"
        backspace, to_type, _ = self._check_bounds(committed, len(committed), "send")
        self.assertEqual((backspace, to_type), (4, " "))  # removes " the", leaves "send "
        self.assertEqual(_apply(committed, backspace, to_type), "send ")

    def test_identical_final_just_terminates_with_space(self):
        committed = "hello"
        backspace, to_type, _ = self._check_bounds(committed, len(committed), "hello")
        self.assertEqual((backspace, to_type), (0, " "))
        self.assertEqual(_apply(committed, backspace, to_type), "hello ")

    def test_total_divergence_replaces_everything_no_leading_space(self):
        committed = "foo bar"
        backspace, to_type, _ = self._check_bounds(committed, len(committed), "baz qux")
        self.assertEqual((backspace, to_type), (7, "baz qux "))  # k==0 -> no leading space
        self.assertEqual(_apply(committed, backspace, to_type), "baz qux ")

    def test_punctuation_insensitive_prefix_match(self):
        committed = "hello world"
        backspace, to_type, _ = self._check_bounds(committed, len(committed), "hello, world.")
        self.assertEqual(backspace, 0)  # words match ignoring punctuation
        self.assertEqual(to_type, " ")
        self.assertEqual(_apply(committed, backspace, to_type), "hello world ")

    def test_screen_never_duplicates_any_word_across_cases(self):
        cases = [
            ("the quick brown", "the quick brown fox jumps"),
            ("the quick braun", "the quick brown fox"),
            ("I really", "I really think so"),
            ("", "single utterance"),
        ]
        for committed, final in cases:
            with self.subTest(committed=committed, final=final):
                backspace, to_type, _ = self._check_bounds(committed, len(committed), final)
                screen = _apply(committed, backspace, to_type)
                # every final word appears exactly once (no duplicated seam)
                for word in final.split():
                    self.assertEqual(
                        screen.split().count(word), 1, f"{word!r} duplicated in {screen!r}"
                    )


@unittest.skipUnless(_HAS_DEPS, "noop_engine dependencies (cv2/PySide6) unavailable")
class SeamNormalizeTest(unittest.TestCase):
    """Utterance-seam casing + spacing. whisper capitalizes the first word of
    every utterance, so a mid-sentence pause yields a wrong cap ("Free") and the
    boundary space is occasionally dropped ("ofFree")."""

    def test_midsentence_common_word_is_decapitalized(self):
        # prev ends mid-sentence, trailing space already present
        lead, cased = _seam_normalize("Free connections", "plenty of", " ")
        self.assertEqual(lead, "")
        self.assertEqual(cased, "free connections")

    def test_dropped_boundary_space_is_filled(self):
        # last typed char is a letter (space went missing) -> add one
        lead, cased = _seam_normalize("One thing", "but there's", "s")
        self.assertEqual(lead, " ")
        self.assertEqual(cased, "one thing")
        self.assertEqual("but there's" + lead + cased, "but there's one thing")

    def test_sentence_start_is_capitalized(self):
        lead, cased = _seam_normalize("we should ship", "I'd change.", " ")
        self.assertEqual(cased, "We should ship")

    def test_colon_counts_as_sentence_start(self):
        _lead, cased = _seam_normalize("we should", "one thing I'd change:", " ")
        self.assertEqual(cased, "We should")

    def test_no_prior_text_capitalizes_and_no_lead(self):
        lead, cased = _seam_normalize("hello there", "", "")
        self.assertEqual((lead, cased), ("", "Hello there"))

    def test_proper_noun_is_preserved_midsentence(self):
        lead, cased = _seam_normalize("GitHub repo", "pushed it to", " ")
        self.assertEqual((lead, cased), ("", "GitHub repo"))

    def test_acronym_is_preserved_midsentence(self):
        _lead, cased = _seam_normalize("API call", "check the", " ")
        self.assertEqual(cased, "API call")  # all-caps first word untouched

    def test_name_is_preserved_midsentence(self):
        _lead, cased = _seam_normalize("John left", "and then", " ")
        self.assertEqual(cased, "John left")  # not in the common-word set

    def test_no_lead_after_open_bracket(self):
        lead, _cased = _seam_normalize("foo", "see the note (", "(")
        self.assertEqual(lead, "")

    def test_length_is_preserved_so_char_bookkeeping_is_safe(self):
        for text in ("Free connections", "we should", "GitHub repo", "API call"):
            _lead, cased = _seam_normalize(text, "plenty of", " ")
            self.assertEqual(len(cased), len(text))


@unittest.skipUnless(_HAS_WHISPER, "whisper_stream (sounddevice) unavailable")
class AdaptiveSilenceThresholdTest(unittest.TestCase):
    """Adaptive noise-floor silence detection: a fixed 0.003 cutoff lets a noisy
    room's ambient flicker around it so pauses never reach 1s of consecutive
    silence and the utterance runs to the 30s cap. The cutoff must float above
    the room's ambient while staying unchanged in a quiet room."""

    def test_quiet_room_is_unchanged(self):
        # ambient well below the fixed floor -> cutoff collapses to the base
        self.assertAlmostEqual(_adaptive_silence_threshold([0.001] * 50), _RMS_SILENCE_THRESHOLD)

    def test_short_window_uses_fixed_floor(self):
        self.assertEqual(_adaptive_silence_threshold([0.05] * 3), _RMS_SILENCE_THRESHOLD)

    def test_noisy_room_cutoff_clears_ambient(self):
        # 20 ambient blocks @0.005 + 30 speech @0.03
        thresh = _adaptive_silence_threshold([0.005] * 20 + [0.03] * 30)
        self.assertGreater(thresh, 0.005)   # ambient now classified as silence
        self.assertLess(thresh, 0.03)       # speech still classified as speech

    def test_flickering_ambient_cutoff_clears_the_peaks(self):
        window = [0.002, 0.004, 0.006] * 10 + [0.03] * 20
        thresh = _adaptive_silence_threshold(window)
        self.assertGreater(thresh, 0.006)   # above the ambient flicker peaks
        self.assertLess(thresh, 0.03)

    def test_speech_always_separable_from_ambient(self):
        # whatever the room, normal speech (0.02+) must stay above the cutoff
        for ambient in (0.001, 0.004, 0.008):
            window = [ambient] * 25 + [0.04] * 25
            with self.subTest(ambient=ambient):
                self.assertLess(_adaptive_silence_threshold(window), 0.04)


# Author: Konstantin Markov
