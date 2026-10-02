"""Case table for the whisper hallucination filter.

This file used to be a hand-run script: it carried its OWN copy of
`_strip_whisper_hallucinations` plus the three stopword/stock-phrase
constants, ran a case table at import time, and printed a "N passed, M
failed" tally. Two things were wrong with that.

It collected no tests -- every top-level statement ran at import and the
assertions were `print` calls, so pytest exited 5 ("no tests ran") and a
suite gate counted the file as a failure while the table's real result
was only ever visible to whoever ran the file by hand.

Worse, it exercised the copy. The production filter lives in
`hgr.app.integration.noop_engine`, and a duplicated implementation can
stay green while the shipped one rots -- the table would have reported 28
passes no matter what production did. So import the real function. (It
was verified identical to the copy on all 28 cases before the copy was
deleted, so this conversion changed no expectation.)
"""
from __future__ import annotations

import unittest

from hgr.app.integration.noop_engine import _strip_whisper_hallucinations


cases = [
    # (input, expected, description)
    ("Thank you", "", "pure 2-word stopword"),
    ("the", "", "single stopword"),
    ("you the", "", "two stopwords"),
    ("Thank you.", "", "with punctuation"),

    ("The meeting is at noon", "The meeting is at noon", "legitimate leading 'The'"),
    ("I want to dictate", "I want to dictate", "legitimate 4 words starting with stopwords"),
    ("A quick test", "A quick test", "legitimate leading 'A'"),

    ("pretty good so far the", "pretty good so far", "trailing 'the' stripped"),
    ("hello you", "hello you", "2-word legitimate keep (not all stopwords? wait 'you' is stopword but 'hello' isn't)"),
    ("it is a test the", "it is a test", "trailing 'the' from 5 words"),

    ("test test test test test", "test", "5-fold consecutive repeat"),
    ("playing playing", "playing", "double repeat"),
    ("Their daughter is playing playing", "Their daughter is playing", "trailing duplicate"),
    ("very very good", "very good", "legitimate 'very very' collapsed (acceptable tradeoff)"),

    ("The meeting is at noon Good afternoon, everyone.", "The meeting is at noon", "whisper stock phrase stripped"),
    ("Good afternoon, everyone.", "", "pure stock phrase"),
    ("thanks for watching", "", "pure stock phrase alt"),
    ("This is real. Thanks for watching!", "This is real.", "stock phrase mid-text"),
    ("Please subscribe to my channel", "to my channel", "partial stock phrase"),

    ("Let's meet at four two", "Let's meet at four two", "non-stopword trailing word kept (limitation)"),

    ("", "", "empty"),
    ("   ", "", "whitespace only"),
    ("Alright, let's test it.", "Alright, let's test it.", "legitimate full sentence"),

    ("Okay, let's test this new dicta- indication method.", "Okay, let's test this new indication method.", "hyphen fragment dropped"),
    ("dicta-", "", "pure hyphen fragment (2 words min check... actually 1 token)"),
    ("well- maybe", "maybe", "fragment + word"),
    ("state-of-the-art", "state-of-the-art", "legitimate hyphenated compound preserved"),
    ("a- b- c- d", "d", "multiple fragments"),
]


class WhisperHallucinationFilterTest(unittest.TestCase):
    def test_case_table(self) -> None:
        """One subTest per row, so a regression names the row it broke
        instead of stopping at the first mismatch."""
        for text, expected, description in cases:
            with self.subTest(case=description, text=text):
                self.assertEqual(_strip_whisper_hallucinations(text), expected)

    def test_table_is_not_empty(self) -> None:
        """Guard the guard: an empty table would make `test_case_table`
        pass vacuously, which is the failure mode this file just came
        out of."""
        self.assertGreaterEqual(len(cases), 28)


if __name__ == "__main__":
    unittest.main()

# Author: Konstantin Markov
