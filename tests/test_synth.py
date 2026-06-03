"""Sanity tests for the morse table and synthesizer."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radio_cw.morse import dit_seconds, morse_to_text, text_to_morse  # noqa: E402
from radio_cw.synth import SynthConfig, synthesize  # noqa: E402


def test_roundtrip_text_morse():
    assert text_to_morse("OK") == ["---", "-.-"]
    assert morse_to_text(text_to_morse("PARIS")) == "PARIS"


def test_word_gap_marker():
    pats = text_to_morse("A B")
    assert " " in pats
    assert morse_to_text(pats) == "A B"


def test_prosign():
    pats = text_to_morse("<AR>")
    assert pats == [".-.-."]
    assert morse_to_text(pats) == "<AR>"


def test_element_counts():
    # "E" is a single dit -> exactly one tone element.
    _, tl = synthesize("E", SynthConfig(sample_rate=8000))
    assert len(tl.elements) == 1
    assert tl.elements[0].kind == "dit"
    # "T" is a single dah.
    _, tl = synthesize("T", SynthConfig(sample_rate=8000))
    assert tl.elements[0].kind == "dah"


def test_dah_is_three_dits():
    cfg = SynthConfig(wpm=20, sample_rate=16000)
    _, tl = synthesize("ET", cfg)  # dit then dah
    dit = tl.elements[0]
    dah = tl.elements[1]
    dit_len = dit.end - dit.start
    dah_len = dah.end - dah.start
    assert abs(dit_len - dit_seconds(20)) < 1e-6
    assert abs(dah_len - 3 * dit_len) < 1e-6


def test_audio_finite_and_bounded():
    audio, _ = synthesize("CQ CQ DE BG1ABC", SynthConfig())
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) <= 0.71  # amplitude default 0.7 + tiny margin


def test_key_envelope_matches_elements():
    cfg = SynthConfig(sample_rate=8000)
    _, tl = synthesize("OK", cfg)
    env = tl.key_envelope()
    # Total on-time should equal sum of element durations.
    on_time = env.sum() / cfg.sample_rate
    expect = sum(e.end - e.start for e in tl.elements)
    assert abs(on_time - expect) < 5e-3
