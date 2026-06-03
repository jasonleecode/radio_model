"""Tests for the DSP front-end, baseline decoder, channel, and metrics."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radio_cw.channel import add_white_noise  # noqa: E402
from radio_cw.decode_dsp import decode_audio  # noqa: E402
from radio_cw.dsp_frontend import analyze  # noqa: E402
from radio_cw.metrics import cer, edit_distance  # noqa: E402
from radio_cw.synth import SynthConfig, synthesize  # noqa: E402

SR = 8000


def test_edit_distance_basic():
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "abd") == 1
    assert edit_distance("", "abc") == 3


def test_cer():
    assert cer("HELLO", "HELLO") == 0.0
    assert cer("HELLO", "HELLP") == 0.2
    assert cer("", "") == 0.0


def test_frontend_detects_frequency():
    audio, _ = synthesize("PARIS", SynthConfig(wpm=20, sidetone_hz=700, sample_rate=SR))
    _, meta = analyze(audio, SR)
    assert abs(meta.sidetone_hz - 700) < 20  # within periodogram resolution


def test_frontend_estimates_wpm():
    audio, _ = synthesize("CQ TEST DE A1AA", SynthConfig(wpm=22, sample_rate=SR))
    _, meta = analyze(audio, SR)
    assert meta.wpm is not None
    assert abs(meta.wpm - 22) < 4  # within ~15%


def test_decode_clean_machine_code():
    for text in ["CQ CQ DE BG1ABC K", "599 TU 73", "PARIS"]:
        audio, _ = synthesize(text, SynthConfig(wpm=20, sidetone_hz=650, sample_rate=SR))
        out, _ = decode_audio(audio, SR)
        assert out == text


def test_decode_robust_to_white_noise():
    # Machine timing should survive moderate noise nearly perfectly.
    rng = np.random.default_rng(0)
    text = "CQ CQ DE W1AW K"
    audio, _ = synthesize(text, SynthConfig(wpm=20, sidetone_hz=600, sample_rate=SR))
    noisy = add_white_noise(audio, 10.0, rng)
    out, _ = decode_audio(noisy, SR)
    assert cer(text, out) < 0.05


def test_white_noise_lowers_snr_estimate():
    rng = np.random.default_rng(1)
    audio, _ = synthesize("PARIS PARIS", SynthConfig(sample_rate=SR))
    _, hi = analyze(audio, SR)
    _, lo = analyze(add_white_noise(audio, 0.0, rng), SR)
    assert hi.snr_db > lo.snr_db
