"""Tests for the live CW engine: activity gate + headless decode/reply path.

These never open an audio device. The gate and LiveEngine.feed() are pure;
transmit() is monkeypatched so the auto-reply path can be checked offline.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radio_cw.channel import add_white_noise  # noqa: E402
from radio_cw.decision import StationInfo  # noqa: E402
from radio_cw.live import CWActivityGate, GateConfig, LiveConfig, LiveEngine  # noqa: E402
from radio_cw.perception import DSPPerception  # noqa: E402
from radio_cw.synth import SynthConfig, synthesize  # noqa: E402

SR = 8000


def _sil(s):
    return np.zeros(int(s * SR), np.float32)


def _stream(texts, gap=1.5, wpm=22, noise_db=18, seed=0):
    rng = np.random.default_rng(seed)
    parts = [_sil(0.5)]
    for t in texts:
        a, _ = synthesize(t, SynthConfig(wpm=wpm, sidetone_hz=650))
        parts += [a, _sil(gap)]
    return add_white_noise(np.concatenate(parts), noise_db, rng)


def _run_gate(stream, cfg=None):
    gate = CWActivityGate(cfg or GateConfig())
    segs = []
    for i in range(0, len(stream), 400):
        segs += gate.push(stream[i:i + 400])
    f = gate.flush()
    if f is not None:
        segs.append(f)
    return segs


def test_gate_segments_two_transmissions():
    segs = _run_gate(_stream(["CQ DE W1AW K", "BG1ABC DE W1AW 599"]))
    assert len(segs) == 2


def test_gate_silence_yields_nothing():
    assert _run_gate(_sil(3.0)) == []


def test_gate_segments_decode_correctly():
    texts = ["CQ DE W1AW K", "BG1ABC DE W1AW 599"]
    segs = _run_gate(_stream(texts))
    dsp = DSPPerception()
    decoded = [dsp(s, SR)[0] for s in segs]
    assert decoded == texts


def test_engine_feed_decodes():
    got = []
    eng = LiveEngine(DSPPerception(), LiveConfig(auto_reply=False))
    eng.on_decode = lambda t, m: got.append(t)
    eng.feed(_stream(["CQ DE W1AW K"]))
    assert got == ["CQ DE W1AW K"]


def test_engine_auto_reply(monkeypatch):
    sent = []
    eng = LiveEngine(
        DSPPerception(),
        LiveConfig(station=StationInfo("BG1ABC"), auto_reply=True),
    )
    # don't touch audio hardware
    monkeypatch.setattr(eng, "transmit", lambda text: sent.append(text))
    eng.feed(_stream(["CQ CQ DE W1AW W1AW K"]))
    assert sent, "auto-reply should have produced a transmission"
    assert "BG1ABC" in sent[0] and "W1AW" in sent[0]
