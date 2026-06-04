"""Tests for the unified perception layer."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radio_cw.dsp_frontend import Metadata  # noqa: E402
from radio_cw.perception import DSPPerception, ModelPerception  # noqa: E402
from radio_cw.synth import SynthConfig, synthesize  # noqa: E402

SR = 8000
CKPT = Path(__file__).resolve().parent.parent / "runs" / "crnn.pt"


def test_dsp_perception_returns_text_and_metadata():
    audio, _ = synthesize("CQ DE W1AW K", SynthConfig(wpm=20, sample_rate=SR))
    p = DSPPerception()
    text, meta = p(audio, SR)
    assert text == "CQ DE W1AW K"
    assert isinstance(meta, Metadata)
    assert meta.wpm is not None


def test_perception_backends_share_signature():
    # Both backends are callables (audio, sr) -> (str, Metadata).
    audio, _ = synthesize("PARIS", SynthConfig(sample_rate=SR))
    text, meta = DSPPerception()(audio, SR)
    assert isinstance(text, str) and isinstance(meta, Metadata)


@pytest.mark.skipif(not CKPT.exists(), reason="no trained checkpoint")
def test_model_perception_loads_and_decodes():
    audio, _ = synthesize("CQ DE W1AW K", SynthConfig(wpm=20, sample_rate=SR))
    # add light noise so input is in-distribution for the model
    rng = np.random.default_rng(0)
    from radio_cw.channel import add_white_noise
    audio = add_white_noise(audio, 20.0, rng)
    p = ModelPerception(str(CKPT), device="cpu")
    text, meta = p(audio, SR)
    assert isinstance(text, str) and isinstance(meta, Metadata)
    assert "W1AW" in text  # callsign should survive at 20 dB
