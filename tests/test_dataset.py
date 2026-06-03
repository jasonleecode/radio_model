"""Tests for the stage-3 data pipeline: vocab, corpus, features, dataset."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radio_cw.corpus import CorpusSampler  # noqa: E402
from radio_cw.dataset import (  # noqa: E402
    AugmentConfig,
    CWDataset,
    collate_ctc,
    fit_text_to_duration,
)
from radio_cw.features import FeatureConfig, feature_dim, frame_rate, spectrogram  # noqa: E402
from radio_cw.synth import SynthConfig, estimate_duration, synthesize  # noqa: E402
from radio_cw.vocab import Vocabulary  # noqa: E402


# --- vocab ---------------------------------------------------------------

def test_vocab_roundtrip():
    v = Vocabulary()
    for t in ["CQ DE W1AW", "UR RST 599 = 73", "BG1ABC/P"]:
        assert v.decode(v.encode(t)) == t.upper()


def test_vocab_alias_collapse():
    v = Vocabulary()
    # '<BT>' aliases to '=', '+' to '<AR>', '(' to '<KN>'
    assert v.tokenize("<BT>") == ["="]
    assert v.tokenize("+") == ["<AR>"]
    assert v.tokenize("(") == ["<KN>"]


def test_ctc_collapse():
    v = Vocabulary()
    c = v.tok2id["C"]
    q = v.tok2id["Q"]
    raw = [0, 0, c, c, c, 0, q, 0]
    assert v.greedy_decode(raw) == "CQ"


def test_blank_is_zero():
    assert Vocabulary().blank_id == 0


# --- corpus --------------------------------------------------------------

def test_corpus_in_vocab():
    v = Vocabulary()
    c = CorpusSampler(seed=1)
    for s in c.samples(50):
        assert v.decode(v.encode(s)) == s.upper(), f"not representable: {s!r}"


def test_corpus_deterministic():
    assert CorpusSampler(seed=7).samples(20) == CorpusSampler(seed=7).samples(20)


# --- features ------------------------------------------------------------

def test_spectrogram_shape_and_dim():
    cfg = FeatureConfig()
    audio, tl = synthesize("PARIS", SynthConfig(sample_rate=cfg.sample_rate))
    spec = spectrogram(audio, cfg)
    assert spec.shape[0] == feature_dim(cfg)
    # frame count roughly duration * frame_rate
    assert abs(spec.shape[1] - tl.duration * frame_rate(cfg)) < 5


def test_spectrogram_localizes_tone():
    cfg = FeatureConfig()
    audio, _ = synthesize("PARIS", SynthConfig(sidetone_hz=600, sample_rate=cfg.sample_rate))
    spec = spectrogram(audio, cfg)
    freqs = np.fft.rfftfreq(cfg.win, 1 / cfg.sample_rate)
    band = freqs[(freqs >= cfg.f_lo) & (freqs <= cfg.f_hi)]
    assert abs(band[spec.mean(axis=1).argmax()] - 600) <= cfg.sample_rate / cfg.win


# --- duration estimate ---------------------------------------------------

def test_estimate_duration_matches_synth():
    cfg = SynthConfig(wpm=20)
    for text in ["PARIS", "CQ DE W1AW K", "599 TU 73"]:
        _, tl = synthesize(text, cfg)
        assert abs(estimate_duration(text, cfg) - tl.duration) < 1e-6


def test_fit_text_truncates():
    long = " ".join(["PARIS"] * 50)
    fitted = fit_text_to_duration(long, wpm=15, max_seconds=10.0)
    assert estimate_duration(fitted, SynthConfig(wpm=15)) <= 10.0 + 1.0
    assert len(fitted) < len(long)


# --- dataset -------------------------------------------------------------

def test_dataset_item_shapes():
    v = Vocabulary()
    ds = CWDataset(v, n_samples=10, seed=0)
    s = ds[0]
    assert s["features"].ndim == 2 and s["features"].shape[1] == feature_dim()
    assert s["targets"].dtype == torch.long
    assert s["targets"].numel() == len(v.encode(s["text"]))


def test_dataset_deterministic_per_index():
    v = Vocabulary()
    ds = CWDataset(v, n_samples=10, seed=0)
    a, b = ds[3], ds[3]
    assert torch.equal(a["features"], b["features"]) and a["text"] == b["text"]


def test_collate_ctc_feasible():
    v = Vocabulary()
    ds = CWDataset(v, n_samples=32, seed=0)
    batch = collate_ctc([ds[i] for i in range(8)])
    assert batch["features"].shape[0] == 8
    assert (batch["input_lengths"] >= batch["target_lengths"]).all()
    assert batch["targets"].numel() == int(batch["target_lengths"].sum())


def test_augment_machine_vs_fist():
    rng = np.random.default_rng(0)
    aug = AugmentConfig(machine_prob=1.0)
    p = aug.sample(rng)
    assert p["timing_jitter"] == 0.0 and p["weight_bias"] == 1.0
    aug2 = AugmentConfig(machine_prob=0.0)
    # over several draws a fist sample should show nonzero jitter
    jit = [aug2.sample(rng)["timing_jitter"] for _ in range(10)]
    assert max(jit) > 0
