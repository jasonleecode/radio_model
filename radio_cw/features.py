"""Narrow-band log-spectrogram features for the perception model.

Per docs/perception_design.md, CW needs the *opposite* STFT trade-off from
speech: short window + short hop, because a 40-WPM dit is only 30 ms and a
long window would smear element edges. We keep only the band around the CW
sidetone -- the model sees a few-bin-tall time-frequency strip and learns to
track a drifting tone and reject adjacent signals.

Defaults at 8 kHz:
    win = 128 samples (16 ms), hop = 32 samples (4 ms)  -> ~250 Hz frame rate
    freq resolution 8000/128 = 62.5 Hz
    band 250-1200 Hz -> ~15 bins
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft


@dataclass
class FeatureConfig:
    sample_rate: int = 8000
    win: int = 128
    hop: int = 32
    f_lo: float = 250.0
    f_hi: float = 1200.0
    log_floor: float = 1e-6  # clamp before log to avoid -inf
    normalize: bool = True    # per-clip mean/std normalization


def spectrogram(audio: np.ndarray, cfg: FeatureConfig | None = None) -> np.ndarray:
    """Return a (n_freq, n_frames) float32 log-magnitude narrow-band spectrogram.

    Frequency axis first so it maps to a CNN's channel/height dimension; the
    dataset transposes to (time, freq) when feeding an RNN.
    """
    cfg = cfg or FeatureConfig()
    noverlap = cfg.win - cfg.hop
    f, _, zxx = stft(
        audio,
        fs=cfg.sample_rate,
        nperseg=cfg.win,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    band = (f >= cfg.f_lo) & (f <= cfg.f_hi)
    mag = np.abs(zxx[band, :])
    logmag = np.log(np.maximum(mag, cfg.log_floor)).astype(np.float32)
    if cfg.normalize:
        mean = logmag.mean()
        std = logmag.std() + 1e-6
        logmag = (logmag - mean) / std
    return logmag


def feature_dim(cfg: FeatureConfig | None = None) -> int:
    """Number of frequency bins kept (the model's input feature dimension)."""
    cfg = cfg or FeatureConfig()
    freqs = np.fft.rfftfreq(cfg.win, d=1.0 / cfg.sample_rate)
    return int(np.sum((freqs >= cfg.f_lo) & (freqs <= cfg.f_hi)))


def frame_rate(cfg: FeatureConfig | None = None) -> float:
    cfg = cfg or FeatureConfig()
    return cfg.sample_rate / cfg.hop
