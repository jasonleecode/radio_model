"""Channel impairments for evaluation and (later) training augmentation.

These functions degrade clean synthesized CW to mimic a real HF channel. Each
is a pure function of (audio, params, rng) so the same code drives both the
baseline eval (this stage) and on-the-fly training augmentation (stage 3).

SNR convention: signal power is measured over the whole clip (keyed audio
includes silence, so it's the average tone power weighted by duty cycle); white
noise is scaled to hit the requested ratio.
"""

from __future__ import annotations

import numpy as np


def _signal_power(audio: np.ndarray) -> float:
    return float(np.mean(audio.astype(np.float64) ** 2)) + 1e-12


def add_white_noise(
    audio: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add white Gaussian noise at the requested SNR (dB)."""
    sig = _signal_power(audio)
    noise_power = sig / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=len(audio))
    return (audio + noise).astype(np.float32)


def add_qsb(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    fade_hz: float = 0.3,
    depth: float = 0.6,
) -> np.ndarray:
    """Slow multiplicative fading (QSB), a low-frequency amplitude envelope."""
    t = np.arange(len(audio)) / sample_rate
    phase = rng.uniform(0, 2 * np.pi)
    fade = 1.0 - depth * 0.5 * (1 + np.sin(2 * np.pi * fade_hz * t + phase))
    return (audio * fade).astype(np.float32)


def add_qrm(
    audio: np.ndarray,
    sample_rate: int,
    interferer: np.ndarray,
    rng: np.random.Generator,
    *,
    level_db: float = -6.0,
) -> np.ndarray:
    """Mix in a second (interfering) CW signal at a relative level."""
    n = min(len(audio), len(interferer))
    scale = 10.0 ** (level_db / 20.0)
    out = audio.copy()
    start = rng.integers(0, max(1, len(audio) - n + 1))
    out[start : start + n] += scale * interferer[:n]
    return out.astype(np.float32)


def apply_channel(
    audio: np.ndarray,
    sample_rate: int,
    *,
    snr_db: float | None = None,
    qsb: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply a configurable chain of impairments."""
    rng = rng or np.random.default_rng()
    out = audio.astype(np.float32)
    if qsb:
        out = add_qsb(out, sample_rate, rng)
    if snr_db is not None:
        out = add_white_noise(out, snr_db, rng)
    return out
