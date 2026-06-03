"""DSP front-end: audio -> tone envelope + metadata.

This is the *metadata sidecar* of the perception layer (see docs/perception_design.md).
It does not decode text -- it produces the physical measurements the decision
layer needs (sidetone frequency, SNR, WPM) plus a clean tone envelope that the
baseline decoder (decode_dsp.py) consumes.

Pipeline:
  1. find the dominant sidetone frequency via a periodogram peak in the CW band
  2. narrow band-pass around it, take the analytic (Hilbert) envelope
  3. smooth the envelope to the frame rate used downstream
  4. estimate SNR (on-tone vs off-tone power) and WPM (from element timing)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch


@dataclass
class Metadata:
    """Physical measurements fed to the decision layer."""

    sidetone_hz: float
    snr_db: float
    wpm: float | None  # None if too few elements to estimate


@dataclass
class Envelope:
    """Smoothed tone-energy envelope, plus the frame rate it lives at."""

    values: np.ndarray  # float32, >= 0
    frame_rate: float   # samples per second of `values`


def detect_sidetone(
    audio: np.ndarray, sample_rate: int, f_lo: float = 250.0, f_hi: float = 1200.0
) -> float:
    """Return the dominant frequency (Hz) within the CW band via a periodogram."""
    nperseg = min(len(audio), 2048)
    freqs, psd = welch(audio, fs=sample_rate, nperseg=nperseg)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return 0.5 * (f_lo + f_hi)
    bfreqs, bpsd = freqs[band], psd[band]
    return float(bfreqs[int(np.argmax(bpsd))])


def tone_envelope(
    audio: np.ndarray,
    sample_rate: int,
    freq: float,
    *,
    bw_hz: float = 200.0,
    frame_rate: float = 200.0,
    smooth_ms: float = 5.0,
) -> Envelope:
    """Band-pass around `freq`, take the Hilbert envelope, smooth, downsample.

    frame_rate ~200 Hz (5 ms/frame) gives several frames per dit even at 40 WPM
    (dit = 30 ms), which is enough timing resolution for the baseline decoder.
    """
    nyq = 0.5 * sample_rate
    lo = max(freq - bw_hz / 2, 20.0) / nyq
    hi = min(freq + bw_hz / 2, nyq - 1.0) / nyq
    b, a = butter(4, [lo, hi], btype="band")
    filtered = filtfilt(b, a, audio)
    env = np.abs(hilbert(filtered)).astype(np.float64)

    # Moving-average smoothing over smooth_ms.
    win = max(1, int(round(smooth_ms * 1e-3 * sample_rate)))
    if win > 1:
        kernel = np.ones(win) / win
        env = np.convolve(env, kernel, mode="same")

    # Downsample to frame_rate by averaging within each frame.
    hop = max(1, int(round(sample_rate / frame_rate)))
    n_frames = len(env) // hop
    if n_frames == 0:
        return Envelope(values=env.astype(np.float32), frame_rate=float(sample_rate))
    framed = env[: n_frames * hop].reshape(n_frames, hop).mean(axis=1)
    return Envelope(values=framed.astype(np.float32), frame_rate=sample_rate / hop)


def _runs(mask: np.ndarray) -> list[tuple[bool, int]]:
    """Run-length encode a boolean array into [(value, length), ...]."""
    if len(mask) == 0:
        return []
    change = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate(([0], change, [len(mask)]))
    return [(bool(mask[bounds[i]]), int(bounds[i + 1] - bounds[i]))
            for i in range(len(bounds) - 1)]


def estimate_threshold(env: np.ndarray) -> float:
    """Adaptive on/off threshold between the noise floor and the tone level.

    Uses robust percentiles (the envelope is strongly bimodal: off ~ noise,
    on ~ tone). Threshold sits partway up from floor to peak.
    """
    floor = np.percentile(env, 20)
    peak = np.percentile(env, 95)
    return float(floor + 0.45 * (peak - floor))


def estimate_snr_db(env: np.ndarray, threshold: float) -> float:
    """SNR from on-tone vs off-tone power in the envelope."""
    on = env[env >= threshold]
    off = env[env < threshold]
    if on.size == 0 or off.size == 0:
        return 0.0
    sig = float(np.mean(on**2))
    noise = float(np.mean(off**2)) + 1e-12
    ratio = max((sig - noise) / noise, 1e-6)
    return float(10.0 * np.log10(ratio))


def estimate_wpm(env: np.ndarray, frame_rate: float, threshold: float) -> float | None:
    """Estimate WPM from the shortest consistent tone-on run (the dit).

    Clusters on-run durations into dit/dah with a 1D 2-means split and takes
    the dit centroid. Returns None if there aren't enough elements.
    """
    mask = env >= threshold
    on_runs = [n for v, n in _runs(mask) if v]
    if len(on_runs) < 3:
        return None
    durs = np.array(on_runs, dtype=np.float64) / frame_rate  # seconds
    dit_len = _smaller_cluster_mean(durs)
    if dit_len <= 0:
        return None
    return float(1.2 / dit_len)  # PARIS: dit_seconds = 1.2 / wpm


def _smaller_cluster_mean(durs: np.ndarray) -> float:
    """1D 2-means on durations; return the mean of the smaller (dit) cluster.

    Falls back to the overall mean if the values don't separate.
    """
    if len(durs) == 1:
        return float(durs[0])
    lo, hi = durs.min(), durs.max()
    if hi - lo < 1e-9:
        return float(lo)
    c_lo, c_hi = lo, hi
    for _ in range(20):
        mid = 0.5 * (c_lo + c_hi)
        low = durs[durs <= mid]
        high = durs[durs > mid]
        new_lo = low.mean() if low.size else c_lo
        new_hi = high.mean() if high.size else c_hi
        if abs(new_lo - c_lo) < 1e-9 and abs(new_hi - c_hi) < 1e-9:
            break
        c_lo, c_hi = new_lo, new_hi
    return float(c_lo)


def analyze(audio: np.ndarray, sample_rate: int) -> tuple[Envelope, Metadata]:
    """Full front-end: return (envelope, metadata) for a buffered transmission."""
    audio = np.asarray(audio, dtype=np.float64)
    freq = detect_sidetone(audio, sample_rate)
    env = tone_envelope(audio, sample_rate, freq)
    thr = estimate_threshold(env.values)
    snr = estimate_snr_db(env.values, thr)
    wpm = estimate_wpm(env.values, env.frame_rate, thr)
    return env, Metadata(sidetone_hz=freq, snr_db=snr, wpm=wpm)
