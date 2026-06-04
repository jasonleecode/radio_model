"""Audio device helpers for the live app: enumeration, resampling, playback.

The whole pipeline runs at 8 kHz, but sound cards capture/play at 44.1/48 kHz,
so we resample between the device rate and 8 kHz here. sounddevice is imported
lazily so the rest of the package (and the tests) don't require it.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

PIPELINE_RATE = 8000


def _sd():
    import sounddevice as sd  # lazy: only needed for live audio
    return sd


def list_devices() -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return (input_devices, output_devices) as lists of (index, name)."""
    sd = _sd()
    ins, outs = [], []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            ins.append((i, d["name"]))
        if d["max_output_channels"] > 0:
            outs.append((i, d["name"]))
    return ins, outs


def default_devices() -> tuple[int | None, int | None]:
    sd = _sd()
    try:
        din, dout = sd.default.device
        return din, dout
    except Exception:
        return None, None


def device_samplerate(index: int | None, *, input: bool) -> int:
    """Best samplerate to open a device at (its default, falling back to 48k)."""
    sd = _sd()
    try:
        info = sd.query_devices(index, "input" if input else "output")
        return int(info["default_samplerate"])
    except Exception:
        return 48000


def resample_to_pipeline(x: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample mono audio from src_rate to the 8 kHz pipeline rate."""
    if src_rate == PIPELINE_RATE:
        return x.astype(np.float32)
    g = gcd(src_rate, PIPELINE_RATE)
    up, down = PIPELINE_RATE // g, src_rate // g
    return resample_poly(x, up, down).astype(np.float32)


def resample_from_pipeline(x: np.ndarray, dst_rate: int) -> np.ndarray:
    """Resample 8 kHz pipeline audio up to a device rate for playback."""
    if dst_rate == PIPELINE_RATE:
        return x.astype(np.float32)
    g = gcd(dst_rate, PIPELINE_RATE)
    up, down = dst_rate // g, PIPELINE_RATE // g
    return resample_poly(x, up, down).astype(np.float32)


def play_blocking(audio_8k: np.ndarray, device: int | None = None) -> None:
    """Play 8 kHz mono audio on the given output device and wait for it."""
    sd = _sd()
    rate = device_samplerate(device, input=False)
    out = resample_from_pipeline(audio_8k, rate)
    sd.play(out, samplerate=rate, device=device, blocking=True)
