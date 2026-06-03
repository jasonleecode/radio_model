"""Baseline DSP decoder: tone envelope -> CW text.

Classic non-learning Morse decoder. Serves three roles (docs/perception_design.md):
  1. gets the loop working immediately
  2. is the control baseline against which the stage-3 model is judged
  3. shares the front-end with the production metadata sidecar

Algorithm: adaptive threshold -> run-length encode on/off -> estimate dit from
the on-runs -> classify elements (dit/dah) and gaps (intra/inter/word) against
the standard 2u / 5u boundaries -> look up characters.

Its known weakness is exactly the model's target: it relies on fixed timing
ratios, so human "fist" jitter degrades it. That contrast is the point.
"""

from __future__ import annotations

import numpy as np

from .dsp_frontend import (
    Envelope,
    _runs,
    _smaller_cluster_mean,
    analyze,
    estimate_threshold,
)
from .morse import morse_to_text


def decode_envelope(env: Envelope, threshold: float | None = None) -> str:
    """Decode a tone envelope to text using estimated element timing."""
    values = env.values
    if threshold is None:
        threshold = estimate_threshold(values)
    mask = values >= threshold
    runs = _runs(mask)
    if not runs:
        return ""

    # Estimate dit length (in frames) from the on-runs.
    on_runs = np.array([n for v, n in runs if v], dtype=np.float64)
    if on_runs.size == 0:
        return ""
    dit = _smaller_cluster_mean(on_runs)
    if dit <= 0:
        return ""

    # Standard decision boundaries, in units of dit length.
    dah_thresh = 2.0 * dit        # on-run: dit vs dah
    inter_thresh = 2.0 * dit      # gap: intra vs inter-char
    word_thresh = 5.0 * dit       # gap: inter-char vs word

    patterns: list[str] = []
    current: list[str] = []

    def flush_char() -> None:
        if current:
            patterns.append("".join(current))
            current.clear()

    for value, length in runs:
        if value:  # tone on
            current.append("." if length < dah_thresh else "-")
        else:       # silence
            if length >= word_thresh:
                flush_char()
                patterns.append(" ")  # word gap marker
            elif length >= inter_thresh:
                flush_char()
            # else: intra-char gap, keep accumulating current character
    flush_char()

    # Drop a leading/trailing word marker from lead/tail silence.
    while patterns and patterns[0] == " ":
        patterns.pop(0)
    while patterns and patterns[-1] == " ":
        patterns.pop()

    return morse_to_text(patterns)


def decode_audio(audio: np.ndarray, sample_rate: int) -> tuple[str, "object"]:
    """Convenience: run the front-end and decode. Returns (text, metadata)."""
    env, meta = analyze(audio, sample_rate)
    text = decode_envelope(env)
    return text, meta
