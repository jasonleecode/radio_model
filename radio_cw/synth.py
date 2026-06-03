"""Execution layer: text -> keyed CW audio.

Deterministic DSP. Given text, a speed (WPM) and a sidetone frequency, produce
a mono float32 waveform of the keyed sidetone, with raised-cosine envelope
ramps to suppress key clicks (the broadband splatter a hard on/off edge makes).

Besides the audio it returns a ``Timeline`` of element events. That timeline is
the ground truth we will reuse to label synthetic training data for the
perception model, so the synthesizer doubles as a data generator.

Optional Farnsworth timing stretches the inter-character and word gaps while
keeping element speed constant -- standard for sending slow practice copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .morse import dit_seconds, text_to_morse


@dataclass
class Element:
    """One keyed event in the output, in seconds.

    kind is 'dit', 'dah' for tone-on events and 'gap' is implied by the space
    between elements. We only record tone-on spans; silence is everything else.
    """

    start: float
    end: float
    kind: str  # 'dit' | 'dah'


@dataclass
class Timeline:
    sample_rate: int
    duration: float
    elements: list[Element] = field(default_factory=list)

    def key_envelope(self, n: int | None = None) -> np.ndarray:
        """Render an ideal (rectangular) on/off key signal at sample_rate.

        Useful as a clean target for the perception layer. ``n`` overrides the
        length in samples (defaults to full duration).
        """
        if n is None:
            n = int(round(self.duration * self.sample_rate))
        env = np.zeros(n, dtype=np.float32)
        for e in self.elements:
            a = int(round(e.start * self.sample_rate))
            b = int(round(e.end * self.sample_rate))
            env[a : min(b, n)] = 1.0
        return env


@dataclass
class SynthConfig:
    wpm: float = 20.0
    sidetone_hz: float = 600.0
    sample_rate: int = 8000
    ramp_ms: float = 5.0          # raised-cosine rise/fall to avoid key clicks
    amplitude: float = 0.7
    farnsworth_wpm: float | None = None  # if set and < wpm, stretch gaps
    lead_silence_s: float = 0.1   # silence padded before first element
    tail_silence_s: float = 0.1   # silence padded after last element
    # --- human "fist" simulation (0 = perfect machine timing) ---
    timing_jitter: float = 0.0    # std of per-element/gap multiplicative jitter
    weight_bias: float = 1.0      # dah/dit ratio multiplier; 1.0 = standard 3:1


def _gaps(cfg: SynthConfig) -> tuple[float, float, float, float, float]:
    """Return element/gap durations in seconds.

    (dit, dah, intra_gap, inter_gap, word_gap)

    With Farnsworth, element and intra-character timing stay at ``wpm`` while
    the inter-character and word gaps are computed at the slower speed.
    """
    u = dit_seconds(cfg.wpm)
    dit, dah, intra = u, 3 * u, u
    if cfg.farnsworth_wpm and cfg.farnsworth_wpm < cfg.wpm:
        # Standard Farnsworth: distribute the extra delay so total reduces to
        # the slow WPM. Simplified here: scale the 3u/7u gaps by the ratio.
        uf = dit_seconds(cfg.farnsworth_wpm)
        inter, word = 3 * uf, 7 * uf
    else:
        inter, word = 3 * u, 7 * u
    return dit, dah, intra, inter, word


def estimate_duration(text: str, cfg: SynthConfig | None = None) -> float:
    """Predict the rendered duration (seconds) without building audio.

    Mirrors the gap logic in ``synthesize`` (ignoring jitter, which is ~0 mean).
    Used to cap/truncate clips so training batches stay a sane length.
    """
    cfg = cfg or SynthConfig()
    dit, dah, intra, inter, word = _gaps(cfg)
    total = cfg.lead_silence_s + cfg.tail_silence_s
    prev_char = False
    for pat in text_to_morse(text):
        if pat == " ":
            total += word
            prev_char = False
            continue
        if prev_char:
            total += inter
        for i, sym in enumerate(pat):
            if i > 0:
                total += intra
            total += dit if sym == "." else dah * cfg.weight_bias
        prev_char = True
    return total


def synthesize(
    text: str,
    cfg: SynthConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, Timeline]:
    """Render ``text`` to (audio, timeline).

    audio: mono float32 in roughly [-amplitude, amplitude].
    timeline: tone-on element spans in seconds (reflects applied jitter).

    With ``cfg.timing_jitter`` > 0 or ``cfg.weight_bias`` != 1 the timing is
    "humanized": each element/gap is perturbed to mimic a hand key (fist). An
    ``rng`` may be supplied for reproducibility.
    """
    cfg = cfg or SynthConfig()
    rng = rng or np.random.default_rng()
    dit, dah, intra, inter, word = _gaps(cfg)
    dah *= cfg.weight_bias  # systematic dah/dit ratio shift (per-operator weight)
    patterns = text_to_morse(text)

    def jitter(dur: float) -> float:
        if cfg.timing_jitter <= 0:
            return dur
        factor = 1.0 + rng.normal(0.0, cfg.timing_jitter)
        return max(dur * factor, 1e-4)

    # Build a list of (duration, is_tone) segments, plus element kinds.
    segments: list[tuple[float, bool]] = []
    kinds: list[str | None] = []  # parallel to tone segments only

    def add(dur: float, tone: bool, kind: str | None = None, jit: bool = True) -> None:
        segments.append((jitter(dur) if jit else dur, tone))
        kinds.append(kind)

    add(cfg.lead_silence_s, False, jit=False)
    prev_was_char = False
    for pat in patterns:
        if pat == " ":
            # word gap: replace the trailing inter-char gap with a word gap.
            if segments and not segments[-1][1]:
                segments.pop()
                kinds.pop()
            add(word, False)
            prev_was_char = False
            continue
        if prev_was_char:
            add(inter, False)
        for i, sym in enumerate(pat):
            if i > 0:
                add(intra, False)
            if sym == ".":
                add(dit, True, "dit")
            else:
                add(dah, True, "dah")
        prev_was_char = True
    add(cfg.tail_silence_s, False, jit=False)

    # Walk segments to build sample buffer and timeline.
    total = sum(d for d, _ in segments)
    n_total = int(round(total * cfg.sample_rate))
    audio = np.zeros(n_total, dtype=np.float32)
    tl = Timeline(sample_rate=cfg.sample_rate, duration=total)

    t = 0.0
    ramp_n = max(1, int(round(cfg.ramp_ms * 1e-3 * cfg.sample_rate)))
    ti = 0  # tone index into kinds
    for (dur, tone), kind in zip(segments, kinds):
        a = int(round(t * cfg.sample_rate))
        b = int(round((t + dur) * cfg.sample_rate))
        if tone:
            seg = _tone(b - a, cfg, ramp_n)
            audio[a : a + len(seg)] += seg
            tl.elements.append(Element(start=t, end=t + dur, kind=kind or "dit"))
        t += dur
        ti += 1
    return audio, tl


def _tone(n: int, cfg: SynthConfig, ramp_n: int) -> np.ndarray:
    """A sidetone burst of n samples with raised-cosine on/off ramps."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    tt = np.arange(n) / cfg.sample_rate
    wave = cfg.amplitude * np.sin(2 * np.pi * cfg.sidetone_hz * tt)
    env = np.ones(n, dtype=np.float64)
    r = min(ramp_n, n // 2)
    if r > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, r)))
        env[:r] = ramp
        env[-r:] = ramp[::-1]
    return (wave * env).astype(np.float32)
