"""Live (real-time) CW: microphone -> decode -> optional auto-reply -> speaker.

Turn-based, half-duplex (CW is one-at-a-time), matching the perception design:
a CWActivityGate watches the mic stream, and when a transmission ends (silence
longer than any word gap) it emits the buffered audio for the perception layer
to decode. Optionally the decision state machine replies, synthesized and
played back; the mic is muted during transmit to avoid decoding our own tone.

CWActivityGate is pure DSP and unit-tested with synthetic audio. LiveEngine
wires in sounddevice; its non-audio path can be driven headless via feed().
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

from .audio_io import PIPELINE_RATE, resample_to_pipeline
from .decision import QSOMachine, StationInfo
from .dsp_frontend import Metadata, analyze
from .synth import SynthConfig, synthesize


@dataclass
class GateConfig:
    sample_rate: int = PIPELINE_RATE
    hop: int = 256              # ~32 ms analysis hop
    band: tuple[float, float] = (300.0, 1000.0)
    threshold_k: float = 4.0    # rms must exceed floor * k to count as tone
    end_gap_s: float = 1.0      # silence that ends a transmission (> word gap)
    min_segment_s: float = 0.2  # drop shorter blips (noise)
    max_segment_s: float = 30.0
    pre_roll_s: float = 0.1     # audio kept before onset so first dit isn't clipped
    floor_window_s: float = 3.0  # history for the adaptive noise floor


class CWActivityGate:
    """Streaming segmenter: push audio, get back completed transmissions.

    Emits a segment (np.float32 at sample_rate) when CW activity is followed by
    `end_gap_s` of silence, or when `max_segment_s` is reached.
    """

    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()
        sr = self.cfg.sample_rate
        nyq = 0.5 * sr
        lo, hi = self.cfg.band[0] / nyq, self.cfg.band[1] / nyq
        self._b, self._a = butter(4, [lo, hi], btype="band")
        self._zi = lfilter_zi(self._b, self._a) * 0.0
        self._leftover = np.zeros(0, dtype=np.float32)
        self._rms_hist: deque[float] = deque(
            maxlen=max(1, int(self.cfg.floor_window_s * sr / self.cfg.hop)))
        self._preroll: deque[np.ndarray] = deque(
            maxlen=max(1, int(self.cfg.pre_roll_s * sr / self.cfg.hop)))
        self.active = False
        self._seg: list[np.ndarray] = []
        self._silence_run = 0.0  # seconds of trailing silence while active

    def _hop_seconds(self) -> float:
        return self.cfg.hop / self.cfg.sample_rate

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        """Feed audio (any length). Return list of completed segments."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        buf = np.concatenate([self._leftover, samples])
        hop = self.cfg.hop
        n_hops = len(buf) // hop
        self._leftover = buf[n_hops * hop:]
        emitted: list[np.ndarray] = []
        for i in range(n_hops):
            emitted += self._process_hop(buf[i * hop:(i + 1) * hop])
        return emitted

    def _process_hop(self, hop_raw: np.ndarray) -> list[np.ndarray]:
        filt, self._zi = lfilter(self._b, self._a, hop_raw, zi=self._zi)
        rms = float(np.sqrt(np.mean(filt ** 2)) + 1e-12)
        floor = (np.percentile(self._rms_hist, 20) if self._rms_hist else rms)
        self._rms_hist.append(rms)
        thr = max(floor * self.cfg.threshold_k, 1e-4)
        is_tone = rms > thr
        out: list[np.ndarray] = []

        if not self.active:
            self._preroll.append(hop_raw)
            if is_tone:
                self.active = True
                self._seg = list(self._preroll)  # include pre-roll
                self._preroll.clear()
                self._silence_run = 0.0
        else:
            self._seg.append(hop_raw)
            self._silence_run = 0.0 if is_tone else self._silence_run + self._hop_seconds()
            seg_dur = len(self._seg) * self._hop_seconds()
            if self._silence_run >= self.cfg.end_gap_s or seg_dur >= self.cfg.max_segment_s:
                seg = self._close_segment()
                if seg is not None:
                    out.append(seg)
        return out

    def _close_segment(self) -> np.ndarray | None:
        audio = np.concatenate(self._seg) if self._seg else np.zeros(0, dtype=np.float32)
        self.active = False
        self._seg = []
        self._silence_run = 0.0
        # Trim most of the trailing silence, keep a short tail.
        keep_tail = int(0.15 * self.cfg.sample_rate)
        trail = int(self.cfg.end_gap_s * self.cfg.sample_rate)
        if len(audio) > trail:
            audio = audio[: len(audio) - trail + keep_tail]
        if len(audio) < self.cfg.min_segment_s * self.cfg.sample_rate:
            return None
        return audio

    def flush(self) -> np.ndarray | None:
        """Force-close any in-progress segment (e.g. on stop)."""
        if self.active and self._seg:
            return self._close_segment()
        return None


# --------------------------------------------------------------------------
# Live engine
# --------------------------------------------------------------------------

@dataclass
class LiveConfig:
    station: StationInfo = field(default_factory=lambda: StationInfo("N0CALL"))
    tx_wpm: float = 20.0
    tx_sidetone_hz: float = 600.0
    auto_reply: bool = False
    input_device: int | None = None
    output_device: int | None = None
    gate: GateConfig = field(default_factory=GateConfig)


# A perception backend: (audio_8k, sample_rate) -> (text, Metadata)
Perception = Callable[[np.ndarray, int], "tuple[str, Metadata]"]


class LiveEngine:
    """Drives mic -> gate -> perception -> (optional) reply -> speaker.

    Callbacks (invoked from the worker thread; GUIs should marshal to their
    own thread):
      on_decode(text, meta)   a received transmission was decoded
      on_transmit(text)       we started sending this text
      on_status(msg)          status/log line
      on_level(rms)           input level meter update
    """

    def __init__(self, perception: Perception, cfg: LiveConfig | None = None):
        self.perception = perception
        self.cfg = cfg or LiveConfig()
        self.qso = QSOMachine(self.cfg.station)
        self.gate = CWActivityGate(self.cfg.gate)
        self.on_decode: Callable[[str, Metadata], None] = lambda t, m: None
        self.on_transmit: Callable[[str], None] = lambda t: None
        self.on_status: Callable[[str], None] = lambda m: None
        self.on_level: Callable[[float], None] = lambda r: None
        self._q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stream = None
        self._running = False
        self._transmitting = threading.Event()  # half-duplex: mute mic while set

    # --- segment handling (shared by live and headless feed) ---------------

    def _handle_segment(self, seg_8k: np.ndarray) -> None:
        text, meta = self.perception(seg_8k, PIPELINE_RATE)
        self.on_decode(text, meta)
        if self.cfg.auto_reply:
            reply = self.qso.process(text, meta.snr_db)
            if reply:
                self.transmit(reply)

    def feed(self, samples_8k: np.ndarray) -> None:
        """Headless: push 8 kHz audio through the gate + decode path (no devices)."""
        for seg in self.gate.push(samples_8k):
            self._handle_segment(seg)

    # --- transmit ----------------------------------------------------------

    def transmit(self, text: str) -> None:
        """Synthesize `text` and play it, muting capture for the duration."""
        from .audio_io import play_blocking
        self.on_transmit(text)
        cfg = SynthConfig(wpm=self.cfg.tx_wpm, sidetone_hz=self.cfg.tx_sidetone_hz,
                          sample_rate=PIPELINE_RATE)
        audio, _ = synthesize(text, cfg)
        self._transmitting.set()
        try:
            play_blocking(audio, self.cfg.output_device)
        finally:
            self._transmitting.clear()

    def send_manual(self, text: str) -> None:
        """Manual TX from the UI (runs on a thread so the UI stays responsive)."""
        threading.Thread(target=self.transmit, args=(text,), daemon=True).start()

    # --- live audio loop ---------------------------------------------------

    def start(self) -> None:
        import sounddevice as sd
        from .audio_io import device_samplerate
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(target=self._consume, daemon=True)
        self._worker.start()

        src_rate = device_samplerate(self.cfg.input_device, input=True)
        self._src_rate = src_rate

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                self.on_status(f"audio: {status}")
            if self._transmitting.is_set():
                return  # half-duplex: ignore mic while transmitting
            self._q.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=src_rate, channels=1, dtype="float32",
            device=self.cfg.input_device, callback=callback,
            blocksize=int(0.05 * src_rate))
        self._stream.start()
        self.on_status(f"listening @ {src_rate} Hz (resampling to {PIPELINE_RATE})")

    def _consume(self) -> None:
        while self._running:
            try:
                chunk = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            audio8k = resample_to_pipeline(chunk, self._src_rate)
            self.on_level(float(np.sqrt(np.mean(audio8k ** 2)) + 1e-12))
            for seg in self.gate.push(audio8k):
                self._handle_segment(seg)

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        seg = self.gate.flush()
        if seg is not None:
            self._handle_segment(seg)
        self.on_status("stopped")
