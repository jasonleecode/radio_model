"""Live CW GUI: listen to CW from the mic, decode it, reply through the speaker.

Point your radio's speaker at the computer mic (or patch it in). Pick the input
device, hit Start, and decoded transmissions scroll in. Auto-reply (off by
default) lets the QSO state machine answer automatically; otherwise type/preset
and send manually. Switch the decoder between the DSP baseline and the trained
model on the fly.

    python scripts/live_gui.py

Runs under the default env (Python 3.13: tkinter + torch CPU + sounddevice).
"""

import queue
import sys
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radio_cw.audio_io import default_devices, list_devices  # noqa: E402
from radio_cw.decision import StationInfo  # noqa: E402
from radio_cw.live import GateConfig, LiveConfig, LiveEngine  # noqa: E402
from radio_cw.perception import DSPPerception, ModelPerception  # noqa: E402

DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "runs" / "crnn.pt"
PRESETS = {  # label -> template ({me}/{dx} filled from station + last peer)
    "CQ": "CQ CQ CQ DE {me} {me} K",
    "599": "{dx} DE {me} = UR RST 599 599 = {me} K",
    "73": "{dx} DE {me} = TNX QSO 73 ES GL = {me} <SK>",
}


class LiveGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("radio_cw — Live CW")
        self.engine: LiveEngine | None = None
        self.events: queue.Queue = queue.Queue()
        self.last_peer = ""
        self._build()
        self.root.after(60, self._drain_events)

    # --- layout ------------------------------------------------------------

    def _build(self) -> None:
        ins, outs = _safe(list_devices, ([], []))
        din, dout = _safe(default_devices, (None, None))
        self.in_devs, self.out_devs = ins, outs

        cfg = ttk.LabelFrame(self.root, text="Setup")
        cfg.grid(row=0, column=0, sticky="ew", padx=6, pady=4)

        ttk.Label(cfg, text="Input").grid(row=0, column=0, sticky="e")
        self.in_var = tk.StringVar()
        self.in_cb = ttk.Combobox(cfg, textvariable=self.in_var, width=38,
                                  values=[f"{i}: {n}" for i, n in ins], state="readonly")
        self.in_cb.grid(row=0, column=1, columnspan=3, sticky="w")
        _select_default(self.in_cb, ins, din)

        ttk.Label(cfg, text="Output").grid(row=1, column=0, sticky="e")
        self.out_var = tk.StringVar()
        self.out_cb = ttk.Combobox(cfg, textvariable=self.out_var, width=38,
                                   values=[f"{i}: {n}" for i, n in outs], state="readonly")
        self.out_cb.grid(row=1, column=1, columnspan=3, sticky="w")
        _select_default(self.out_cb, outs, dout)

        ttk.Label(cfg, text="Decoder").grid(row=2, column=0, sticky="e")
        self.backend = tk.StringVar(value="DSP")
        ttk.Radiobutton(cfg, text="DSP", value="DSP", variable=self.backend).grid(row=2, column=1, sticky="w")
        ttk.Radiobutton(cfg, text="Model", value="Model", variable=self.backend).grid(row=2, column=2, sticky="w")
        self.ckpt_var = tk.StringVar(value=str(DEFAULT_CKPT))
        ttk.Entry(cfg, textvariable=self.ckpt_var, width=28).grid(row=2, column=3, sticky="w")

        # station + tx params
        st = ttk.LabelFrame(self.root, text="My station / TX")
        st.grid(row=1, column=0, sticky="ew", padx=6, pady=4)
        self.call = _labeled_entry(st, "Call", "BG1ABC", 0, 0, 10)
        self.name = _labeled_entry(st, "Name", "OP", 0, 2, 8)
        self.qth = _labeled_entry(st, "QTH", "NIL", 0, 4, 10)
        self.wpm = _labeled_entry(st, "WPM", "20", 1, 0, 5)
        self.tone = _labeled_entry(st, "Tone Hz", "600", 1, 2, 6)
        self.auto = tk.BooleanVar(value=False)
        ttk.Checkbutton(st, text="Auto-reply", variable=self.auto).grid(row=1, column=4, columnspan=2, sticky="w")

        # controls
        ctl = ttk.Frame(self.root)
        ctl.grid(row=2, column=0, sticky="ew", padx=6)
        self.start_btn = ttk.Button(ctl, text="Start", command=self.toggle)
        self.start_btn.grid(row=0, column=0, sticky="w")
        self.status = tk.StringVar(value="idle")
        ttk.Label(ctl, textvariable=self.status).grid(row=0, column=1, padx=8)
        self.meta = tk.StringVar(value="")
        ttk.Label(ctl, textvariable=self.meta).grid(row=0, column=2, padx=8)
        self.level = ttk.Progressbar(ctl, length=120, maximum=0.3)
        self.level.grid(row=0, column=3, padx=8)

        # transcript
        self.text = scrolledtext.ScrolledText(self.root, width=80, height=18, wrap="word")
        self.text.grid(row=3, column=0, sticky="nsew", padx=6, pady=4)
        self.text.tag_config("rx", foreground="#0a0")
        self.text.tag_config("tx", foreground="#06c")
        self.text.tag_config("sys", foreground="#888")
        self.text.configure(state="disabled")

        # manual tx
        tx = ttk.Frame(self.root)
        tx.grid(row=4, column=0, sticky="ew", padx=6, pady=4)
        self.tx_entry = ttk.Entry(tx, width=50)
        self.tx_entry.grid(row=0, column=0, sticky="w")
        self.tx_entry.bind("<Return>", lambda e: self.send_manual())
        ttk.Button(tx, text="Send", command=self.send_manual).grid(row=0, column=1, padx=4)
        for i, label in enumerate(PRESETS):
            ttk.Button(tx, text=label, width=5,
                       command=lambda l=label: self.send_preset(l)).grid(row=0, column=2 + i, padx=2)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)
        self._set_tx_enabled(False)

    # --- engine control ----------------------------------------------------

    def toggle(self) -> None:
        if self.engine is None:
            self.start()
        else:
            self.stop()

    def _make_perception(self):
        if self.backend.get() == "Model":
            return ModelPerception(self.ckpt_var.get(), device="cpu")
        return DSPPerception()

    def start(self) -> None:
        try:
            perception = self._make_perception()
        except Exception as e:  # bad ckpt path, etc.
            self._log(f"cannot load decoder: {e}", "sys")
            return
        cfg = LiveConfig(
            station=StationInfo(self.call.get().upper().strip() or "N0CALL",
                                self.name.get().strip(), self.qth.get().strip()),
            tx_wpm=_f(self.wpm.get(), 20.0), tx_sidetone_hz=_f(self.tone.get(), 600.0),
            auto_reply=self.auto.get(),
            input_device=_dev(self.in_var.get()), output_device=_dev(self.out_var.get()),
            gate=GateConfig(),
        )
        self.engine = LiveEngine(perception, cfg)
        self.engine.on_decode = lambda t, m: self.events.put(("rx", (t, m)))
        self.engine.on_transmit = lambda t: self.events.put(("tx", t))
        self.engine.on_status = lambda m: self.events.put(("sys", m))
        self.engine.on_level = lambda r: self.events.put(("lvl", r))
        try:
            self.engine.start()
        except Exception as e:
            self._log(f"audio start failed: {e}", "sys")
            self.engine = None
            return
        self.start_btn.config(text="Stop")
        self._set_tx_enabled(True)
        self._set_config_enabled(False)

    def stop(self) -> None:
        if self.engine:
            self.engine.stop()
            self.engine = None
        self.start_btn.config(text="Start")
        self.status.set("idle")
        self._set_tx_enabled(False)
        self._set_config_enabled(True)

    # --- transmit ----------------------------------------------------------

    def send_manual(self) -> None:
        text = self.tx_entry.get().strip()
        if text and self.engine:
            self.engine.send_manual(text)
            self.tx_entry.delete(0, "end")

    def send_preset(self, label: str) -> None:
        if not self.engine:
            return
        me = self.call.get().upper().strip() or "N0CALL"
        dx = self.last_peer or "DX"
        self.engine.send_manual(PRESETS[label].format(me=me, dx=dx))

    # --- event pump (worker thread -> Tk main thread) ----------------------

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "rx":
                    text, meta = payload
                    self.last_peer = _peer_from(text) or self.last_peer
                    self._log(f"RX: {text}", "rx")
                    self.meta.set(f"{meta.sidetone_hz:.0f}Hz  "
                                  f"{meta.snr_db:.0f}dB  "
                                  f"{meta.wpm:.0f}wpm" if meta.wpm else
                                  f"{meta.sidetone_hz:.0f}Hz  {meta.snr_db:.0f}dB")
                elif kind == "tx":
                    self._log(f"TX: {payload}", "tx")
                elif kind == "sys":
                    self.status.set(payload)
                elif kind == "lvl":
                    self.level["value"] = min(payload, 0.3)
        except queue.Empty:
            pass
        self.root.after(60, self._drain_events)

    # --- helpers -----------------------------------------------------------

    def _log(self, msg: str, tag: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", msg + "\n", tag)
        self.text.see("end")
        self.text.configure(state="disabled")

    def _set_tx_enabled(self, on: bool) -> None:
        state = "normal" if on else "disabled"
        self.tx_entry.configure(state=state)

    def _set_config_enabled(self, on: bool) -> None:
        state = "readonly" if on else "disabled"
        self.in_cb.configure(state=state)
        self.out_cb.configure(state=state)


# --- module helpers --------------------------------------------------------

def _safe(fn, fallback):
    try:
        return fn()
    except Exception:
        return fallback


def _labeled_entry(parent, label, default, row, col, width):
    ttk.Label(parent, text=label).grid(row=row, column=col, sticky="e")
    var = tk.StringVar(value=default)
    ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=col + 1, sticky="w", padx=2)
    return var


def _select_default(cb, devs, default_idx):
    for pos, (i, _) in enumerate(devs):
        if i == default_idx:
            cb.current(pos)
            return
    if devs:
        cb.current(0)


def _dev(s: str) -> int | None:
    try:
        return int(s.split(":", 1)[0])
    except Exception:
        return None


def _f(s: str, default: float) -> float:
    try:
        return float(s)
    except Exception:
        return default


def _peer_from(text: str) -> str:
    """Pull the sender's callsign (token after DE) to address presets back."""
    from radio_cw.decision import parse_incoming
    return parse_incoming(text, my_call="").peer_call or ""


def main() -> None:
    root = tk.Tk()
    LiveGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
