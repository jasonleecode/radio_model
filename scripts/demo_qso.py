"""End-to-end closed loop: two stations hold a QSO *through the audio pipeline*.

Every turn exercises all three layers:
    reply text --synth--> audio --(channel)--> perception --> text+SNR --> decision

Station A calls CQ; Station B answers; they run to sign-off.

Perception backend is pluggable (the whole point of the layered design):
    --model runs/crnn.pt   use the trained CRNN; else the DSP baseline.

--jitter sends with human "fist" timing. The baseline collapses on heavy
jitter; the model keeps copying. Run both to see the difference:

    python scripts/demo_qso.py                          # DSP, clean
    python scripts/demo_qso.py --jitter 0.3             # DSP struggles
    .../torch_env/bin/python scripts/demo_qso.py --jitter 0.3 --model runs/crnn.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radio_cw.channel import add_white_noise  # noqa: E402
from radio_cw.decision import QSOMachine, QSOState, StationInfo  # noqa: E402
from radio_cw.perception import DSPPerception, ModelPerception  # noqa: E402
from radio_cw.synth import SynthConfig, synthesize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, default=None, help="add white noise (dB)")
    ap.add_argument("--wpm", type=float, default=20.0)
    ap.add_argument("--jitter", type=float, default=0.0, help="human fist timing jitter")
    ap.add_argument("--sr", type=int, default=8000)
    ap.add_argument("--model", default=None, help="checkpoint for model perception")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--wav-dir", default=None, help="dump each transmission as WAV")
    ap.add_argument("--max-turns", type=int, default=12)
    args = ap.parse_args()

    perception = (ModelPerception(args.model, args.device) if args.model
                  else DSPPerception())
    backend = f"model({args.model})" if args.model else "DSP baseline"

    # The model was trained with SNR in [-3, 25] dB and never saw *noiseless*
    # audio, so a silent channel is out-of-distribution and decodes poorly.
    # Real bands always have noise, so default the model path to a light floor.
    snr = args.snr
    if args.model and snr is None:
        snr = 30.0
        print("[note] model needs in-distribution noise; defaulting --snr 30")

    rng = np.random.default_rng(0)
    a = QSOMachine(StationInfo(callsign="BG1ABC", name="LI", qth="BEIJING"))
    b = QSOMachine(StationInfo(callsign="W1AW", name="BOB", qth="NYC"))
    freqs = {"BG1ABC": 600.0, "W1AW": 720.0}  # distinct sidetones, as on a band

    wav_dir = Path(args.wav_dir) if args.wav_dir else None
    if wav_dir:
        wav_dir.mkdir(parents=True, exist_ok=True)

    sender, listener = a, b
    msg = a.call_cq()
    print(f"--- QSO start (perception={backend}, snr={snr}, "
          f"wpm={args.wpm}, jitter={args.jitter}) ---\n")

    for turn in range(args.max_turns):
        # Execution layer: sender renders its message to audio (with fist timing).
        cfg = SynthConfig(wpm=args.wpm, sidetone_hz=freqs[sender.me.callsign],
                          sample_rate=args.sr, timing_jitter=args.jitter,
                          weight_bias=1.0 + 0.5 * args.jitter)
        audio, _ = synthesize(msg, cfg, rng)
        if snr is not None:
            audio = add_white_noise(audio, snr, rng)
        if wav_dir:
            pcm = np.int16(np.clip(audio, -1, 1) * 32767)
            wavfile.write(wav_dir / f"turn{turn:02d}_{sender.me.callsign}.wav",
                          args.sr, pcm)

        # Perception layer (model or DSP): decode text + SNR metadata.
        decoded, meta = perception(audio, args.sr)
        print(f"[{sender.me.callsign} TX]   {msg}")
        print(f"[{listener.me.callsign} copy] {decoded}   (snr~{meta.snr_db:.0f}dB)")

        # Decision layer: listener decides the reply.
        reply = listener.process(decoded, meta.snr_db)
        print(f"   {listener.me.callsign} state -> {listener.state.name}\n")

        if reply is None:
            break
        msg, sender, listener = reply, listener, sender

    done = {a.state, b.state}
    print("--- QSO complete ---" if QSOState.DONE in done or QSOState.CLOSING in done
          else "--- QSO stalled ---")
    print(f"final states: {a.me.callsign}={a.state.name}  {b.me.callsign}={b.state.name}")


if __name__ == "__main__":
    main()
