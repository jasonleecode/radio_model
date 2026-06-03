"""End-to-end closed loop: two stations hold a QSO *through the audio pipeline*.

Every turn exercises all three layers:
    reply text --synth--> audio --(channel)--> decode --> text+SNR --> decision

Station A calls CQ; Station B answers; they run to sign-off. This is the first
milestone where you can hear a full round trip.

    python scripts/demo_qso.py
    python scripts/demo_qso.py --snr 5 --wpm 22
    python scripts/demo_qso.py --wav-dir /tmp/qso   # dump each transmission
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radio_cw.channel import add_white_noise  # noqa: E402
from radio_cw.decision import QSOMachine, QSOState, StationInfo  # noqa: E402
from radio_cw.decode_dsp import decode_audio  # noqa: E402
from radio_cw.synth import SynthConfig, synthesize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, default=None, help="add white noise (dB)")
    ap.add_argument("--wpm", type=float, default=20.0)
    ap.add_argument("--sr", type=int, default=8000)
    ap.add_argument("--wav-dir", default=None, help="dump each transmission as WAV")
    ap.add_argument("--max-turns", type=int, default=12)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    a = QSOMachine(StationInfo(callsign="BG1ABC", name="LI", qth="BEIJING"))
    b = QSOMachine(StationInfo(callsign="W1AW", name="BOB", qth="NYC"))
    # Give the two stations distinct sidetones, as on a real band.
    freqs = {"BG1ABC": 600.0, "W1AW": 720.0}

    wav_dir = Path(args.wav_dir) if args.wav_dir else None
    if wav_dir:
        wav_dir.mkdir(parents=True, exist_ok=True)

    sender, listener = a, b
    msg = a.call_cq()  # A starts
    print(f"--- QSO start (snr={args.snr}, wpm={args.wpm}) ---\n")

    for turn in range(args.max_turns):
        # Execution layer: sender renders its message to audio.
        cfg = SynthConfig(wpm=args.wpm, sidetone_hz=freqs[sender.me.callsign],
                          sample_rate=args.sr)
        audio, _ = synthesize(msg, cfg)
        if args.snr is not None:
            audio = add_white_noise(audio, args.snr, rng)
        if wav_dir:
            pcm = np.int16(np.clip(audio, -1, 1) * 32767)
            wavfile.write(wav_dir / f"turn{turn:02d}_{sender.me.callsign}.wav",
                          args.sr, pcm)

        # Perception layer: listener decodes audio + gets SNR metadata.
        decoded, meta = decode_audio(audio, args.sr)
        print(f"[{sender.me.callsign} TX]   {msg}")
        print(f"[{listener.me.callsign} copy] {decoded}   "
              f"(snr~{meta.snr_db:.0f}dB)")

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
