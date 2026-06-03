"""Render a CW message to a WAV file you can listen to.

Usage:
    python scripts/demo_synth.py "CQ CQ DE BG1ABC" --wpm 20 --hz 600 -o cq.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radio_cw.synth import SynthConfig, synthesize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", help="message to send, e.g. 'CQ CQ DE BG1ABC'")
    ap.add_argument("--wpm", type=float, default=20.0)
    ap.add_argument("--hz", type=float, default=600.0, help="sidetone frequency")
    ap.add_argument("--sr", type=int, default=8000, help="sample rate")
    ap.add_argument("--farnsworth", type=float, default=None)
    ap.add_argument("-o", "--out", default="cw.wav")
    args = ap.parse_args()

    cfg = SynthConfig(
        wpm=args.wpm, sidetone_hz=args.hz, sample_rate=args.sr,
        farnsworth_wpm=args.farnsworth,
    )
    audio, tl = synthesize(args.text, cfg)
    pcm = np.int16(np.clip(audio, -1, 1) * 32767)
    wavfile.write(args.out, args.sr, pcm)
    print(f"wrote {args.out}  ({tl.duration:.2f}s, {len(tl.elements)} elements, "
          f"{args.wpm} wpm)")


if __name__ == "__main__":
    main()
