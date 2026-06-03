"""Measure baseline DSP decoder CER across SNR and WPM.

Establishes the control numbers the stage-3 model must beat.

    python scripts/eval_baseline.py
    python scripts/eval_baseline.py --trials 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radio_cw.channel import add_white_noise  # noqa: E402
from radio_cw.decode_dsp import decode_audio  # noqa: E402
from radio_cw.metrics import cer  # noqa: E402
from radio_cw.synth import SynthConfig, synthesize  # noqa: E402

# Realistic QSO copy: callsigns, exchanges, prosigns, plain text.
TEXTS = [
    "CQ CQ CQ DE BG1ABC BG1ABC K",
    "BG1ABC DE W1AW W1AW",
    "UR RST 599 599 QTH BOSTON",
    "NAME IS JOHN JOHN HW CPY",
    "TNX FER QSO 73 ES GL <SK>",
    "RR FB OM UR 5NN BK",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10, help="noise seeds per cell")
    ap.add_argument("--snrs", type=float, nargs="+",
                    default=[20, 10, 5, 0, -3])
    ap.add_argument("--wpms", type=float, nargs="+", default=[15, 20, 30])
    ap.add_argument("--sr", type=int, default=8000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)

    print("=== Machine timing (clean fist) vs SNR ===")
    print(f"{'WPM':>4} | " + " ".join(f"{s:>6.0f}dB" for s in args.snrs))
    print("-" * (7 + 9 * len(args.snrs)))
    for wpm in args.wpms:
        cells = []
        for snr in args.snrs:
            errs = []
            for text in TEXTS:
                freq = rng.uniform(500, 800)
                clean, _ = synthesize(text, SynthConfig(wpm=wpm, sidetone_hz=freq,
                                                        sample_rate=args.sr))
                for _ in range(args.trials):
                    noisy = add_white_noise(clean, snr, rng)
                    out, _ = decode_audio(noisy, args.sr)
                    errs.append(cer(text, out))
            cells.append(f"{np.mean(errs)*100:6.1f}%")
        print(f"{wpm:>4.0f} | " + " ".join(cells))

    print("\n=== Human fist (timing jitter, weight bias) at 10 dB SNR ===")
    jitters = [0.0, 0.10, 0.20, 0.30]
    print(f"{'WPM':>4} | " + " ".join(f"j={j:>4.2f}" for j in jitters))
    print("-" * (7 + 8 * len(jitters)))
    for wpm in args.wpms:
        cells = []
        for j in jitters:
            errs = []
            for text in TEXTS:
                freq = rng.uniform(500, 800)
                cfg = SynthConfig(wpm=wpm, sidetone_hz=freq, sample_rate=args.sr,
                                  timing_jitter=j, weight_bias=1.0 + 0.5 * j)
                for _ in range(args.trials):
                    clip, _ = synthesize(text, cfg, rng)
                    noisy = add_white_noise(clip, 10.0, rng)
                    out, _ = decode_audio(noisy, args.sr)
                    errs.append(cer(text, out))
            cells.append(f"{np.mean(errs)*100:6.1f}%")
        print(f"{wpm:>4.0f} | " + " ".join(cells))

    print("\nCER = character error rate (lower is better). j = timing jitter std.")
    print("The machine-timing table is the easy regime; the fist table is where")
    print("the stage-3 model must beat the baseline.")


if __name__ == "__main__":
    main()
