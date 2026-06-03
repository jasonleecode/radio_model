"""Inspect the training data pipeline: print a few samples and batch stats.

    python scripts/demo_dataset.py
    python scripts/demo_dataset.py --n 5 --save-spec /tmp/spec.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radio_cw.dataset import CWDataset, collate_ctc  # noqa: E402
from radio_cw.features import feature_dim, frame_rate  # noqa: E402
from radio_cw.vocab import Vocabulary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-spec", default=None, help="save first sample spectrogram PNG")
    args = ap.parse_args()

    v = Vocabulary()
    print(f"vocab={len(v)} tokens  feat_dim={feature_dim()} bins  "
          f"frame_rate={frame_rate():.0f} Hz\n")

    ds = CWDataset(v, n_samples=1000, seed=args.seed)
    for i in range(args.n):
        s = ds[i]
        p = s["params"]
        T, F = s["features"].shape
        kind = "machine" if p["timing_jitter"] == 0 else f"fist j={p['timing_jitter']:.2f}"
        print(f"[{i}] T={T:5d} L={s['targets'].numel():3d} | "
              f"{p['wpm']:.0f}wpm {p['sidetone_hz']:.0f}Hz {p['snr_db']:.0f}dB "
              f"{kind}{' +qsb' if p['qsb'] else ''}")
        print(f"     {s['text']}")

    dl = DataLoader(ds, batch_size=8, collate_fn=collate_ctc)
    b = next(iter(dl))
    print(f"\nbatch features {tuple(b['features'].shape)}  "
          f"targets {tuple(b['targets'].shape)}  "
          f"feasible={(b['input_lengths'] >= b['target_lengths']).all().item()}")

    if args.save_spec:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        spec = ds[0]["features"].numpy().T  # (F, T)
        plt.figure(figsize=(12, 3))
        plt.imshow(spec, aspect="auto", origin="lower", cmap="magma")
        plt.title(ds[0]["text"][:60])
        plt.xlabel("frame"); plt.ylabel("freq bin")
        plt.tight_layout(); plt.savefig(args.save_spec, dpi=100)
        print(f"saved spectrogram -> {args.save_spec}")


if __name__ == "__main__":
    main()
