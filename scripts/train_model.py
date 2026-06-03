"""Train the CRNN perception model (or run a smoke test).

Smoke test (overfit a tiny fixed set; expect loss -> ~0, train CER -> ~0):
    python scripts/train_model.py --smoke

Real run:
    python scripts/train_model.py --steps 5000 --batch-size 32
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radio_cw.dataset import AugmentConfig  # noqa: E402
from radio_cw.train import TrainConfig, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="overfit a tiny clean set to validate the path")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-size", type=int, default=20000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save", default=None, help="path to save the trained model")
    args = ap.parse_args()

    if args.smoke:
        # Short clips (~4 s) keep the blank/label ratio sane so the model
        # escapes CTC blank-collapse fast, and run quickly on CPU.
        cfg = TrainConfig(steps=500, batch_size=8, lr=3e-3, train_size=16,
                          log_every=25, device=args.device, max_seconds=4.0)
        # Easy regime: clean machine timing, high SNR -> must overfit fast.
        aug = AugmentConfig(machine_prob=1.0, snr_db=(20.0, 25.0))
        print("=== SMOKE TEST: overfitting 16 short clean samples ===")
    else:
        cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                          train_size=args.train_size, device=args.device)
        aug = None

    model = train(cfg, aug=aug)

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()
