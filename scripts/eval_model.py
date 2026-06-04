"""Head-to-head: trained CRNN model vs DSP baseline, same data, same buckets.

Mirrors scripts/eval_baseline.py so the numbers are directly comparable. Both
decoders see the *identical* audio clips; we report CER for each side by side.
The decisive cell is the human-fist table -- the model should beat the baseline
there even though the baseline is near-perfect on clean machine timing.

    torch_env/bin/python scripts/eval_model.py runs/crnn.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radio_cw.corpus import CorpusSampler  # noqa: E402
from radio_cw.dataset import render_audio  # noqa: E402
from radio_cw.decode_dsp import decode_audio  # noqa: E402
from radio_cw.metrics import cer  # noqa: E402
from radio_cw.perception import load_model, model_decode_audio  # noqa: E402
from radio_cw.vocab import Vocabulary  # noqa: E402


def _norm(text: str, vocab: Vocabulary) -> str:
    """Canonicalize to the vocab's token convention for a fair comparison.

    The DSP decoder renders code -...- as the prosign <BT>, while the corpus /
    model use '='; both are the same code. Normalizing both sides removes that
    representation mismatch so we measure decoding accuracy, not notation.
    """
    return vocab.decode(vocab.encode(text))


def eval_bucket(model, feat_cfg, vocab, device, texts, params_fn, rng):
    """Return (dsp_cer, model_cer) averaged over texts for one pinned param set."""
    dsp_errs, mdl_errs = [], []
    for text in texts:
        params = params_fn(rng)
        audio = render_audio(text, params, rng, feat_cfg.sample_rate)
        dsp_text, _ = decode_audio(audio, feat_cfg.sample_rate)
        mdl_text = model_decode_audio(model, feat_cfg, vocab, audio, device)
        ref = _norm(text, vocab)
        dsp_errs.append(cer(ref, _norm(dsp_text, vocab)))
        mdl_errs.append(cer(ref, mdl_text))  # model output already canonical
    return float(np.mean(dsp_errs)), float(np.mean(mdl_errs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="trained model checkpoint (runs/crnn.pt)")
    ap.add_argument("--n", type=int, default=40, help="texts per bucket")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    model, feat_cfg, vocab = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt}  device={device}\n")

    corpus = CorpusSampler(seed=123)
    texts = corpus.samples(args.n)

    def fmt(d, m):
        return f"{d*100:5.1f}/{m*100:5.1f}"

    rng = np.random.default_rng(0)
    print("=== Machine timing vs SNR   (cells: DSP/MODEL CER %) ===")
    snrs = [20, 10, 5, 0, -3]
    print("  WPM | " + " ".join(f"{s:>11.0f}dB" for s in snrs))
    for wpm in [15, 20, 30]:
        cells = []
        for snr in snrs:
            d, m = eval_bucket(model, feat_cfg, vocab, device, texts,
                               lambda r, w=wpm, s=snr: {
                                   "wpm": w, "sidetone_hz": float(r.uniform(500, 800)),
                                   "snr_db": s, "timing_jitter": 0.0, "weight_bias": 1.0,
                                   "qsb": False}, rng)
            cells.append(fmt(d, m))
        print(f"  {wpm:>3.0f} | " + " ".join(f"{c:>13}" for c in cells))

    print("\n=== Human fist vs jitter @ 10 dB   (cells: DSP/MODEL CER %) ===")
    jitters = [0.0, 0.10, 0.20, 0.30]
    print("  WPM | " + " ".join(f"{j:>11.2f}j" for j in jitters))
    for wpm in [15, 20, 30]:
        cells = []
        for j in jitters:
            d, m = eval_bucket(model, feat_cfg, vocab, device, texts,
                               lambda r, w=wpm, jj=j: {
                                   "wpm": w, "sidetone_hz": float(r.uniform(500, 800)),
                                   "snr_db": 10.0, "timing_jitter": jj,
                                   "weight_bias": 1.0 + 0.5 * jj, "qsb": False}, rng)
            cells.append(fmt(d, m))
        print(f"  {wpm:>3.0f} | " + " ".join(f"{c:>13}" for c in cells))

    print("\nEach cell is DSP/MODEL character error rate (%). Lower is better.")
    print("The fist table is the decisive comparison.")


if __name__ == "__main__":
    main()
