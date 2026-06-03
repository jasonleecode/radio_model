"""CTC training loop for the CRNN perception model.

Reusable for real training and for the smoke test (overfit a tiny fixed set
to prove the whole path learns). Greedy decoding + CER is used for monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import AugmentConfig, CWDataset, collate_ctc
from .features import FeatureConfig, feature_dim
from .metrics import cer
from .model import CRNN, count_params
from .vocab import Vocabulary


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 16
    lr: float = 1e-3
    train_size: int = 20000
    seed: int = 0
    log_every: int = 50
    grad_clip: float = 5.0
    device: str = "auto"
    max_seconds: float = 20.0  # clip-length cap (smoke test uses short clips)
    num_workers: int = 2       # DataLoader workers (raise for long GPU runs)
    save_path: str | None = None
    save_every: int = 1000     # periodic checkpoint interval (steps)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def greedy_decode_batch(
    log_probs: torch.Tensor, out_lengths: torch.Tensor, vocab: Vocabulary
) -> list[str]:
    """log_probs (T', N, C) -> list of decoded strings (CTC greedy)."""
    paths = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()  # (N, T')
    texts = []
    for path, L in zip(paths, out_lengths.cpu().tolist()):
        texts.append(vocab.greedy_decode(path[:L]))
    return texts


def evaluate(model, batch, vocab, device) -> tuple[float, list[tuple[str, str]]]:
    """Return mean CER and (ref, hyp) pairs for a batch."""
    model.eval()
    with torch.no_grad():
        feats = batch["features"].to(device)
        log_probs = model(feats)
        out_len = model.output_lengths(batch["input_lengths"], log_probs.shape[0])
        hyps = greedy_decode_batch(log_probs, out_len, vocab)
    refs = batch["texts"]
    pairs = list(zip(refs, hyps))
    mean_cer = float(np.mean([cer(r, h) for r, h in pairs]))
    model.train()
    return mean_cer, pairs


def train(cfg: TrainConfig, aug: AugmentConfig | None = None,
          feat_cfg: FeatureConfig | None = None) -> CRNN:
    device = pick_device(cfg.device)
    feat_cfg = feat_cfg or FeatureConfig()
    vocab = Vocabulary()
    ds = CWDataset(vocab, n_samples=cfg.train_size, seed=cfg.seed,
                   aug=aug, feat_cfg=feat_cfg, max_seconds=cfg.max_seconds)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=collate_ctc, num_workers=cfg.num_workers,
                    drop_last=True, persistent_workers=cfg.num_workers > 0)

    model = CRNN(n_freq=feature_dim(feat_cfg), vocab_size=len(vocab)).to(device)
    print(f"device={device}  params={count_params(model):,}  vocab={len(vocab)}")

    ctc = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    model.train()
    step = 0
    monitor_batch = None
    while step < cfg.steps:
        for batch in dl:
            if monitor_batch is None:
                monitor_batch = batch  # first batch, reused for progress readout
            feats = batch["features"].to(device)
            log_probs = model(feats)  # (T', N, C)
            out_len = model.output_lengths(batch["input_lengths"], log_probs.shape[0])
            loss = ctc(log_probs, batch["targets"].to(device),
                       out_len.to(device), batch["target_lengths"].to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            if step % cfg.log_every == 0:
                mcer, pairs = evaluate(model, monitor_batch, vocab, device)
                print(f"step {step:5d}  loss {loss.item():7.3f}  "
                      f"train_cer {mcer*100:5.1f}%", flush=True)
                if step % (cfg.log_every * 4) == 0 and pairs:
                    r, h = pairs[0]
                    print(f"        ref: {r[:60]!r}")
                    print(f"        hyp: {h[:60]!r}", flush=True)
            if cfg.save_path and step > 0 and step % cfg.save_every == 0:
                _save(model, vocab, feat_cfg, cfg.save_path)
                print(f"        checkpoint -> {cfg.save_path} (step {step})", flush=True)
            step += 1
            if step >= cfg.steps:
                break
    if cfg.save_path:
        _save(model, vocab, feat_cfg, cfg.save_path)
    return model


def _save(model: CRNN, vocab: Vocabulary, feat_cfg: FeatureConfig, path: str) -> None:
    """Save weights + the config needed to rebuild and run the model."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_freq": feature_dim(feat_cfg),
            "vocab_size": len(vocab),
            "feat_cfg": vars(feat_cfg),
        },
        path,
    )
