"""On-the-fly synthetic dataset for the perception model.

Combines the pieces built earlier into an infinite labelled stream:

    corpus text --synth(jitter,weight)--> audio --channel(noise,qsb)-->
        --features(narrowband spectrogram)--> (T, F) input
    text --vocab.encode--> CTC target ids

The augmentation ranges (the 6 dimensions from docs/perception_design.md) are
the source of robustness. ``render_features`` is also reusable with *pinned*
params so the eval harness can build SNR / jitter buckets that mirror the
baseline benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from .channel import add_qsb, add_white_noise
from .corpus import CorpusSampler
from .features import FeatureConfig, spectrogram
from .synth import SynthConfig, estimate_duration, synthesize
from .vocab import Vocabulary


def fit_text_to_duration(text: str, wpm: float, max_seconds: float) -> str:
    """Drop trailing words until the clip fits ``max_seconds`` at ``wpm``.

    Keeps audio and labels aligned (we synth exactly the truncated text).
    """
    words = text.split(" ")
    cfg = SynthConfig(wpm=wpm)
    while len(words) > 1 and estimate_duration(" ".join(words), cfg) > max_seconds:
        words.pop()
    return " ".join(words)


@dataclass
class AugmentConfig:
    """Random ranges for the augmentation dimensions."""

    wpm: tuple[float, float] = (12.0, 40.0)
    sidetone_hz: tuple[float, float] = (400.0, 900.0)
    snr_db: tuple[float, float] = (-3.0, 25.0)
    timing_jitter: tuple[float, float] = (0.0, 0.30)
    weight_bias: tuple[float, float] = (0.85, 1.25)
    machine_prob: float = 0.25  # fraction sent with perfect (machine) timing
    qsb_prob: float = 0.3

    def sample(self, rng: np.random.Generator) -> dict:
        machine = rng.random() < self.machine_prob
        return {
            "wpm": float(rng.uniform(*self.wpm)),
            "sidetone_hz": float(rng.uniform(*self.sidetone_hz)),
            "snr_db": float(rng.uniform(*self.snr_db)),
            "timing_jitter": 0.0 if machine else float(rng.uniform(*self.timing_jitter)),
            "weight_bias": 1.0 if machine else float(rng.uniform(*self.weight_bias)),
            "qsb": (not machine) and (rng.random() < self.qsb_prob),
        }


def render_features(
    text: str,
    params: dict,
    rng: np.random.Generator,
    feat_cfg: FeatureConfig | None = None,
) -> np.ndarray:
    """Render text+params to a (T, F) feature array. Reusable with pinned params."""
    feat_cfg = feat_cfg or FeatureConfig()
    scfg = SynthConfig(
        wpm=params["wpm"],
        sidetone_hz=params["sidetone_hz"],
        sample_rate=feat_cfg.sample_rate,
        timing_jitter=params.get("timing_jitter", 0.0),
        weight_bias=params.get("weight_bias", 1.0),
    )
    audio, _ = synthesize(text, scfg, rng)
    if params.get("qsb"):
        audio = add_qsb(audio, feat_cfg.sample_rate, rng)
    if params.get("snr_db") is not None:
        audio = add_white_noise(audio, params["snr_db"], rng)
    spec = spectrogram(audio, feat_cfg)  # (F, T)
    return spec.T  # (T, F) for the RNN


class CWDataset(Dataset):
    """Map-style dataset; each index deterministically synthesizes one sample.

    Deterministic-per-index means eval sets are reproducible. For training,
    vary ``seed`` per epoch (or just rely on the large virtual length).
    """

    def __init__(
        self,
        vocab: Vocabulary,
        n_samples: int = 10000,
        *,
        seed: int = 0,
        aug: AugmentConfig | None = None,
        feat_cfg: FeatureConfig | None = None,
        max_seconds: float = 20.0,
    ):
        self.vocab = vocab
        self.n_samples = n_samples
        self.seed = seed
        self.aug = aug or AugmentConfig()
        self.feat_cfg = feat_cfg or FeatureConfig()
        self.max_seconds = max_seconds

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng([self.seed, idx])
        text = CorpusSampler(seed=int(rng.integers(0, 2**31))).sample()
        params = self.aug.sample(rng)
        text = fit_text_to_duration(text, params["wpm"], self.max_seconds)
        feats = render_features(text, params, rng, self.feat_cfg)
        targets = self.vocab.encode(text)
        return {
            "features": torch.from_numpy(feats),          # (T, F) float32
            "targets": torch.tensor(targets, dtype=torch.long),
            "text": text,
            "params": params,
        }


def collate_ctc(batch: list[dict]) -> dict:
    """Pad features to (N, Tmax, F); concat targets for nn.CTCLoss."""
    feats = [b["features"] for b in batch]
    input_lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    f_dim = feats[0].shape[1]
    t_max = int(input_lengths.max())
    padded = torch.zeros(len(batch), t_max, f_dim, dtype=torch.float32)
    for i, f in enumerate(feats):
        padded[i, : f.shape[0]] = f

    target_lengths = torch.tensor([b["targets"].numel() for b in batch], dtype=torch.long)
    targets = torch.cat([b["targets"] for b in batch]) if batch else torch.empty(0, dtype=torch.long)
    return {
        "features": padded,                 # (N, Tmax, F)
        "input_lengths": input_lengths,     # (N,)
        "targets": targets,                 # (sum L,)
        "target_lengths": target_lengths,   # (N,)
        "texts": [b["text"] for b in batch],
    }
