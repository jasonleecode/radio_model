"""Unified perception layer: audio -> (text, metadata).

Two interchangeable backends behind one call signature, so the closed loop
(or any caller) can swap the decoder without changing anything downstream:

  - DSPPerception:   text from the baseline DSP decoder
  - ModelPerception: text from the trained CRNN+CTC model

Both follow the hybrid design (docs/perception_design.md): the physical
metadata (sidetone freq / SNR / WPM) always comes from the DSP front-end's
analyze(); only the *text* source differs.
"""

from __future__ import annotations

import numpy as np
import torch

from .decode_dsp import decode_audio
from .dsp_frontend import Metadata, analyze
from .features import FeatureConfig, spectrogram
from .model import CRNN
from .train import greedy_decode_batch
from .vocab import Vocabulary


def load_model(
    path: str, device: torch.device | str = "cpu"
) -> tuple[CRNN, FeatureConfig, Vocabulary]:
    """Load a checkpoint saved by train._save into a ready-to-run model."""
    device = torch.device(device) if isinstance(device, str) else device
    ckpt = torch.load(path, map_location=device)
    feat_cfg = FeatureConfig(**ckpt["feat_cfg"])
    model = CRNN(n_freq=ckpt["n_freq"], vocab_size=ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, feat_cfg, Vocabulary()


@torch.no_grad()
def model_decode_audio(
    model: CRNN, feat_cfg: FeatureConfig, vocab: Vocabulary,
    audio: np.ndarray, device: torch.device,
) -> str:
    """Greedy-decode one audio clip to text with the CRNN+CTC model."""
    feats = torch.from_numpy(spectrogram(audio, feat_cfg).T).unsqueeze(0).to(device)
    log_probs = model(feats)  # (T', 1, C)
    out_len = model.output_lengths(torch.tensor([feats.shape[1]]), log_probs.shape[0])
    return greedy_decode_batch(log_probs, out_len, vocab)[0]


class DSPPerception:
    """Baseline backend: DSP decoder for text, DSP front-end for metadata."""

    def __call__(self, audio: np.ndarray, sample_rate: int) -> tuple[str, Metadata]:
        return decode_audio(audio, sample_rate)


class ModelPerception:
    """Model backend: CRNN text + DSP metadata sidecar."""

    def __init__(self, ckpt_path: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.feat_cfg, self.vocab = load_model(ckpt_path, self.device)

    def __call__(self, audio: np.ndarray, sample_rate: int) -> tuple[str, Metadata]:
        _, meta = analyze(audio, sample_rate)  # metadata sidecar
        text = model_decode_audio(self.model, self.feat_cfg, self.vocab,
                                  audio, self.device)
        return text, meta
