"""Tests for the CRNN model and the CTC training path (fast, CPU)."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radio_cw.dataset import AugmentConfig, CWDataset, collate_ctc  # noqa: E402
from radio_cw.features import FeatureConfig, feature_dim  # noqa: E402
from radio_cw.model import TIME_DOWNSAMPLE, CRNN, count_params  # noqa: E402
from radio_cw.vocab import Vocabulary  # noqa: E402


def test_forward_shapes():
    v = Vocabulary()
    m = CRNN(n_freq=feature_dim(), vocab_size=len(v))
    x = torch.randn(3, 400, feature_dim())
    y = m(x)  # (T', N, C)
    assert y.shape[1] == 3 and y.shape[2] == len(v)
    assert y.shape[0] == 400 // TIME_DOWNSAMPLE


def test_output_lengths_clamp():
    il = torch.tensor([400, 200, 8])
    out = CRNN.output_lengths(il, t_out=100)
    assert out.tolist() == [100, 50, 2]  # 8//4=2, 400//4=100 clamped to t_out


def test_param_count_small():
    v = Vocabulary()
    m = CRNN(n_freq=feature_dim(), vocab_size=len(v))
    assert count_params(m) < 2_000_000  # "small model"


def test_ctc_loss_decreases():
    """A few optimizer steps on one tiny clip must reduce CTC loss."""
    torch.manual_seed(0)
    v = Vocabulary()
    feat = FeatureConfig()
    # one short, clean, machine-timed clip -> few frames, fast
    ds = CWDataset(v, n_samples=1, seed=0,
                   aug=AugmentConfig(machine_prob=1.0, snr_db=(20, 20)),
                   feat_cfg=feat, max_seconds=3.0)
    batch = collate_ctc([ds[0]])
    m = CRNN(n_freq=feature_dim(feat), vocab_size=len(v))
    ctc = nn.CTCLoss(blank=v.blank_id, zero_infinity=True)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)

    losses = []
    m.train()
    for _ in range(40):
        log_probs = m(batch["features"])
        out_len = m.output_lengths(batch["input_lengths"], log_probs.shape[0])
        loss = ctc(log_probs, batch["targets"], out_len, batch["target_lengths"])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, f"loss did not drop: {losses[0]:.2f}->{losses[-1]:.2f}"
