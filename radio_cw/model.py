"""CRNN + CTC perception model: narrow-band spectrogram -> character logits.

Architecture (small by design, per docs/perception_design.md):

    (N, T, F)  input features
      -> reshape (N, 1, F, T)            freq as height, time as width
      -> 2x [Conv2d 3x3 + BN + ReLU + MaxPool(2,2)]   freq & time /4
      -> collapse (channels x freq) -> per-time feature vector
      -> 2-layer BiGRU
      -> Linear -> (T', N, vocab)        log-probs for CTC

Time is downsampled 4x (T' = T//4); even so T' >> label length, so CTC stays
feasible. Frequency is pooled away entirely (16 -> 4 bins kept as features).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TIME_DOWNSAMPLE = 4  # product of the two MaxPool stride-2 ops in time


class CRNN(nn.Module):
    def __init__(
        self,
        n_freq: int,
        vocab_size: int,
        *,
        conv_channels: tuple[int, int] = (16, 32),
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        c1, c2 = conv_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # freq/2, time/2
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # freq/2, time/2
        )
        freq_out = n_freq // 4  # two freq pools
        self.rnn_input = c2 * freq_out
        self.rnn = nn.GRU(
            self.rnn_input,
            gru_hidden,
            num_layers=gru_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(2 * gru_hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, T, F) -> log_probs (T', N, vocab) for nn.CTCLoss."""
        n = x.shape[0]
        x = x.transpose(1, 2).unsqueeze(1)   # (N, 1, F, T)
        x = self.cnn(x)                      # (N, C, F', T')
        c, fbins, t = x.shape[1], x.shape[2], x.shape[3]
        x = x.permute(0, 3, 1, 2).reshape(n, t, c * fbins)  # (N, T', C*F')
        x, _ = self.rnn(x)                   # (N, T', 2H)
        x = self.fc(x)                       # (N, T', vocab)
        return F.log_softmax(x, dim=-1).transpose(0, 1)  # (T', N, vocab)

    @staticmethod
    def output_lengths(input_lengths: torch.Tensor, t_out: int) -> torch.Tensor:
        """Map padded input frame counts to post-CNN frame counts.

        Two stride-2 pools floor-divide the time axis by 4. Clamp to the actual
        output length so CTC never sees a length past the tensor.
        """
        out = torch.div(input_lengths, TIME_DOWNSAMPLE, rounding_mode="floor")
        return out.clamp(min=1, max=t_out)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
