"""Small multimodal 60-day encoder with ordered multi-horizon probabilities."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TemporalRiskNet(nn.Module):
    def __init__(self, groups: dict[str, list[int]], width: int, hidden: int = 24, dropout: float = 0.15):
        super().__init__()
        self.groups = groups
        self.width = width
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(2 * len(indices), 16), nn.LayerNorm(16), nn.GELU())
            for name, indices in groups.items()
        })
        self.fusion = nn.Sequential(nn.Linear(64, 48), nn.LayerNorm(48), nn.GELU(), nn.Dropout(dropout))
        self.temporal = nn.GRU(48, hidden, batch_first=True, bidirectional=True)
        self.risk = nn.Sequential(nn.Linear(3 * hidden * 2, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 5))
        self.reconstruct = nn.Linear(2 * hidden, width)

    def forward(self, values: torch.Tensor, observed: torch.Tensor, lengths: torch.Tensor):
        if values.shape != observed.shape or values.shape[-1] != self.width:
            raise ValueError("数值和缺测掩码尺寸不匹配")
        if bool((lengths < 1).any()) or bool((lengths > values.shape[1]).any()):
            raise ValueError("历史长度超出张量范围")
        modality = [self.encoders[name](torch.cat((values[:, :, ids], observed[:, :, ids]), dim=-1))
                    for name, ids in self.groups.items()]
        day = self.fusion(torch.cat(modality, dim=-1))
        packed = pack_padded_sequence(day, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.temporal(packed)
        states, _ = pad_packed_sequence(encoded, batch_first=True, total_length=values.shape[1])
        valid = torch.arange(values.shape[1], device=values.device)[None, :] < lengths[:, None]
        mean = (states * valid.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        negative = torch.finfo(states.dtype).min
        maximum = states.masked_fill(~valid.unsqueeze(-1), negative).max(dim=1).values
        last = states[torch.arange(len(states), device=states.device), lengths - 1]
        segment_logits = self.risk(torch.cat((last, mean, maximum), dim=-1))
        # Five disjoint bins: days 1..7, 8..14, 15..21, 22..30, 31..40.
        # Nonnegative hazards yield P(7) <= P(14) <= ... <= P(40).
        segment_survival = 1 - torch.sigmoid(segment_logits)
        probability = 1 - torch.cumprod(segment_survival, dim=-1)
        return probability.clamp(1e-6, 1 - 1e-6), self.reconstruct(states)
