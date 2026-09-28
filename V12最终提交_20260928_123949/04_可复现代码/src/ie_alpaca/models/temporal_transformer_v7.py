"""One-layer day-sequence hazard and a parameter-matched no-attention control."""

from __future__ import annotations

import math

import torch
from torch import nn


class TemporalHazardV7(nn.Module):
    def __init__(self, input_width: int, *, mode: str, d_model: int = 24,
                 heads: int = 2, ff_dim: int = 48, dropout: float = .2):
        super().__init__()
        if mode not in {"transformer", "no_attention"}:
            raise ValueError("mode must be transformer or no_attention")
        if d_model % 2 or d_model % heads:
            raise ValueError("d_model must be even and divisible by heads")
        self.mode = mode
        self.d_model = d_model
        self.input_projection = nn.Linear(input_width, d_model)
        if mode == "transformer":
            self.encoder = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=heads, dim_feedforward=ff_dim,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
        else:
            # Same day tokens and pooling, comparable parameter count, no cross-day interaction.
            self.encoder = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, 96), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(96, d_model),
            )
        self.output_norm = nn.LayerNorm(d_model)
        self.hazard_head = nn.Linear(2 * d_model, 1)

    def _positions(self, valid: torch.Tensor) -> torch.Tensor:
        batch, days = valid.shape
        lengths = valid.sum(dim=1)
        recency = (lengths[:, None] - 1 - torch.arange(days, device=valid.device)[None, :]).clamp_min(0)
        scales = torch.exp(-math.log(10000) * torch.arange(0, self.d_model, 2,
                                                         device=valid.device, dtype=torch.float32) / self.d_model)
        angles = recency.float()[..., None] * scales
        positions = torch.zeros(batch, days, self.d_model, device=valid.device, dtype=torch.float32)
        positions[..., 0::2] = torch.sin(angles)
        positions[..., 1::2] = torch.cos(angles)
        return positions * valid[..., None]

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or valid.shape != x.shape[:2] or x.shape[-1] != self.input_projection.in_features:
            raise ValueError("V7 input must be [samples, days, features] with a day mask")
        valid = valid.bool()
        lengths = valid.sum(dim=1)
        expected = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        if torch.any(lengths == 0) or not torch.equal(valid, expected):
            raise ValueError("each sample must contain a nonempty day prefix and right padding")
        h = self.input_projection(x) + self._positions(valid).to(x.dtype)
        if self.mode == "transformer":
            h = self.encoder(h, src_key_padding_mask=~valid)
        else:
            h = h + self.encoder(h)
        h = self.output_norm(h)
        mean = (h * valid[..., None]).sum(dim=1) / lengths[:, None]
        last = h[torch.arange(len(h), device=h.device), lengths - 1]
        return self.hazard_head(torch.cat((mean, last), dim=-1)).squeeze(-1)
