"""Shared small encoder for V9 exposure auxiliary and factorized hazard tests."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ExposureRiskNetV9(nn.Module):
    def __init__(self, input_width: int, *, variant: str, dropout: float,
                 base_daily_hazard: float, base_log_exposure: float):
        super().__init__()
        if variant not in {"e4a_direct", "e4b_auxiliary", "e4c_factorized"}:
            raise ValueError("unsupported V9 E4 variant")
        self.variant = variant
        self.encoder = nn.Sequential(
            nn.Linear(input_width, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.risk_head = nn.Linear(8, 1)
        self.exposure_head = nn.Linear(8, 1)
        exposure_raw_bias = math.log(math.expm1(max(base_log_exposure, 1e-3)))
        nn.init.zeros_(self.risk_head.weight)
        nn.init.zeros_(self.exposure_head.weight)
        self.exposure_head.bias.data.fill_(exposure_raw_bias)
        if variant == "e4c_factorized":
            base_mu = -math.log1p(-base_daily_hazard)
            base_exposure = math.expm1(base_log_exposure) + .05
            self.risk_head.bias.data.fill_(math.log(base_mu / base_exposure))
        else:
            self.risk_head.bias.data.fill_(math.log(base_daily_hazard / (1 - base_daily_hazard)))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(inputs)
        risk = self.risk_head(hidden).squeeze(-1)
        predicted_log_exposure = F.softplus(self.exposure_head(hidden).squeeze(-1))
        if self.variant == "e4c_factorized":
            exposure = torch.expm1(predicted_log_exposure).clamp_min(0) + .05
            intensity = torch.exp(risk.clamp(-14, 3)) * exposure
            daily_hazard = -torch.expm1(-intensity.clamp_max(20))
        else:
            daily_hazard = torch.sigmoid(risk)
        return daily_hazard.clamp(1e-7, 1 - 1e-7), predicted_log_exposure

