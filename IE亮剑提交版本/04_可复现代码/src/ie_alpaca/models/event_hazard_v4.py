"""Tiny additive PyTorch hazard with shared event response and type weights."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ie_alpaca.features.landmark_v4 import EVENT_CODES, EVENT_METRICS, GROUPS, QUALITY_CODES, RISK_CODES


RISK_POSITIONS = [EVENT_CODES.index(code) for code in RISK_CODES]
QUALITY_POSITIONS = [EVENT_CODES.index(code) for code in QUALITY_CODES]
GROUP_INDEX = [next(i for i, (_, codes) in enumerate(GROUPS.items()) if code in codes)
               for code in RISK_CODES]


class EventHazardNet(nn.Module):
    def __init__(self, context_width: int, *, learned_weights: bool, dropout: float = 0.1,
                 base_daily_hazard: float = 0.01):
        super().__init__()
        self.learned_weights = learned_weights
        self.event_encoder = nn.Sequential(nn.Linear(len(EVENT_METRICS), 8), nn.Tanh(), nn.Linear(8, 1))
        self.context_encoder = nn.Sequential(
            nn.Linear(context_width, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1),
        )
        self.group_log_weight = nn.Parameter(torch.zeros(5))
        self.type_delta = nn.Parameter(torch.zeros(len(RISK_CODES)))
        self.quality_weight = nn.Parameter(torch.zeros(len(QUALITY_CODES)))
        self.bias = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(base_daily_hazard)))))
        self.register_buffer("group_index", torch.tensor(GROUP_INDEX, dtype=torch.long))

    def risk_weights(self) -> torch.Tensor:
        if not self.learned_weights:
            return torch.ones_like(self.type_delta)
        log_w = (self.group_log_weight[self.group_index] + self.type_delta).clamp(-4, 4)
        weights = log_w.exp()
        return weights / weights.mean()

    def forward(self, events: torch.Tensor, present: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if events.shape[-2:] != (len(EVENT_CODES), len(EVENT_METRICS)) or present.shape != events.shape[:-1]:
            raise ValueError("事件张量的最后两维必须为 [事件类型数, 统计项数]")
        if context.shape[:-1] != events.shape[:-2]:
            raise ValueError("上下文与事件的样本维度不一致")
        dose = F.softplus(self.event_encoder(events).squeeze(-1)) * present
        risk = (dose[..., RISK_POSITIONS] * self.risk_weights()).sum(dim=-1) / len(EVENT_CODES)
        quality = (dose[..., QUALITY_POSITIONS] * self.quality_weight).sum(dim=-1) / len(EVENT_CODES)
        return self.bias + risk + quality + self.context_encoder(context).squeeze(-1)

    def type_penalty(self, rarity: torch.Tensor, strength: float) -> torch.Tensor:
        if not self.learned_weights:
            return self.quality_weight.square().mean() * strength
        return strength * (
            self.group_log_weight.square().mean()
            + (rarity * self.type_delta.square()).mean()
            + self.quality_weight.square().mean()
        )

    @torch.no_grad()
    def weight_report(self) -> dict[int, float]:
        risk = self.risk_weights().cpu().numpy()
        quality = self.quality_weight.cpu().numpy()
        return {**dict(zip(RISK_CODES, map(float, risk))),
                **dict(zip(QUALITY_CODES, map(float, quality)))}
