"""Deeper non-temporal hazard models built on the V5 landmark inputs."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ie_alpaca.features.landmark_v4 import (
    EVENT_CODES, EVENT_METRICS, GROUPS, QUALITY_CODES, RISK_CODES,
)


RISK_POSITIONS = [EVENT_CODES.index(code) for code in RISK_CODES]
QUALITY_POSITIONS = [EVENT_CODES.index(code) for code in QUALITY_CODES]
RISK_GROUP_NAMES = tuple(name for name in GROUPS if name != "device_quality")
RISK_GROUP_INDEX = torch.tensor([
    next(i for i, name in enumerate(RISK_GROUP_NAMES) if code in GROUPS[name])
    for code in RISK_CODES
], dtype=torch.long)


class EventHazardNetV8(nn.Module):
    """Two hidden layers, with either additive or joint event-context fusion."""

    def __init__(self, context_width: int, *, architecture: str,
                 learned_weights: bool, dropout: float = .1,
                 base_daily_hazard: float = .01):
        super().__init__()
        if architecture not in {"deep_additive", "deep_fusion"}:
            raise ValueError("architecture must be deep_additive or deep_fusion")
        self.architecture = architecture
        self.learned_weights = learned_weights
        self.event_encoder = nn.Sequential(
            nn.Linear(len(EVENT_METRICS), 16), nn.GELU(),
            nn.Linear(16, 8), nn.GELU(),
        )
        self.event_score = nn.Linear(8, 1)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_width, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.GELU(), nn.Dropout(dropout),
        )
        if architecture == "deep_additive":
            self.output_head = nn.Linear(16, 1)
        else:
            # 5 risk-family summaries + 2 quality signals + 16 context dimensions.
            self.output_head = nn.Sequential(
                nn.Linear(len(RISK_GROUP_NAMES) + len(QUALITY_CODES) + 16, 16),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(16, 1, bias=False),
            )
        self.group_log_weight = nn.Parameter(torch.zeros(len(RISK_GROUP_NAMES)))
        self.type_delta = nn.Parameter(torch.zeros(len(RISK_CODES)))
        self.quality_weight = nn.Parameter(torch.zeros(len(QUALITY_CODES)))
        self.bias = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(base_daily_hazard)))))
        self.register_buffer("risk_group_index", RISK_GROUP_INDEX.clone())

    def risk_weights(self) -> torch.Tensor:
        if not self.learned_weights:
            return torch.ones_like(self.type_delta)
        log_weight = (self.group_log_weight[self.risk_group_index] + self.type_delta).clamp(-4, 4)
        weight = log_weight.exp()
        return weight / weight.mean()

    def _parts(self, events: torch.Tensor, present: torch.Tensor,
               context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if events.shape[-2:] != (len(EVENT_CODES), len(EVENT_METRICS)):
            raise ValueError("event input must end with [24 event types, 9 metrics]")
        if present.shape != events.shape[:-1] or context.shape[:-1] != events.shape[:-2]:
            raise ValueError("event, presence, and context leading dimensions must agree")
        embedding = self.event_encoder(events)
        dose = F.softplus(self.event_score(embedding).squeeze(-1)) * present
        context_embedding = self.context_encoder(context)
        return dose, context_embedding, self.risk_weights()

    def forward(self, events: torch.Tensor, present: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        dose, context_embedding, weights = self._parts(events, present, context)
        risk_dose = dose[..., RISK_POSITIONS] * weights
        quality_dose = dose[..., QUALITY_POSITIONS] * self.quality_weight
        if self.architecture == "deep_additive":
            risk = risk_dose.sum(dim=-1) / len(EVENT_CODES)
            quality = quality_dose.sum(dim=-1) / len(EVENT_CODES)
            return self.bias + risk + quality + self.output_head(context_embedding).squeeze(-1)
        groups = torch.stack([
            risk_dose[..., self.risk_group_index.eq(index)].sum(dim=-1) / len(EVENT_CODES)
            for index in range(len(RISK_GROUP_NAMES))
        ], dim=-1)
        quality = quality_dose / len(EVENT_CODES)
        fused = torch.cat((groups, quality, context_embedding), dim=-1)
        return self.bias + self.output_head(fused).squeeze(-1)

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
