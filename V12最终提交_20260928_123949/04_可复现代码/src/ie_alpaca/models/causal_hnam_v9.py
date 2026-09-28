"""Hierarchical additive hazard models for the V9 human-prior ablation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ie_alpaca.features.causal_v9 import EVENT_ROLES, RISK_CHAINS
from ie_alpaca.features.landmark_v4 import EVENT_CODES, EVENT_METRICS, QUALITY_CODES, RISK_CODES


RISK_POSITIONS = [EVENT_CODES.index(code) for code in RISK_CODES]
QUALITY_POSITIONS = [EVENT_CODES.index(code) for code in QUALITY_CODES]
ROLE_NAMES = tuple(name for name in EVENT_ROLES if name != "quality")
ROLE_INDEX = torch.tensor([
    next(i for i, name in enumerate(ROLE_NAMES) if code in EVENT_ROLES[name])
    for code in RISK_CODES
], dtype=torch.long)
RISK_LOCAL = {code: i for i, code in enumerate(RISK_CODES)}
CHAIN_LOCAL = [
    ([RISK_LOCAL[code] for code in source if code in RISK_LOCAL],
     [RISK_LOCAL[code] for code in target if code in RISK_LOCAL])
    for source, target in RISK_CHAINS.values()
]
CHRONIC_METRICS = (0, 1, 5, 6)
ACUTE_METRICS = (2, 3, 4, 7, 8)


class CausalHNAMV9(nn.Module):
    """Small event-additive model with explicit hierarchy and fixed interactions."""

    def __init__(self, context_width: int, *, variant: str, learned_weights: bool,
                 dropout: float = .1, base_daily_hazard: float = .01):
        super().__init__()
        if variant not in {"e1a_flat", "e1b_hierarchical", "e1c_chronic_acute", "e1d_interactions"}:
            raise ValueError("unsupported V9 E1 variant")
        self.variant = variant
        self.learned_weights = learned_weights
        split = variant in {"e1c_chronic_acute", "e1d_interactions"}
        if split:
            self.chronic_encoder = nn.Sequential(
                nn.Linear(len(CHRONIC_METRICS), 8), nn.Tanh(), nn.Linear(8, 1))
            self.acute_encoder = nn.Sequential(
                nn.Linear(len(ACUTE_METRICS), 8), nn.Tanh(), nn.Linear(8, 1))
            heads = 2
        else:
            self.event_encoder = nn.Sequential(
                nn.Linear(len(EVENT_METRICS), 8), nn.Tanh(), nn.Linear(8, 1))
            heads = 1
        self.type_log_weight = nn.Parameter(torch.zeros(heads, len(RISK_CODES)))
        self.group_log_weight = nn.Parameter(torch.zeros(heads, len(ROLE_NAMES)))
        self.quality_weight = nn.Parameter(torch.zeros(len(QUALITY_CODES)))
        self.interaction_weight = nn.Parameter(torch.zeros(len(CHAIN_LOCAL)))
        self.context_encoder = nn.Sequential(
            nn.Linear(context_width, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1))
        self.bias = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(base_daily_hazard)))))
        self.register_buffer("role_index", ROLE_INDEX.clone())

    def _weights(self) -> torch.Tensor:
        if not self.learned_weights:
            return torch.ones_like(self.type_log_weight)
        if self.variant == "e1a_flat":
            log_weight = self.type_log_weight
        else:
            log_weight = self.group_log_weight[:, self.role_index] + self.type_log_weight
        weights = log_weight.clamp(-4, 4).exp()
        return weights / weights.mean(dim=-1, keepdim=True)

    def _risk_dose(self, events: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        risk_present = present[..., RISK_POSITIONS]
        if self.variant in {"e1c_chronic_acute", "e1d_interactions"}:
            chronic = F.softplus(self.chronic_encoder(events[..., CHRONIC_METRICS]).squeeze(-1))
            acute = F.softplus(self.acute_encoder(events[..., ACUTE_METRICS]).squeeze(-1))
            chronic = chronic[..., RISK_POSITIONS] * risk_present
            acute = acute[..., RISK_POSITIONS] * risk_present
            weights = self._weights()
            return chronic * weights[0] + acute * weights[1], (chronic + acute) / 2
        dose = F.softplus(self.event_encoder(events).squeeze(-1))
        risk = dose[..., RISK_POSITIONS] * risk_present * self._weights()[0]
        return risk, dose[..., RISK_POSITIONS] * risk_present

    def forward(self, events: torch.Tensor, present: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        if events.shape[-2:] != (len(EVENT_CODES), len(EVENT_METRICS)):
            raise ValueError("event input must end with [24, 9]")
        risk, unweighted = self._risk_dose(events, present)
        event_score = risk.sum(dim=-1) / len(EVENT_CODES)
        quality_base = (events[..., QUALITY_POSITIONS, 0] + events[..., QUALITY_POSITIONS, 5]) / 2
        quality = (quality_base * present[..., QUALITY_POSITIONS] * self.quality_weight).sum(dim=-1)
        quality = quality / len(EVENT_CODES)
        interaction = torch.zeros_like(event_score)
        if self.variant == "e1d_interactions":
            terms = []
            for source, target in CHAIN_LOCAL:
                source_dose = unweighted[..., source].mean(dim=-1)
                target_dose = unweighted[..., target].mean(dim=-1)
                terms.append(source_dose * target_dose)
            interaction = (torch.stack(terms, dim=-1) * F.softplus(self.interaction_weight)).sum(dim=-1)
            interaction = interaction / len(CHAIN_LOCAL)
        return self.bias + event_score + quality + interaction + self.context_encoder(context).squeeze(-1)

    def type_penalty(self, rarity: torch.Tensor, strength: float) -> torch.Tensor:
        type_term = (rarity.unsqueeze(0) * self.type_log_weight.square()).mean()
        if self.variant == "e1a_flat":
            group_term = torch.zeros((), device=type_term.device)
        else:
            group_term = self.group_log_weight.square().mean()
        interaction_term = self.interaction_weight.square().mean()
        return strength * (type_term + group_term + self.quality_weight.square().mean()
                           + .25 * interaction_term)

    @torch.no_grad()
    def weight_report(self) -> dict[int, float]:
        risk = self._weights().mean(dim=0).cpu().numpy()
        quality = self.quality_weight.cpu().numpy()
        return {**dict(zip(RISK_CODES, map(float, risk))),
                **dict(zip(QUALITY_CODES, map(float, quality)))}

