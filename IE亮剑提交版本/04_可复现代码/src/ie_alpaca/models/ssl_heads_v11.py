"""Pretraining heads for V11 masked, contrastive and JEPA experiments."""

from __future__ import annotations

import copy
import torch
from torch import nn

from ie_alpaca.features.daily_v10 import CONTEXT_COLUMNS, EVENT_FEATURE_NAMES
from ie_alpaca.features.landmark_v4 import EVENT_CODES
from ie_alpaca.models.representation_v11 import RepresentationEncoderV11


class MaskedHeadV11(nn.Module):
    def __init__(self, latent: int):
        super().__init__()
        self.event_id = nn.Embedding(len(EVENT_CODES), 8)
        self.event = nn.Sequential(nn.Linear(latent + 8, 64), nn.GELU(), nn.Linear(64, len(EVENT_FEATURE_NAMES)))
        self.context = nn.Sequential(nn.Linear(latent, 64), nn.GELU(), nn.Linear(64, len(CONTEXT_COLUMNS)))

    def forward(self, z: torch.Tensor):
        ids = self.event_id.weight[None, None].expand(*z.shape[:2], -1, -1)
        repeated = z[:, :, None].expand(-1, -1, len(EVENT_CODES), -1)
        return self.event(torch.cat((repeated, ids), dim=-1)), self.context(z)


class ContrastiveHeadV11(nn.Module):
    def __init__(self, latent: int):
        super().__init__()
        self.stable = nn.Sequential(nn.Linear(latent, latent), nn.GELU(), nn.Linear(latent, 32))
        self.acute = nn.Sequential(nn.Linear(latent, latent), nn.GELU(), nn.Linear(latent, 32))

    def forward(self, z: torch.Tensor):
        return self.stable(z.mean(dim=1)), self.acute(z[:, -14:].mean(dim=1))


class JEPAV11(nn.Module):
    def __init__(self, encoder: RepresentationEncoderV11, width: int):
        super().__init__()
        self.online = encoder
        self.target = copy.deepcopy(encoder)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.horizon = nn.Embedding(3, 8)
        self.predictor = nn.Sequential(nn.Linear(2 * width + 8, 128), nn.GELU(), nn.Linear(128, 12 * width))

    @torch.no_grad()
    def update_target(self, momentum: float):
        for target, online in zip(self.target.parameters(), self.online.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1 - momentum)

