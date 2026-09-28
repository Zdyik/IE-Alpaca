"""Masked reconstruction and EMA targets for V12 risk-chain SSL."""

from __future__ import annotations

import copy

import torch
from torch import nn

from ie_alpaca.features.daily_v10 import CONTEXT_COLUMNS, EVENT_FEATURE_NAMES
from ie_alpaca.features.landmark_v4 import EVENT_CODES
from ie_alpaca.models.state_encoder_v12 import StateEncoderV12, V12Backbone


class StateDecoderV12(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.event_id = nn.Embedding(len(EVENT_CODES), 8)
        self.event = nn.Sequential(nn.Linear(width + 8, 64), nn.GELU(),
                                   nn.Linear(64, len(EVENT_FEATURE_NAMES)))
        self.context = nn.Sequential(nn.Linear(6 * width, 96), nn.GELU(),
                                     nn.Linear(96, len(CONTEXT_COLUMNS)))

    def forward(self, states: torch.Tensor, assignment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        event_latent = torch.einsum("btsw,es->btew", states, assignment)
        identity = self.event_id.weight[None, None].expand(*event_latent.shape[:2], -1, -1)
        event = self.event(torch.cat((event_latent, identity), dim=-1))
        context = self.context(states.flatten(-2))
        return event, context


class SSLModelV12(nn.Module):
    def __init__(self, context_width: int, method: str, width: int = 32, dropout: float = .10):
        super().__init__()
        self.method = method
        self.online = V12Backbone(context_width, method, width, dropout)
        self.target = copy.deepcopy(self.online.encoder)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.decoder = StateDecoderV12(width)
        self.future_event_heads = nn.ModuleList([nn.Linear(width, 2) for _ in range(3)])

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for target, online in zip(self.target.parameters(), self.online.encoder.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)

    def decode_future(self, future_states: torch.Tensor) -> torch.Tensor:
        """Return occurrence logit and positive magnitude for U/C/P."""
        return torch.stack([self.future_event_heads[index](future_states[..., index, :])
                            for index in range(3)], dim=-2)


def clone_backbone_state(model: SSLModelV12) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.online.state_dict().items()}


def load_target_from_encoder(target: StateEncoderV12, encoder: StateEncoderV12) -> None:
    target.load_state_dict(encoder.state_dict())

