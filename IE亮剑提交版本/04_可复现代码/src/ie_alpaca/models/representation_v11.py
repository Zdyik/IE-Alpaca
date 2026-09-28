"""Shared causal encoder and downstream risk network for every V11 experiment."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ie_alpaca.features.daily_v10 import AUX_GROUPS, CONTEXT_COLUMNS, EVENT_FEATURE_NAMES
from ie_alpaca.features.landmark_v4 import EVENT_CODES
from ie_alpaca.models.risk_chain_v10 import (
    RELATIONS, ROLES, STAGES, STATE_GROUPS, STATE_RELATIONS, _positions,
)


class CausalResidualConv(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = 2 * dilation
        self.depthwise = nn.Conv1d(width, width, 3, dilation=dilation, groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(F.pad(x.transpose(1, 2), (self.padding, 0)))
        y = self.pointwise(y).transpose(1, 2)
        return self.norm(x + self.dropout(F.gelu(y)))


class RepresentationEncoderV11(nn.Module):
    """Encode 24 event nodes plus exposure/dynamics/quality into causal daily states."""

    def __init__(self, context_width: int, width: int = 32, dropout: float = .10,
                 event_embedding: int = 8, role_embedding: int = 4, stage_embedding: int = 4):
        super().__init__()
        self.width = width
        self.event_encoder = nn.Sequential(nn.Linear(len(EVENT_FEATURE_NAMES), 32), nn.GELU(), nn.Linear(32, width))
        self.event_id = nn.Embedding(len(EVENT_CODES), event_embedding)
        self.role_id = nn.Embedding(len(ROLES), role_embedding)
        self.stage_id = nn.Embedding(len(STAGES), stage_embedding)
        self.identity = nn.Linear(width + event_embedding + role_embedding + stage_embedding, width)
        self.event_mask_token = nn.Parameter(torch.zeros(width))
        self.context_mask_token = nn.Parameter(torch.zeros(width))
        self.exposure_node = nn.Sequential(nn.Linear(context_width, width), nn.GELU(), nn.Linear(width, width))
        self.dynamics_node = nn.Sequential(nn.Linear(context_width, width), nn.GELU(), nn.Linear(width, width))
        self.quality_node = nn.Sequential(nn.Linear(context_width, width), nn.GELU(), nn.Linear(width, width))
        self.gate = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))
        self.tcn = nn.ModuleList([CausalResidualConv(width, d, dropout) for d in (1, 2, 4, 8)])
        self.graph_linear = nn.ModuleList([
            nn.ModuleList([nn.Linear(width, width, bias=False) for _ in RELATIONS]) for _ in range(2)])
        self.graph_norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.edge_strength = nn.Parameter(torch.zeros(2, len(RELATIONS)))
        self.lag_logits = nn.Parameter(torch.zeros(2, len(RELATIONS), 7))
        self.public_projection = nn.Sequential(nn.Linear(4 * width, 2 * width), nn.GELU(), nn.LayerNorm(2 * width))
        role_lookup = {code: i for i, values in enumerate(ROLES.values()) for code in values}
        stage_lookup = {code: i for i, values in enumerate(STAGES.values()) for code in values}
        self.register_buffer("event_indices", torch.arange(len(EVENT_CODES), dtype=torch.long))
        self.register_buffer("role_indices", torch.tensor([role_lookup[c] for c in EVENT_CODES]))
        self.register_buffer("stage_indices", torch.tensor([stage_lookup[c] for c in EVENT_CODES]))
        self.relations = [([list(STATE_GROUPS).index(a)], [list(STATE_GROUPS).index(b)])
                          for a, b in STATE_RELATIONS]
        self.risk_positions = [i for i, name in enumerate(STATE_GROUPS) if name != "quality"]
        self.escalation_positions = [list(STATE_GROUPS).index(name)
                                     for name in ("control", "close", "fcw", "severe")]

    def _nodes(self, events: torch.Tensor, context: torch.Tensor,
               event_mask: torch.Tensor | None, context_mask: torch.Tensor | None):
        content = self.event_encoder(events)
        identity = torch.cat((
            self.event_id(self.event_indices)[None, None].expand(*content.shape[:2], -1, -1),
            self.role_id(self.role_indices)[None, None].expand(*content.shape[:2], -1, -1),
            self.stage_id(self.stage_indices)[None, None].expand(*content.shape[:2], -1, -1),
        ), dim=-1)
        event_nodes = self.identity(torch.cat((content, identity), dim=-1))
        if event_mask is not None:
            event_nodes = event_nodes + event_mask[..., None].to(event_nodes.dtype) * self.event_mask_token
        exposure, dynamics, quality = self.exposure_node(context), self.dynamics_node(context), self.quality_node(context)
        if context_mask is not None:
            day_mask = context_mask.any(dim=-1, keepdim=True).to(context.dtype)
            token = day_mask * self.context_mask_token
            exposure, dynamics, quality = exposure + token, dynamics + token, quality + token
        gates = torch.sigmoid(self.gate(torch.cat((event_nodes, exposure[:, :, None].expand_as(event_nodes)), dim=-1)))
        event_nodes = event_nodes * gates
        states = torch.stack([event_nodes[:, :, _positions(codes)].mean(dim=2)
                              for codes in STATE_GROUPS.values()], dim=2)
        return torch.cat((states, exposure[:, :, None], dynamics[:, :, None], quality[:, :, None]), dim=2), gates.squeeze(-1)

    def _temporal(self, nodes: torch.Tensor) -> torch.Tensor:
        b, t, n, w = nodes.shape
        x = nodes.permute(0, 2, 1, 3).reshape(b * n, t, w)
        for layer in self.tcn:
            x = layer(x)
        return x.reshape(b, n, t, w).permute(0, 2, 1, 3)

    @staticmethod
    def _lagged(source: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(source)
        for lag in range(1, 8):
            result[:, lag:] += weights[lag - 1] * source[:, :-lag]
        return result

    def _graph(self, states: torch.Tensor) -> torch.Tensor:
        output = states
        for layer in range(2):
            messages = torch.zeros_like(output)
            for relation, (sources, targets) in enumerate(self.relations):
                source = output[:, :, sources].mean(dim=2)
                lag = self._lagged(source, torch.softmax(self.lag_logits[layer, relation], dim=0))
                message = torch.sigmoid(self.edge_strength[layer, relation]) * self.graph_linear[layer][relation](lag)
                messages[:, :, targets] += message[:, :, None]
            output = self.graph_norm[layer](output + F.gelu(messages))
        return output

    def forward(self, events: torch.Tensor, context: torch.Tensor,
                event_mask: torch.Tensor | None = None, context_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        nodes, gates = self._nodes(events, context, event_mask, context_mask)
        states = self._graph(self._temporal(nodes))
        event = states[:, :, self.risk_positions].mean(dim=2)
        exposure, dynamics, quality = states[:, :, -3], states[:, :, -2], states[:, :, -1]
        branches = torch.stack((event, exposure, dynamics, quality), dim=2)
        public = self.public_projection(branches.flatten(2))
        return {"states": states, "branches": branches, "public": public, "event_gates": gates}


class RiskNetV11(nn.Module):
    def __init__(self, context_width: int, width: int = 32, dropout: float = .10,
                 base_daily_hazard: float = .01):
        super().__init__()
        self.encoder = RepresentationEncoderV11(context_width, width=width, dropout=dropout)
        self.risk_head = nn.Sequential(nn.Linear(6 * width, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.aux_head = nn.Linear(width, len(AUX_GROUPS))
        self.hazard_bias = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(base_daily_hazard)))))

    def _anchor(self, encoded: dict[str, torch.Tensor], anchor: int) -> torch.Tensor:
        states, branches = encoded["states"][:, :anchor], encoded["branches"][:, :anchor]
        events = states[:, :, self.encoder.risk_positions]
        chronic = events.mean(dim=(1, 2))
        acute = events[:, max(0, anchor - 14):].mean(dim=(1, 2))
        escalation = states[:, max(0, anchor - 7):, self.encoder.escalation_positions].mean(dim=(1, 2))
        exposure = branches[:, max(0, anchor - 14):, 1].mean(dim=1)
        dynamics = branches[:, max(0, anchor - 14):, 2].mean(dim=1)
        quality = branches[:, max(0, anchor - 14):, 3].mean(dim=1)
        return self.risk_head(torch.cat((chronic, acute, escalation, exposure, dynamics, quality), dim=-1)).squeeze(-1)

    def forward(self, events: torch.Tensor, context: torch.Tensor, anchors: tuple[int, ...]):
        encoded = self.encoder(events, context)
        logits = torch.stack([self.hazard_bias + self._anchor(encoded, int(a)) for a in anchors], dim=1)
        aux = self.aux_head(encoded["states"][:, :, :len(STATE_GROUPS)].mean(dim=2))
        return logits, aux, encoded

