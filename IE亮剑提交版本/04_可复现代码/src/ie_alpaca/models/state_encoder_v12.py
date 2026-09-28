"""Six-state causal encoder and learnable risk-chain backbone for V12."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ie_alpaca.features.daily_v10 import AUX_GROUPS, CONTEXT_COLUMNS, EVENT_FEATURE_NAMES
from ie_alpaca.features.daily_v12 import (
    CONTEXT_POSITIONS, PRIMARY_STATE, RISK_CHAIN_STATES, STATE_INDEX, STATE_NAMES,
    shuffled_primary_state,
)
from ie_alpaca.features.landmark_v4 import EVENT_CODES
from ie_alpaca.models.risk_chain_v10 import ROLES, STAGES


class CausalResidualConvV12(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = 2 * dilation
        self.depthwise = nn.Conv1d(width, width, 3, dilation=dilation, groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(F.pad(values.transpose(1, 2), (self.padding, 0)))
        update = self.pointwise(update).transpose(1, 2)
        return self.norm(values + self.dropout(F.gelu(update)))


class StateEncoderV12(nn.Module):
    """Map daily event nodes to six inspectable, prefix-only state sequences."""

    def __init__(self, context_width: int, width: int = 32, dropout: float = .10,
                 shuffle_states: bool = False):
        super().__init__()
        self.width = width
        primary = shuffled_primary_state() if shuffle_states else PRIMARY_STATE
        self.register_buffer("primary_state", torch.as_tensor(primary, dtype=torch.long))
        self.register_buffer("event_indices", torch.arange(len(EVENT_CODES), dtype=torch.long))
        role_lookup = {code: index for index, codes in enumerate(ROLES.values()) for code in codes}
        stage_lookup = {code: index for index, codes in enumerate(STAGES.values()) for code in codes}
        self.register_buffer("role_indices", torch.tensor([role_lookup[code] for code in EVENT_CODES]))
        self.register_buffer("stage_indices", torch.tensor([stage_lookup[code] for code in EVENT_CODES]))

        self.event_encoder = nn.Sequential(
            nn.Linear(len(EVENT_FEATURE_NAMES), width), nn.GELU(), nn.Linear(width, width))
        self.event_id = nn.Embedding(len(EVENT_CODES), 8)
        self.role_id = nn.Embedding(len(ROLES), 4)
        self.stage_id = nn.Embedding(len(STAGES), 4)
        self.identity = nn.Linear(width + 16, width)
        self.event_mask_token = nn.Parameter(torch.zeros(width))
        self.context_mask_tokens = nn.Parameter(torch.zeros(3, width))
        self.exposure_node = nn.Sequential(nn.Linear(context_width, width), nn.GELU(), nn.Linear(width, width))
        self.control_node = nn.Sequential(nn.Linear(context_width, width), nn.GELU(), nn.Linear(width, width))
        self.quality_node = nn.Sequential(nn.Linear(context_width, width), nn.GELU(), nn.Linear(width, width))
        self.gate = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))

        initial = torch.full((len(EVENT_CODES), len(STATE_NAMES)), -1.0)
        initial.scatter_(1, self.primary_state[:, None].cpu(), 2.0)
        self.assignment_logits = nn.Parameter(initial)
        self.state_id = nn.Embedding(len(STATE_NAMES), width)
        self.tcn = nn.ModuleList([CausalResidualConvV12(width, dilation, dropout)
                                  for dilation in (1, 2, 4, 8)])
        self.final_norm = nn.LayerNorm(width)
        self.risk_indices = tuple(STATE_INDEX[name] for name in RISK_CHAIN_STATES)

    def assignment_weights(self) -> torch.Tensor:
        return torch.softmax(self.assignment_logits, dim=-1)

    def assignment_loss(self) -> torch.Tensor:
        return F.cross_entropy(self.assignment_logits, self.primary_state)

    def _event_nodes(self, events: torch.Tensor, exposure: torch.Tensor,
                     event_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        content = self.event_encoder(events)
        identity = torch.cat((
            self.event_id(self.event_indices)[None, None].expand(*content.shape[:2], -1, -1),
            self.role_id(self.role_indices)[None, None].expand(*content.shape[:2], -1, -1),
            self.stage_id(self.stage_indices)[None, None].expand(*content.shape[:2], -1, -1),
        ), dim=-1)
        nodes = self.identity(torch.cat((content, identity), dim=-1))
        if event_mask is not None:
            nodes = nodes + event_mask[..., None].to(nodes.dtype) * self.event_mask_token
        gates = torch.sigmoid(self.gate(torch.cat((nodes, exposure[:, :, None].expand_as(nodes)), dim=-1)))
        return nodes * gates, gates.squeeze(-1)

    def _temporal(self, states: torch.Tensor) -> torch.Tensor:
        batch, days, count, width = states.shape
        values = states.permute(0, 2, 1, 3).reshape(batch * count, days, width)
        for block in self.tcn:
            values = block(values)
        return values.reshape(batch, count, days, width).permute(0, 2, 1, 3)

    def forward(self, events: torch.Tensor, context: torch.Tensor,
                event_mask: torch.Tensor | None = None,
                context_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if events.ndim != 4 or events.shape[-2:] != (len(EVENT_CODES), len(EVENT_FEATURE_NAMES)):
            raise ValueError("V12 events must be [batch,time,24,8]")
        if context.shape[:2] != events.shape[:2] or context.shape[-1] != len(CONTEXT_COLUMNS):
            raise ValueError("V12 context shape is invalid")
        exposure = self.exposure_node(context)
        control = self.control_node(context)
        quality = self.quality_node(context)
        if context_mask is not None:
            for node_index, (state_name, node) in enumerate((("exposure", exposure),
                                                              ("control", control),
                                                              ("quality", quality))):
                positions = CONTEXT_POSITIONS[STATE_INDEX[state_name]].tolist()
                masked = context_mask[..., positions].any(dim=-1).to(context.dtype)[:, :, None]
                if node_index == 0:
                    exposure = node + masked * self.context_mask_tokens[node_index]
                elif node_index == 1:
                    control = node + masked * self.context_mask_tokens[node_index]
                else:
                    quality = node + masked * self.context_mask_tokens[node_index]
        event_nodes, gates = self._event_nodes(events, exposure, event_mask)
        weights = self.assignment_weights()
        pooled = torch.einsum("btew,es->btsw", event_nodes, weights)
        pooled = pooled / weights.sum(dim=0).clamp_min(.1)[None, None, :, None]
        pooled[:, :, STATE_INDEX["exposure"]] += exposure
        pooled[:, :, STATE_INDEX["control"]] += control
        pooled[:, :, STATE_INDEX["quality"]] += quality
        pooled = pooled + self.state_id.weight[None, None]
        states = self.final_norm(self._temporal(pooled))
        return {"states": states, "event_gates": gates, "assignment": weights}


class TransitionPredictorV12(nn.Module):
    """Predict U/C/P changes with either a dense baseline or a directed graph."""

    HORIZONS = (1, 3, 7)
    GRAPH_MODES = {"hrc", "hrc_no_prior", "hrc_shuffle"}

    def __init__(self, width: int, mode: str):
        super().__init__()
        if mode not in {"none", "dense", *self.GRAPH_MODES}:
            raise ValueError(f"unsupported V12 transition mode: {mode}")
        self.width, self.mode = width, mode
        self.risk_indices = tuple(STATE_INDEX[name] for name in RISK_CHAIN_STATES)
        self.horizon_id = nn.Embedding(len(self.HORIZONS), 8)
        self.dense = nn.Sequential(nn.Linear(len(STATE_NAMES) * width + 8, 3 * width), nn.GELU(),
                                   nn.Linear(3 * width, 3 * width))
        self.source = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(3)])
        self.condition = nn.Sequential(nn.Linear(2 * width + 8, 3 * width), nn.GELU(),
                                       nn.Linear(3 * width, 3 * width))
        self.target = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width)) for _ in range(3)])
        self.beta = nn.Parameter(torch.zeros(()))

        prior = torch.eye(3)
        prior[0, 1] = 1.0  # U -> C
        prior[0, 2] = 1.0  # U -> P
        prior[1, 2] = 1.0  # C -> P
        if mode == "hrc_shuffle":
            permutation = torch.tensor((2, 0, 1))
            prior = prior[permutation][:, permutation]
        self.register_buffer("human_prior", prior)
        initial = torch.full((len(self.HORIZONS), 3, 3), -1.5)
        if mode in {"hrc", "hrc_shuffle"}:
            initial = torch.where(prior[None] > 0, torch.full_like(initial, .5), initial)
        elif mode == "hrc_no_prior":
            initial.fill_(-.5)
        self.edge_logits = nn.Parameter(initial)

    @property
    def uses_dynamics(self) -> bool:
        return self.mode != "none"

    def forward(self, states: torch.Tensor, horizon_index: int) -> torch.Tensor:
        """Return predicted deltas [batch,time,3,width]."""
        if not self.uses_dynamics:
            return states.new_zeros((*states.shape[:2], 3, self.width))
        horizon = self.horizon_id.weight[horizon_index].view(*([1] * (states.ndim - 2)), 8)
        horizon = horizon.expand(*states.shape[:-2], -1)
        if self.mode == "dense":
            flat = states.flatten(-2)
            return self.dense(torch.cat((flat, horizon), dim=-1)).view(*states.shape[:-2], 3, self.width)
        risk = states[..., list(self.risk_indices), :]
        transformed = torch.stack([self.source[index](risk[..., index, :]) for index in range(3)], dim=-2)
        edges = torch.sigmoid(self.edge_logits[horizon_index])
        message = torch.einsum("...iw,ij->...jw", transformed, edges)
        condition = self.condition(torch.cat((states[..., STATE_INDEX["exposure"], :],
                                              states[..., STATE_INDEX["quality"], :], horizon), dim=-1))
        condition = condition.view(*states.shape[:-2], 3, self.width)
        updated = risk + torch.tanh(self.beta) * (message + condition)
        return torch.stack([self.target[index](updated[..., index, :])
                            for index in range(3)], dim=-2)

    def graph_loss(self, progress: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode not in self.GRAPH_MODES:
            zero = self.edge_logits.sum() * 0
            return zero, zero, zero
        edges = torch.sigmoid(self.edge_logits)
        sparsity = edges.mean()
        if self.mode in {"hrc", "hrc_shuffle"}:
            prior = self.human_prior[None].expand_as(edges)
            prior_loss = F.binary_cross_entropy(edges, prior)
        else:
            prior_loss = edges.sum() * 0
        anneal = max(0.0, 1.0 - progress / .30)
        total = .01 * sparsity + .02 * anneal * prior_loss
        return total, sparsity, prior_loss

    def edge_weights(self) -> torch.Tensor:
        return torch.sigmoid(self.edge_logits)


def transition_mode(method: str) -> str:
    if method in {"state_mae", "state_shuffle"}:
        return "none"
    if method == "dynamics":
        return "dense"
    if method in TransitionPredictorV12.GRAPH_MODES:
        return method
    raise ValueError(f"unsupported V12 method: {method}")


class V12Backbone(nn.Module):
    def __init__(self, context_width: int, method: str, width: int = 32, dropout: float = .10):
        super().__init__()
        self.method = method
        self.encoder = StateEncoderV12(context_width, width, dropout, shuffle_states=method == "state_shuffle")
        self.transition = TransitionPredictorV12(width, transition_mode(method))

    def forward(self, events: torch.Tensor, context: torch.Tensor,
                event_mask: torch.Tensor | None = None,
                context_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.encoder(events, context, event_mask, context_mask)


class RiskNetV12(nn.Module):
    def __init__(self, context_width: int, method: str, width: int = 32,
                 dropout: float = .10, base_daily_hazard: float = .01):
        super().__init__()
        self.method, self.width = method, width
        self.backbone = V12Backbone(context_width, method, width, dropout)
        risk_input = 24 * width + 27
        self.risk_head = nn.Sequential(nn.Linear(risk_input, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.aux_head = nn.Linear(width, len(AUX_GROUPS))
        self.hazard_bias = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(base_daily_hazard)))))

    def _anchor(self, states: torch.Tensor, anchor: int) -> torch.Tensor:
        prefix = states[:, :anchor]
        chronic = prefix.mean(dim=1).flatten(1)
        acute = prefix[:, max(0, anchor - 14):].mean(dim=1).flatten(1)
        recent = prefix[:, max(0, anchor - 7):, list(self.backbone.encoder.risk_indices)].mean(dim=1)
        earlier_start = max(0, anchor - 14)
        earlier_stop = max(1, anchor - 7)
        earlier = prefix[:, earlier_start:earlier_stop, list(self.backbone.encoder.risk_indices)].mean(dim=1)
        trend = (recent - earlier).flatten(1)
        current = prefix[:, -1]
        transitions = torch.cat([self.backbone.transition(current[:, None], index).flatten(1)
                                 for index in range(3)], dim=-1)
        if self.backbone.transition.mode in TransitionPredictorV12.GRAPH_MODES:
            edges = self.backbone.transition.edge_weights().flatten()[None].expand(len(states), -1)
        else:
            edges = states.new_zeros((len(states), 27))
        return self.risk_head(torch.cat((chronic, acute, trend, transitions, edges), dim=-1)).squeeze(-1)

    def forward(self, events: torch.Tensor, context: torch.Tensor, anchors: tuple[int, ...]):
        encoded = self.backbone(events, context)
        states = encoded["states"]
        logits = torch.stack([self.hazard_bias + self._anchor(states, int(anchor)) for anchor in anchors], dim=1)
        auxiliary = self.aux_head(states.mean(dim=2))
        details = {**encoded, "edge_weights": self.backbone.transition.edge_weights(),
                   "transition_beta": torch.tanh(self.backbone.transition.beta)}
        return logits, auxiliary, details

