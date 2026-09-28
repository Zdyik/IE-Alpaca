"""V10 event-node TCN with optional roles, lag graph, auxiliary head and quality gate."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ie_alpaca.features.daily_v10 import AUX_GROUPS, CONTEXT_COLUMNS, EVENT_FEATURE_NAMES
from ie_alpaca.features.landmark_v4 import EVENT_CODES


ROLES = {
    "severe": (11803, 11804),
    "proximal": (30000, 30005),
    "dangerous": (11401, 11402, 11403, 11405, 11406, 41001, 41003, 41023, 41029),
    "control": (30002, 30003, 30017, 41002, 41004, 41005, 41009),
    "environment": (60292, 60294),
    "quality": (41006, 41021),
}
STAGES = {
    "context": (60292, 60294, 41006, 41021),
    "upstream": (11401, 11402, 11403, 11405, 11406, 41001, 41003, 41023, 41029),
    "control": (30002, 30003, 30017, 41002, 41004, 41005, 41009),
    "proximal": (30000, 30005),
    "severe": (11803, 11804),
}
STATE_GROUPS = {
    "speed": (11401, 11402, 11403, 11405, 11406),
    "fatigue": (41001, 41002, 41029),
    "distraction": (41003, 41004, 41005, 41009, 41023),
    "control": (30002, 30003, 30017),
    "close": (30005,),
    "fcw": (30000,),
    "severe": (11803, 11804),
    "environment": (60292, 60294),
    "quality": (41006, 41021),
}
STATE_RELATIONS = (
    ("fatigue", "control"), ("distraction", "control"),
    ("speed", "close"), ("speed", "control"),
    ("control", "fcw"), ("close", "fcw"), ("fcw", "severe"),
)
RELATIONS = tuple((STATE_GROUPS[source], STATE_GROUPS[target])
                  for source, target in STATE_RELATIONS)
DMS_CODES = (41001, 41002, 41003, 41004, 41005, 41009, 41023, 41029)
QUALITY_CODES = (41006, 41021)


def _positions(codes: tuple[int, ...]) -> list[int]:
    return [EVENT_CODES.index(code) for code in codes]


class CausalResidualConv(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = 2 * dilation
        # Node-wise temporal filtering is depthwise separable: the temporal
        # receptive field is unchanged, while CPU work is far smaller than a
        # dense width×width convolution for every one of the 27 nodes.
        self.depthwise = nn.Conv1d(width, width, kernel_size=3, dilation=dilation,
                                   groups=width)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # [batch*nodes, time, width]; left padding makes every state prefix-only.
        convolved = self.depthwise(F.pad(values.transpose(1, 2), (self.padding, 0)))
        convolved = self.pointwise(convolved).transpose(1, 2)
        return self.norm(values + self.dropout(F.gelu(convolved)))


class RiskChainNetV10(nn.Module):
    MODES = {"flat", "role", "chain", "chain_shuffle", "chain_aux", "quality_chain"}

    def __init__(self, *, context_width: int, mode: str, width: int = 24,
                 event_embedding: int = 8, role_embedding: int = 4,
                 stage_embedding: int = 4, dropout: float = .15,
                 modality_dropout: float = .15, base_daily_hazard: float = .01):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported V10 mode: {mode}")
        self.mode, self.width, self.modality_dropout = mode, width, modality_dropout
        self.use_roles = mode != "flat"
        self.use_graph = mode in {"chain", "chain_shuffle", "chain_aux", "quality_chain"}
        self.use_aux = mode in {"chain_aux", "quality_chain"}
        self.use_quality = mode == "quality_chain"
        self.event_encoder = nn.Sequential(
            nn.Linear(len(EVENT_FEATURE_NAMES), 32), nn.GELU(), nn.Linear(32, width))
        self.event_id = nn.Embedding(len(EVENT_CODES), event_embedding)
        self.role_id = nn.Embedding(len(ROLES), role_embedding)
        self.stage_id = nn.Embedding(len(STAGES), stage_embedding)
        self.identity_projection = nn.Linear(width + event_embedding + role_embedding + stage_embedding, width)
        self.exposure_node = nn.Sequential(nn.Linear(context_width, width), nn.GELU(), nn.Linear(width, width))
        self.dynamics_node = nn.Sequential(nn.Linear(context_width, width), nn.GELU(), nn.Linear(width, width))
        self.quality_node = nn.Sequential(nn.Linear(context_width, width), nn.GELU(), nn.Linear(width, width))
        self.gate = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))
        self.tcn = nn.ModuleList([CausalResidualConv(width, dilation, dropout) for dilation in (1, 2, 4)])
        self.graph_linear = nn.ModuleList([
            nn.ModuleList([nn.Linear(width, width, bias=False) for _ in RELATIONS]) for _ in range(2)])
        self.graph_norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.edge_strength = nn.Parameter(torch.zeros(2, len(RELATIONS)))
        self.lag_logits = nn.Parameter(torch.zeros(2, len(RELATIONS), 7))
        self.missing_dms_prior = nn.Parameter(torch.zeros(width))
        self.quality_gate = nn.Sequential(nn.Linear(width, 8), nn.GELU(), nn.Linear(8, 1))
        self.flat_head = nn.Sequential(nn.Linear(2 * width, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.structured_head = nn.Sequential(
            nn.Linear(4 * width, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.aux_head = nn.Linear(width, len(AUX_GROUPS))
        self.hazard_bias = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(base_daily_hazard)))))
        role_lookup = {code: index for index, codes in enumerate(ROLES.values()) for code in codes}
        stage_lookup = {code: index for index, codes in enumerate(STAGES.values()) for code in codes}
        self.register_buffer("event_indices", torch.arange(len(EVENT_CODES), dtype=torch.long))
        self.register_buffer("role_indices", torch.tensor([role_lookup[code] for code in EVENT_CODES]))
        self.register_buffer("stage_indices", torch.tensor([stage_lookup[code] for code in EVENT_CODES]))
        self.dms_positions = _positions(DMS_CODES)
        self.quality_positions = _positions(QUALITY_CODES)
        self.state_positions = {name: index for index, name in enumerate(STATE_GROUPS)}
        self.risk_positions = [self.state_positions[name] for name in STATE_GROUPS if name != "quality"]
        self.escalation_positions = [self.state_positions[name] for name in
                                     ("control", "close", "fcw", "severe")]
        self.relations = self._relation_positions(shuffle=mode == "chain_shuffle")

    @staticmethod
    def _relation_positions(shuffle: bool) -> list[tuple[list[int], list[int]]]:
        mapping = {index: index for index in range(len(STATE_GROUPS))}
        if shuffle:
            eligible = list(range(len(STATE_GROUPS) - 1))
            shuffled = np.random.default_rng(2026).permutation(eligible).tolist()
            mapping.update(dict(zip(eligible, shuffled)))
        return [([mapping[list(STATE_GROUPS).index(source)]],
                 [mapping[list(STATE_GROUPS).index(target)]])
                for source, target in STATE_RELATIONS]

    def _nodes(self, events: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if events.ndim != 4 or events.shape[-2:] != (len(EVENT_CODES), len(EVENT_FEATURE_NAMES)):
            raise ValueError("V10 events must be [batch,time,24,8]")
        if context.shape[:2] != events.shape[:2] or context.shape[-1] != len(CONTEXT_COLUMNS):
            raise ValueError("V10 context shape is invalid")
        content = self.event_encoder(events)
        event = self.event_id(self.event_indices)[None, None].expand(*content.shape[:2], -1, -1)
        if self.use_roles:
            role = self.role_id(self.role_indices)[None, None].expand(*content.shape[:2], -1, -1)
            stage = self.stage_id(self.stage_indices)[None, None].expand(*content.shape[:2], -1, -1)
        else:
            role = torch.zeros((*content.shape[:3], self.role_id.embedding_dim), device=content.device, dtype=content.dtype)
            stage = torch.zeros((*content.shape[:3], self.stage_id.embedding_dim), device=content.device, dtype=content.dtype)
        event_nodes = self.identity_projection(torch.cat((content, event, role, stage), dim=-1))
        exposure = self.exposure_node(context)
        dynamics = self.dynamics_node(context)
        quality = self.quality_node(context)
        context_for_gate = exposure[:, :, None].expand(-1, -1, len(EVENT_CODES), -1)
        gates = torch.sigmoid(self.gate(torch.cat((event_nodes, context_for_gate), dim=-1)))
        event_nodes = event_nodes * gates
        if self.use_quality:
            reliability = torch.sigmoid(self.quality_gate(quality))[:, :, None, :]
            if self.training and self.modality_dropout > 0:
                dropped = torch.rand((len(events), 1, 1, 1), device=events.device) < self.modality_dropout
                reliability = torch.where(dropped, torch.zeros_like(reliability), reliability)
            observed = event_nodes[:, :, self.dms_positions]
            prior = self.missing_dms_prior.view(1, 1, 1, -1)
            event_nodes[:, :, self.dms_positions] = reliability * observed + (1 - reliability) * prior
        # Individual event gates stay inspectable, then related events are pooled
        # into nine semantic states before the expensive temporal encoder.
        state_nodes = torch.stack([
            event_nodes[:, :, _positions(codes)].mean(dim=2) for codes in STATE_GROUPS.values()
        ], dim=2)
        nodes = torch.cat((state_nodes, exposure[:, :, None], dynamics[:, :, None], quality[:, :, None]), dim=2)
        return nodes, gates.squeeze(-1)

    def _temporal(self, nodes: torch.Tensor) -> torch.Tensor:
        batch, days, count, width = nodes.shape
        values = nodes.permute(0, 2, 1, 3).reshape(batch * count, days, width)
        for block in self.tcn:
            values = block(values)
        return values.reshape(batch, count, days, width).permute(0, 2, 1, 3)

    @staticmethod
    def _lagged(source: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(source)
        for lag in range(1, 8):
            result[:, lag:] += weights[lag - 1] * source[:, :-lag]
        return result

    def _graph(self, nodes: torch.Tensor) -> torch.Tensor:
        if not self.use_graph:
            return nodes
        output = nodes
        for layer in range(2):
            messages = torch.zeros_like(output)
            for relation, (source_positions, target_positions) in enumerate(self.relations):
                source = output[:, :, source_positions].mean(dim=2)
                lagged = self._lagged(source, torch.softmax(self.lag_logits[layer, relation], dim=0))
                message = torch.sigmoid(self.edge_strength[layer, relation]) * self.graph_linear[layer][relation](lagged)
                messages[:, :, target_positions] += message[:, :, None] / len(target_positions)
            output = self.graph_norm[layer](output + F.gelu(messages))
        return output

    def _pool_anchor(self, states: torch.Tensor, anchor: int) -> torch.Tensor:
        prefix = states[:, :anchor]
        if self.mode == "flat":
            mean = prefix.mean(dim=(1, 2))
            last = prefix[:, -1].mean(dim=1)
            return self.flat_head(torch.cat((mean, last), dim=-1)).squeeze(-1)
        chronic = prefix[:, :, self.risk_positions].mean(dim=(1, 2))
        acute = prefix[:, max(0, anchor - 14):, self.risk_positions].mean(dim=(1, 2))
        escalation = prefix[:, max(0, anchor - 7):, self.escalation_positions].mean(dim=(1, 2))
        quality = prefix[:, max(0, anchor - 7):, -1].mean(dim=1)
        return self.structured_head(torch.cat((chronic, acute, escalation, quality), dim=-1)).squeeze(-1)

    def forward(self, events: torch.Tensor, context: torch.Tensor,
                anchors: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        nodes, gates = self._nodes(events, context)
        states = self._graph(self._temporal(nodes))
        logits = torch.stack([self.hazard_bias + self._pool_anchor(states, int(anchor)) for anchor in anchors], dim=1)
        auxiliary = self.aux_head(states[:, :, :len(STATE_GROUPS)].mean(dim=2)) if self.use_aux else None
        details = {"event_gates": gates,
                   "edge_strength": torch.sigmoid(self.edge_strength),
                   "lag_weights": torch.softmax(self.lag_logits, dim=-1)}
        return logits, auxiliary, details
