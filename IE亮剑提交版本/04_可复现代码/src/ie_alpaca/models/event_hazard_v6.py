"""V4 hazard equations with only the two hidden widths made configurable."""

from __future__ import annotations

from torch import nn

from ie_alpaca.features.landmark_v4 import EVENT_METRICS
from ie_alpaca.models.event_hazard_v4 import EventHazardNet


class EventHazardNetV6(EventHazardNet):
    def __init__(self, context_width: int, *, learned_weights: bool,
                 dropout: float, base_daily_hazard: float,
                 event_hidden: int, context_hidden: int):
        if event_hidden < 1 or context_hidden < 1:
            raise ValueError("hidden widths must be positive")
        super().__init__(context_width, learned_weights=learned_weights,
                         dropout=dropout, base_daily_hazard=base_daily_hazard)
        self.event_encoder = nn.Sequential(
            nn.Linear(len(EVENT_METRICS), event_hidden), nn.Tanh(),
            nn.Linear(event_hidden, 1),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_width, context_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(context_hidden, 1),
        )
