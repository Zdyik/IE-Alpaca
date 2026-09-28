"""V11 self-supervised views built from the audited V10 daily tensors."""

from __future__ import annotations

import numpy as np

from ie_alpaca.features.daily_v10 import (  # re-export one frozen data contract
    AUX_GROUPS, CONTEXT_COLUMNS, EVENT_FEATURE_NAMES, NodeDayScaler,
    build_day_table, future_group_targets,
)


EVENT_BUNDLE_COLUMNS = (0, 1, 2, 3, 4)


def event_bundle_mask(shape: tuple[int, int, int], rng: np.random.Generator,
                      day_fraction: float = .15, group_fraction: float = .10) -> np.ndarray:
    """Mask coherent event bundles; derived count/rate/occurrence fields stay together."""
    batch, days, events = shape
    mask = np.zeros(shape, dtype=bool)
    for item in range(batch):
        target_days = max(1, int(round(days * day_fraction)))
        covered = 0
        while covered < target_days:
            length = int(rng.integers(2, min(5, days) + 1))
            start = int(rng.integers(0, max(1, days - length + 1)))
            mask[item, start:start + length, :] = True
            covered += length
        count = max(1, int(round(events * group_fraction)))
        selected = rng.choice(events, size=count, replace=False)
        mask[item, :, selected] = True
    return mask


def context_block_mask(shape: tuple[int, int, int], rng: np.random.Generator,
                       fraction: float = .10) -> np.ndarray:
    """Mask complete short spans of context channels to simulate sensor gaps."""
    batch, days, width = shape
    mask = np.zeros(shape, dtype=bool)
    channels = max(1, int(round(width * fraction)))
    for item in range(batch):
        selected = rng.choice(width, size=channels, replace=False)
        length = int(rng.integers(2, min(5, days) + 1))
        start = int(rng.integers(0, max(1, days - length + 1)))
        mask[item, start:start + length, selected] = True
    return mask


def apply_masks(events: np.ndarray, context: np.ndarray, event_mask: np.ndarray,
                context_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero only inputs selected by SSL masks; model receives masks separately."""
    masked_events = events.copy()
    for column in EVENT_BUNDLE_COLUMNS:
        masked_events[..., column][event_mask] = 0.0
    masked_context = context.copy()
    masked_context[context_mask] = 0.0
    return masked_events, masked_context


def light_view_masks(shape_events: tuple[int, int, int], shape_context: tuple[int, int, int],
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A semantics-preserving contrastive view: small coherent missing spans only."""
    return (event_bundle_mask(shape_events, rng, day_fraction=.05, group_fraction=.04),
            context_block_mask(shape_context, rng, fraction=.04))

