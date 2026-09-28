"""V12 semantic state masks and future risk-chain target definitions."""

from __future__ import annotations

import numpy as np

from ie_alpaca.features.daily_v10 import CONTEXT_COLUMNS
from ie_alpaca.features.landmark_v4 import EVENT_CODES


STATE_NAMES = ("exposure", "upstream", "control", "proximal", "history", "quality")
STATE_INDEX = {name: index for index, name in enumerate(STATE_NAMES)}

STATE_EVENT_CODES = {
    "exposure": (60292, 60294),
    "upstream": (11401, 11402, 11403, 11405, 11406, 41001, 41003, 41023, 41029),
    "control": (30002, 30003, 30017, 41002, 41004, 41005, 41009),
    "proximal": (30000, 30005),
    "history": (11803, 11804),
    "quality": (41006, 41021),
}
RISK_CHAIN_STATES = ("upstream", "control", "proximal")
RISK_CHAIN_EVENT_CODES = {name: STATE_EVENT_CODES[name] for name in RISK_CHAIN_STATES}


def _validate_partition() -> None:
    flattened = [code for codes in STATE_EVENT_CODES.values() for code in codes]
    if len(flattened) != len(EVENT_CODES) or set(flattened) != set(EVENT_CODES):
        raise RuntimeError("V12 state event groups must partition the 24 audited event codes")


_validate_partition()
PRIMARY_STATE = np.asarray([
    next(STATE_INDEX[name] for name, codes in STATE_EVENT_CODES.items() if code in codes)
    for code in EVENT_CODES
], dtype=np.int64)


def shuffled_primary_state(seed: int = 2026) -> np.ndarray:
    """Permute semantic labels while preserving every state's event-group size."""
    rng = np.random.default_rng(seed)
    labels = PRIMARY_STATE.copy()
    rng.shuffle(labels)
    return labels


CONTEXT_STATE_NAMES = {
    "exposure": (
        "distance_km", "drive_hours", "night_distance_km", "trip_starts",
        "speed_p50", "speed_p90",
    ),
    "control": (
        "speed_change_p90", "turn_rate_p90", "imu_valid_windows",
        "imu_accel_peak_p99", "imu_accel_variability_p90", "imu_gyro_peak",
    ),
    "quality": (
        "distance_invalid_rows", "long_gap_rows", "trajectory_recorded_today",
        "imu_recorded_today", "observed_stationary_day",
    ),
}


def _context_positions() -> dict[int, np.ndarray]:
    result: dict[int, list[int]] = {index: [] for index in range(len(STATE_NAMES))}
    for state_name, base_names in CONTEXT_STATE_NAMES.items():
        wanted = set(base_names) | {f"{name}_missing" for name in base_names}
        result[STATE_INDEX[state_name]] = [
            index for index, column in enumerate(CONTEXT_COLUMNS) if column in wanted
        ]
    return {key: np.asarray(value, dtype=np.int64) for key, value in result.items()}


CONTEXT_POSITIONS = _context_positions()


def state_block_mask(shape: tuple[int, int, int], rng: np.random.Generator,
                     blocks_per_item: int = 2, min_span: int = 2,
                     max_span: int = 5) -> np.ndarray:
    """Mask complete semantic states over short contiguous spans."""
    batch, days, states = shape
    if states != len(STATE_NAMES) or days < min_span:
        raise ValueError("invalid V12 state-mask shape")
    probabilities = np.asarray((.15, .23, .22, .18, .05, .17), dtype=float)
    probabilities /= probabilities.sum()
    mask = np.zeros(shape, dtype=bool)
    for item in range(batch):
        selected = rng.choice(states, size=blocks_per_item, replace=False, p=probabilities)
        for state in selected:
            span = int(rng.integers(min_span, min(max_span, days) + 1))
            start = int(rng.integers(0, days - span + 1))
            mask[item, start:start + span, int(state)] = True
    return mask


def state_masks_to_inputs(state_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand state blocks to coherent event bundles and related context fields."""
    if state_mask.ndim != 3 or state_mask.shape[-1] != len(STATE_NAMES):
        raise ValueError("V12 state mask must be [batch,day,6]")
    event_mask = state_mask[..., PRIMARY_STATE]
    context_mask = np.zeros((*state_mask.shape[:2], len(CONTEXT_COLUMNS)), dtype=bool)
    for state, positions in CONTEXT_POSITIONS.items():
        if len(positions):
            context_mask[..., positions] |= state_mask[..., state, None]
    return event_mask, context_mask


def risk_group_positions() -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(EVENT_CODES.index(code) for code in RISK_CHAIN_EVENT_CODES[name])
                 for name in RISK_CHAIN_STATES)


RISK_GROUP_POSITIONS = risk_group_positions()

