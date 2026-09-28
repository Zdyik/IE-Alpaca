"""Human-prior event roles, co-occurrence and directed-chain landmark features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ie_alpaca.features.landmark_v3 import build_landmarks, feature_columns


EVENT_ROLES: dict[str, tuple[int, ...]] = {
    "severe": (11803, 11804),
    "proximal": (30000, 30005),
    "dangerous": (11401, 11402, 11403, 11405, 11406, 41001, 41003, 41023, 41029),
    "control": (30002, 30003, 30017, 41002, 41004, 41005, 41009),
    "environment": (60292, 60294),
    "quality": (41006, 41021),
}

# These five hypotheses are fixed before looking at validation scores.
RISK_CHAINS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "fatigue_to_control": ((41001, 41002, 41029), (30002, 30003, 30017)),
    "distraction_to_control": ((41003, 41004, 41009, 41023), (30002, 30003, 30017)),
    "speed_to_proximity": ((11401, 11402, 11403, 11405, 11406), (30000, 30005)),
    "close_to_fcw": ((30005,), (30000,)),
    "control_to_severe": ((30002, 30003, 30017), (11803, 11804)),
}


def _matrix(frame: pd.DataFrame, codes: tuple[int, ...], field: str) -> np.ndarray:
    values = []
    for code in codes:
        name = f"event_{code}_{field}"
        if name not in frame:
            raise ValueError(f"日表缺少 {name}")
        values.append(pd.to_numeric(frame[name], errors="coerce").fillna(0).to_numpy(dtype=float))
    return np.column_stack(values)


def _rate(numerator: float, denominator: float, scale: float = 1.0) -> float:
    return scale * numerator / denominator if denominator > 0 else np.nan


def _last_proximity(binary_days: np.ndarray) -> float:
    found = np.flatnonzero(binary_days)
    return 1.0 / (1.0 + len(binary_days) - found[-1] - 1) if len(found) else 0.0


def _directed_fraction(source: np.ndarray, target: np.ndarray, lag: int) -> float:
    """Fraction of source-active days followed by target in the next 1..lag days."""
    source_days = np.flatnonzero(source)
    if not len(source_days):
        return 0.0
    followed = 0
    for day in source_days:
        if target[day + 1:min(len(target), day + lag + 1)].any():
            followed += 1
    return followed / len(source_days)


def build_causal_landmarks(
    daily: pd.DataFrame, vehicle_ids: list[str], anchors: tuple[int, ...]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Add only prefix-derived causal-prior features to the exact V3 feature table."""
    base = build_landmarks(daily, vehicle_ids, anchors=anchors)
    base_columns, _ = feature_columns(base)
    lookup = {str(key): part.sort_values("day") for key, part in daily.groupby("gpsno", sort=False)}
    additions: list[dict[str, float | int | str]] = []

    for gpsno in vehicle_ids:
        vehicle = lookup[str(gpsno)]
        for anchor in anchors:
            history = vehicle.iloc[:anchor]
            recent = history.iloc[-7:]
            previous = history.iloc[-14:-7]
            km = float(pd.to_numeric(history.distance_km, errors="coerce").fillna(0).sum())
            hours = float(pd.to_numeric(history.drive_hours, errors="coerce").fillna(0).sum())
            row: dict[str, float | int | str] = {"gpsno": str(gpsno), "anchor_day": anchor}

            for role, codes in EVENT_ROLES.items():
                counts = _matrix(history, codes, "count")
                episodes = _matrix(history, codes, "episodes")
                recent_counts = _matrix(recent, codes, "count")
                previous_counts = _matrix(previous, codes, "count")
                total_count = float(counts.sum())
                total_episodes = float(episodes.sum())
                active = episodes.sum(axis=1) > 0
                prefix = f"role_{role}"
                row[f"{prefix}_count_per_day"] = total_count / anchor
                row[f"{prefix}_count_per_100km"] = _rate(total_count, km, 100)
                row[f"{prefix}_count_per_100hours"] = _rate(total_count, hours, 100)
                row[f"{prefix}_episodes_per_day"] = total_episodes / anchor
                row[f"{prefix}_episodes_per_100km"] = _rate(total_episodes, km, 100)
                row[f"{prefix}_recent7_per_day"] = float(recent_counts.sum()) / 7
                row[f"{prefix}_previous7_per_day"] = float(previous_counts.sum()) / 7
                row[f"{prefix}_delta7_per_day"] = float(recent_counts.sum() - previous_counts.sum()) / 7
                row[f"{prefix}_active_type_fraction"] = float((counts.sum(axis=0) > 0).mean())
                row[f"{prefix}_recency_proximity"] = _last_proximity(active)

            for chain, (source_codes, target_codes) in RISK_CHAINS.items():
                source = _matrix(history, source_codes, "episodes").sum(axis=1) > 0
                target = _matrix(history, target_codes, "episodes").sum(axis=1) > 0
                both = source & target
                row[f"cooccur_{chain}_per_day"] = float(both.sum()) / anchor
                row[f"cooccur_{chain}_given_source"] = float(both.sum()) / max(float(source.sum()), 1.0)
                row[f"chain_{chain}_within3_given_source"] = _directed_fraction(source, target, 3)
                row[f"chain_{chain}_within7_given_source"] = _directed_fraction(source, target, 7)
            additions.append(row)

    extra = pd.DataFrame(additions)
    result = base.merge(extra, on=["gpsno", "anchor_day"], how="left", validate="one_to_one")
    role_columns = [c for c in result if c.startswith("role_")]
    cooccur_columns = [c for c in result if c.startswith("cooccur_")]
    chain_columns = [c for c in result if c.startswith("chain_")]
    sets = {
        "e0a_v3_base": list(base_columns),
        "e0b_roles": list(base_columns) + role_columns,
        "e0c_cooccurrence": list(base_columns) + role_columns + cooccur_columns,
        "e0d_directed_chains": list(base_columns) + role_columns + cooccur_columns + chain_columns,
    }
    return result, sets

