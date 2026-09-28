"""Leakage-checked vehicle sequences and observable multi-horizon targets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from ie_alpaca.data.task_one import FIRST_DAY, PREDICTION_ANCHOR, REFERENCE_ANCHOR


HORIZONS = (7, 14, 21, 30, 40)
ANCHOR_DAYS = (20, 30, 39, 46, 53)  # inclusive history length; all targets end by day 60
MAX_DAYS = 60
EXCLUDED = {"gpsno", "day", "energy_type", "gps_distance_check_km"}


@dataclass
class SequenceData:
    vehicles: list[str]
    features: list[str]
    groups: dict[str, list[int]]
    raw: np.ndarray  # [vehicles, 60, features], NaN is missing
    events: np.ndarray  # [vehicles, 60], 1 if accident or near miss observed that day
    event_feed: np.ndarray  # [vehicles], observed anywhere by July 30

    def positions(self, ids: set[str]) -> np.ndarray:
        return np.asarray([i for i, vehicle in enumerate(self.vehicles) if vehicle in ids], dtype=np.int64)


def build_sequences(daily: pd.DataFrame, reference: pd.DataFrame) -> SequenceData:
    features = [
        col for col in daily.columns
        if col not in EXCLUDED and (pd.api.types.is_numeric_dtype(daily[col]) or pd.api.types.is_bool_dtype(daily[col]))
    ]
    if not features:
        raise ValueError("日特征中没有数值列")
    vehicles = sorted(daily.gpsno.unique().tolist())
    if len(daily) != len(vehicles) * MAX_DAYS:
        raise ValueError("每辆车必须有完整的 60 日历日索引")
    ordered = daily.sort_values(["gpsno", "day"]).reset_index(drop=True)
    expected_days = [FIRST_DAY + timedelta(days=i) for i in range(MAX_DAYS)]
    for _, frame in ordered.groupby("gpsno", sort=False):
        if frame.day.tolist() != expected_days:
            raise ValueError("车辆日历索引缺失、重复或超出 06-01..07-30")
    raw = ordered[features].astype(float).to_numpy(dtype=np.float32, na_value=np.nan)
    raw = raw.reshape(len(vehicles), MAX_DAYS, len(features))
    raw[~np.isfinite(raw)] = np.nan
    accidents = ordered[["event_11803_count", "event_11804_count"]].fillna(0).astype(float)
    events = accidents.gt(0).any(axis=1).to_numpy().reshape(len(vehicles), MAX_DAYS)
    feed = ordered.has_event_feed.to_numpy(dtype=bool).reshape(len(vehicles), MAX_DAYS).any(axis=1)
    result = SequenceData(vehicles, features, feature_groups(features), raw, events, feed)
    for bag in reference.itertuples(index=False):
        i = vehicles.index(bag.gpsno)
        if not feed[i]:
            raise ValueError(f"有监督车辆缺少事件源：{bag.gpsno}")
        if int(events[i, 20:60].any()) != int(bag.label):
            raise ValueError(f"06-20 代理标签与逐日事件不一致：{bag.gpsno}")
    return result


def feature_groups(features: list[str]) -> dict[str, list[int]]:
    groups = {"event": [], "trajectory": [], "imu": [], "quality": []}
    for i, name in enumerate(features):
        if name.startswith("event_") or name.startswith("events_per_"):
            key = "event"
        elif name.startswith("imu_"):
            key = "imu"
        elif name.startswith(("has_", "observed_")) or name.endswith("_recorded_today"):
            key = "quality"
        else:
            key = "trajectory"
        groups[key].append(i)
    if any(not group for group in groups.values()):
        raise ValueError(f"模态特征缺失：{groups}")
    return groups


def fit_scaler(data: SequenceData, train_ids: set[str]) -> dict:
    rows = data.raw[data.positions(train_ids), :20, :].reshape(-1, len(data.features)).astype(np.float64)
    if rows.size == 0:
        raise ValueError("标准化参数没有训练车辆")
    params = []
    for i, name in enumerate(data.features):
        valid = rows[:, i][np.isfinite(rows[:, i])]
        log = (name.endswith(("_count", "_episodes", "_rows", "_samples", "_windows"))
               or name in {"trip_starts", "trajectory_rows", "moving_rows", "long_gap_rows"})
        log = bool(log and len(valid) and np.min(valid) >= 0)
        if log:
            valid = np.log1p(valid)
        if len(valid):
            lo, hi = np.quantile(valid, [0.01, 0.99])
            clipped = np.clip(valid, lo, hi)
            center = float(np.median(clipped))
            q25, q75 = np.quantile(clipped, [0.25, 0.75])
            scale = max(float(q75 - q25), 1.0)
        else:
            lo = hi = center = 0.0
            scale = 1.0
        params.append({"name": name, "log1p": log, "low": float(lo), "high": float(hi),
                       "center": center, "scale": scale, "fit_count": int(len(valid))})
    return {"fit_days": "2026-06-01..2026-06-20", "fit_vehicles": len(train_ids), "columns": params}


def transform(data: SequenceData, scaler: dict) -> tuple[np.ndarray, np.ndarray]:
    if [p["name"] for p in scaler["columns"]] != data.features:
        raise ValueError("标准化参数的列顺序与输入不一致")
    mask = np.isfinite(data.raw)
    values = np.where(mask, data.raw, 0).astype(np.float32)
    for i, p in enumerate(scaler["columns"]):
        col = values[:, :, i]
        if p["log1p"]:
            col = np.log1p(np.maximum(col, 0))
        values[:, :, i] = np.where(mask[:, :, i],
                                   (np.clip(col, p["low"], p["high"]) - p["center"]) / p["scale"], 0)
    np.clip(values, -10, 10, out=values)
    return values, mask.astype(np.float32)


def labels_for(data: SequenceData, vehicle_indices: np.ndarray, anchors: tuple[int, ...] = ANCHOR_DAYS):
    """One row per car/anchor; unavailable future horizons are masked, never filled as 0."""
    rows, targets, availability = [], [], []
    for vehicle_i in vehicle_indices:
        for days in anchors:
            if not 1 <= days <= MAX_DAYS:
                raise ValueError("锚点超出观测期")
            rows.append((int(vehicle_i), days))
            targets.append([int(data.events[vehicle_i, days:days+h].any()) if days+h <= MAX_DAYS else 0
                            for h in HORIZONS])
            availability.append([days+h <= MAX_DAYS for h in HORIZONS])
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2), np.asarray(targets, dtype=np.float32).reshape(-1, len(HORIZONS)), np.asarray(availability, dtype=np.float32).reshape(-1, len(HORIZONS))


def assert_primary_boundary() -> None:
    if FIRST_DAY + timedelta(days=19) != REFERENCE_ANCHOR:
        raise AssertionError("主验证锚点与 20 天输入不一致")
    if REFERENCE_ANCHOR + timedelta(days=40) != PREDICTION_ANCHOR:
        raise AssertionError("主验证标签必须完整落在 60 天内")
