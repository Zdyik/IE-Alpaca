"""Vehicle-level features using only days at or before each prediction anchor."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


LOOKBACK_WINDOWS = (7, 14, 20)
EVENT_COLUMNS = ("event_11804_count", "event_11803_count", "event_total_count", "event_total_episodes")


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _sum_or_nan(series: pd.Series) -> float:
    values = _number(series)
    return float(values.sum()) if values.notna().any() else float("nan")


def _rate(numerator: float, denominator: float, scale: float = 1.0) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(scale * numerator / denominator)


def build_features(daily: pd.DataFrame, bags: pd.DataFrame, lookback_days: int = 20) -> pd.DataFrame:
    if lookback_days != 20:
        raise ValueError("V1 只支持经代理标签验证的 20 天历史")
    if bags.duplicated(["gpsno", "anchor_date"]).any():
        raise ValueError("样本索引包含重复 gpsno × anchor_date")
    lookup = {str(gpsno): group.set_index("day").sort_index() for gpsno, group in daily.groupby("gpsno", sort=False)}
    rows: list[dict] = []
    for bag in bags.itertuples(index=False):
        gpsno, anchor = str(bag.gpsno), bag.anchor_date
        vehicle = lookup.get(gpsno)
        if vehicle is None:
            raise ValueError(f"缺少车辆日特征：{gpsno}")
        start = anchor - timedelta(days=lookback_days - 1)
        frame = vehicle.loc[(vehicle.index >= start) & (vehicle.index <= anchor)].copy()
        if len(frame) != lookback_days:
            raise ValueError(f"{gpsno} 在 {anchor} 前只有 {len(frame)} 日，预期 {lookback_days} 日")
        result: dict[str, object] = {"gpsno": gpsno, "anchor_date": anchor}
        result["event_feed_at_anchor"] = int(bool(frame.has_event_feed.iloc[-1]))
        result["trajectory_at_anchor"] = int(bool(frame.has_trajectory.iloc[-1]))
        for window in LOOKBACK_WINDOWS:
            part = frame.iloc[-window:]
            observed_event = part.has_event_feed.fillna(False).astype(bool)
            observed_traj = part.trajectory_recorded_today.fillna(False).astype(bool)
            km = _sum_or_nan(part.distance_km)
            hours = _sum_or_nan(part.drive_hours)
            result[f"trajectory_recorded_days_{window}"] = int(observed_traj.sum())
            result[f"distance_km_{window}"] = km
            result[f"drive_hours_{window}"] = hours
            result[f"km_per_calendar_day_{window}"] = km / window if np.isfinite(km) else np.nan
            result[f"hours_per_calendar_day_{window}"] = hours / window if np.isfinite(hours) else np.nan
            result[f"km_per_recorded_day_{window}"] = _rate(km, float(observed_traj.sum()))
            result[f"event_feed_days_{window}"] = int(observed_event.sum())
            for column in EVENT_COLUMNS:
                # No feed is unknown, while an observed day with zero events is a true zero.
                counts = _number(part[column]).where(observed_event)
                count = _sum_or_nan(counts)
                result[f"{column}_{window}"] = count
                result[f"{column}_per_100km_{window}"] = _rate(count, km, 100.0)
                result[f"{column}_per_hour_{window}"] = _rate(count, hours)
            near_miss = _number(part.event_11804_count).where(observed_event)
            hit_days = [day for day, count in near_miss.items() if pd.notna(count) and count > 0]
            result[f"near_miss_days_{window}"] = len(hit_days) if observed_event.any() else np.nan
            result[f"near_miss_recency_{window}"] = (anchor - hit_days[-1]).days if hit_days else np.nan
        recent = frame.iloc[-7:]
        previous = frame.iloc[-14:-7]
        result["distance_change_7d"] = _sum_or_nan(recent.distance_km) - _sum_or_nan(previous.distance_km)
        result["hours_change_7d"] = _sum_or_nan(recent.drive_hours) - _sum_or_nan(previous.drive_hours)
        rows.append(result)
    features = pd.DataFrame(rows)
    if features.empty or features.duplicated(["gpsno", "anchor_date"]).any():
        raise ValueError("特征表为空或有重复样本")
    for name in features.columns.difference(["gpsno", "anchor_date"]):
        features[name] = pd.to_numeric(features[name], errors="raise").replace([np.inf, -np.inf], np.nan)
    return features


def feature_columns(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    full = [column for column in features.columns if column not in ("gpsno", "anchor_date")]
    fallback = [
        column for column in full
        if not (column.startswith("event_") or column.startswith("near_miss_"))
    ]
    if not full or not fallback:
        raise ValueError("特征组为空")
    return full, fallback
