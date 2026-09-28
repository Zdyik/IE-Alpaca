"""Small, fixed history summaries for day-end landmark prediction."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ie_alpaca.data.task_one import FIRST_DAY


ANCHORS = (14, 20, 30, 40, 50, 53)
EVENT_FIELDS = (
    "event_11803_count", "event_11804_count", "event_11401_count",
    "event_11402_count", "event_11403_count", "event_30000_count",
    "event_30002_count", "event_30003_count", "event_total_episodes",
)


def _sum(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def _rate(n: float, d: float, scale: float = 1.0) -> float:
    return scale * n / d if d > 0 else np.nan


def build_landmarks(daily: pd.DataFrame, vehicle_ids: list[str], anchors=ANCHORS) -> pd.DataFrame:
    """Use only days 1..t. Event absence is provisional zero, never a coverage gate."""
    lookup = {str(key): part.sort_values("day") for key, part in daily.groupby("gpsno", sort=False)}
    rows = []
    for gpsno in vehicle_ids:
        vehicle = lookup.get(str(gpsno))
        if vehicle is None or len(vehicle) != 60:
            raise ValueError(f"车辆 {gpsno} 没有完整 60 天日索引")
        expected = [FIRST_DAY + timedelta(days=i) for i in range(60)]
        if vehicle.day.tolist() != expected:
            raise ValueError(f"车辆 {gpsno} 的日索引不连续")
        for t in anchors:
            if not 14 <= t <= 60:
                raise ValueError("锚点必须在第 14 至 60 天")
            history = vehicle.iloc[:t]
            recent, previous = history.iloc[-7:], history.iloc[-14:-7]
            out: dict[str, float | int | str] = {"gpsno": str(gpsno), "anchor_day": t}
            km, hours = _sum(history, "distance_km"), _sum(history, "drive_hours")
            out["exposure_km_per_day"] = km / t
            out["exposure_hours_per_day"] = hours / t
            out["exposure_driving_fraction"] = float((history.distance_km.fillna(0) > 0).mean())
            out["exposure_trajectory_fraction"] = float(history.trajectory_recorded_today.fillna(False).mean())
            out["exposure_imu_fraction"] = float(history.imu_recorded_today.fillna(False).mean())
            out["exposure_night_share"] = _rate(_sum(history, "night_distance_km"), km)
            out["exposure_km_recent7"] = _sum(recent, "distance_km") / 7
            out["exposure_hours_recent7"] = _sum(recent, "drive_hours") / 7
            out["exposure_km_delta7"] = (_sum(recent, "distance_km") - _sum(previous, "distance_km")) / 7
            out["exposure_hours_delta7"] = (_sum(recent, "drive_hours") - _sum(previous, "drive_hours")) / 7
            out["quality_invalid_distance_share"] = _rate(
                _sum(history, "distance_invalid_rows"),
                _sum(history, "distance_valid_rows") + _sum(history, "distance_invalid_rows"),
            )
            out["quality_long_gap_per_day"] = _sum(history, "long_gap_rows") / t
            out["imu_accel_peak_mean"] = _mean(history, "imu_accel_peak_p99")
            out["imu_gyro_peak_mean"] = _mean(history, "imu_gyro_peak")
            out["imu_accel_recent7"] = _mean(recent, "imu_accel_peak_p99")
            imu_recent, imu_previous = _mean(recent, "imu_accel_peak_p99"), _mean(previous, "imu_accel_peak_p99")
            out["imu_accel_delta7"] = imu_recent - imu_previous
            out["imu_valid_windows_per_day"] = _sum(history, "imu_valid_windows") / t
            for field in EVENT_FIELDS:
                total = _sum(history, field)
                short = _sum(recent, field)
                old = _sum(previous, field)
                out[f"{field}_per_day"] = total / t
                out[f"{field}_per_100km"] = _rate(total, km, 100)
                out[f"{field}_recent7_per_day"] = short / 7
                out[f"{field}_delta7_per_day"] = (short - old) / 7
            for code in ("11803", "11804"):
                values = pd.to_numeric(history[f"event_{code}_count"], errors="coerce").fillna(0)
                occurred = np.flatnonzero(values.to_numpy() > 0)
                out[f"event_{code}_recency_days"] = float(t - occurred[-1] - 1) if len(occurred) else np.nan
            rows.append(out)
    result = pd.DataFrame(rows)
    if result.duplicated(["gpsno", "anchor_day"]).any():
        raise ValueError("重复 landmark")
    return result


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    full = [c for c in frame.columns if c not in {"gpsno", "anchor_day"}]
    fallback = [c for c in full if not c.startswith("event_")]
    return full, fallback
