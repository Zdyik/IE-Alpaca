"""All 24 event types at each day-end landmark, with V3 context features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ie_alpaca.features.landmark_v3 import ANCHORS, build_landmarks


EVENT_NAMES = {
    11401: "路口超速", 11402: "主路弯道超速", 11403: "危险路段超速报警",
    11405: "匝道超速", 11406: "高速长下坡超速", 30000: "前碰撞预警",
    30002: "左车道偏移", 30003: "右车道偏移", 30005: "车距过近",
    30017: "长时间压线", 41001: "疲劳（闭眼）", 41002: "打哈欠",
    41003: "注意力分散", 41004: "打电话", 41005: "抽烟",
    41006: "摄像头遮挡", 41009: "频繁低头", 41021: "摄像头角度扭转",
    41023: "看手机", 41029: "困倦", 60292: "后盲区报警",
    60294: "右盲区报警", 11803: "事故", 11804: "未遂事故",
}
EVENT_CODES = tuple(EVENT_NAMES)
QUALITY_CODES = (41006, 41021)
RISK_CODES = tuple(code for code in EVENT_CODES if code not in QUALITY_CODES)
GROUPS = {
    "speed": (11401, 11402, 11403, 11405, 11406),
    "proximity_lane": (30000, 30002, 30003, 30005, 30017),
    "fatigue_distraction": (41001, 41002, 41003, 41004, 41005, 41009, 41023, 41029),
    "blind_spot": (60292, 60294),
    "history_target": (11803, 11804),
    "device_quality": QUALITY_CODES,
}
EVENT_METRICS = (
    "episodes_per_day", "episodes_per_100km", "recent7_per_day",
    "previous7_per_day", "recency_proximity", "counts_per_day",
    "counts_per_100km", "counts_recent7_per_day", "counts_previous7_per_day",
)
CONTEXT_COLUMNS = (
    "exposure_km_per_day", "exposure_hours_per_day", "exposure_driving_fraction",
    "exposure_trajectory_fraction", "exposure_imu_fraction", "exposure_night_share",
    "exposure_km_recent7", "exposure_hours_recent7", "exposure_km_delta7",
    "exposure_hours_delta7", "quality_invalid_distance_share",
    "quality_long_gap_per_day", "imu_accel_peak_mean", "imu_gyro_peak_mean",
    "imu_accel_recent7", "imu_accel_delta7", "imu_valid_windows_per_day",
)


def event_columns() -> list[str]:
    return [f"event_{code}_{metric}" for code in EVENT_CODES for metric in EVENT_METRICS]


def context_columns() -> list[str]:
    return list(CONTEXT_COLUMNS)


def build_features(daily: pd.DataFrame, vehicle_ids: list[str], anchors=ANCHORS) -> pd.DataFrame:
    """Only days <= t may change the returned row for landmark t."""
    missing = [f"event_{code}_{field}" for code in EVENT_CODES for field in ("episodes", "count")
               if f"event_{code}_{field}" not in daily]
    if missing:
        raise ValueError(f"日表缺少事件分型列：{missing}")
    context = build_landmarks(daily, vehicle_ids, anchors=anchors)
    keep = ["gpsno", "anchor_day", *context_columns()]
    context = context[keep]
    lookup = {str(gpsno): part.sort_values("day") for gpsno, part in daily.groupby("gpsno", sort=False)}
    rows = []
    for gpsno in vehicle_ids:
        vehicle = lookup[str(gpsno)]
        episodes = np.column_stack([
            pd.to_numeric(vehicle[f"event_{code}_episodes"], errors="coerce").fillna(0).to_numpy(dtype=float)
            for code in EVENT_CODES
        ])
        counts = np.column_stack([
            pd.to_numeric(vehicle[f"event_{code}_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
            for code in EVENT_CODES
        ])
        if np.any(episodes < 0) or np.any(counts < 0):
            raise ValueError("事件段数和原始次数不能为负")
        cumulative = np.vstack((np.zeros(len(EVENT_CODES)), np.cumsum(episodes, axis=0)))
        count_cumulative = np.vstack((np.zeros(len(EVENT_CODES)), np.cumsum(counts, axis=0)))
        km = pd.to_numeric(vehicle.distance_km, errors="coerce").fillna(0).to_numpy(dtype=float)
        km_prefix = np.r_[0.0, np.cumsum(np.maximum(km, 0))]
        for t in anchors:
            total = cumulative[t]
            recent = cumulative[t] - cumulative[t - 7]
            previous = cumulative[t - 7] - cumulative[t - 14]
            total_count = count_cumulative[t]
            recent_count = count_cumulative[t] - count_cumulative[t - 7]
            previous_count = count_cumulative[t - 7] - count_cumulative[t - 14]
            km_total = km_prefix[t]
            values = np.column_stack((
                total / t,
                100 * total / (km_total + 100),
                recent / 7,
                previous / 7,
                np.zeros(len(EVENT_CODES)),
                total_count / t,
                np.divide(100 * total_count, km_total, out=np.zeros_like(total_count), where=km_total > 0),
                recent_count / 7,
                previous_count / 7,
            ))
            for index in range(len(EVENT_CODES)):
                occurred = np.flatnonzero(episodes[:t, index] > 0)
                if len(occurred):
                    values[index, 4] = np.exp(-(t - 1 - occurred[-1]) / 14)
            row: dict[str, str | int | float] = {"gpsno": str(gpsno), "anchor_day": int(t)}
            row.update(dict(zip(event_columns(), values.ravel().tolist())))
            rows.append(row)
    events = pd.DataFrame(rows)
    result = context.merge(events, on=["gpsno", "anchor_day"], validate="one_to_one")
    if len(result) != len(vehicle_ids) * len(anchors) or result.duplicated(["gpsno", "anchor_day"]).any():
        raise ValueError("V4 landmark 特征缺失或重复")
    return result
