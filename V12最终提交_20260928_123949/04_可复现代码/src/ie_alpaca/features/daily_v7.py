"""Day-level V7 inputs; event absence is provisional, not a feed-online flag."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ie_alpaca.data.task_one import FIRST_DAY
from ie_alpaca.features.landmark_v4 import EVENT_CODES


NUMERIC_COLUMNS = (
    "distance_km", "drive_hours", "night_distance_km", "trip_starts",
    "speed_p50", "speed_p90", "speed_change_p90", "turn_rate_p90",
    "imu_valid_windows", "imu_accel_peak_p99", "imu_accel_variability_p90",
    "imu_gyro_peak", "distance_invalid_rows", "long_gap_rows",
)
QUALITY_COLUMNS = (
    "trajectory_recorded_today", "imu_recorded_today", "observed_stationary_day",
)
EVENT_COLUMNS = tuple(
    name for code in EVENT_CODES
    for name in (f"event_{code}_count", f"event_{code}_episodes", f"event_{code}_rate100_smooth")
)
TOKEN_COLUMNS = (*EVENT_COLUMNS, *NUMERIC_COLUMNS, *QUALITY_COLUMNS,
                 *(f"{name}_missing" for name in NUMERIC_COLUMNS))


def build_day_table(daily: pd.DataFrame, vehicle_ids: list[str]) -> pd.DataFrame:
    """Use only existing daily fields and a stable 1..60 calendar grid."""
    required = {"gpsno", "day", *NUMERIC_COLUMNS, *QUALITY_COLUMNS}
    required.update(f"event_{code}_{kind}" for code in EVENT_CODES for kind in ("count", "episodes"))
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"V7 日表缺少列：{missing}")
    if len(vehicle_ids) != len(set(vehicle_ids)):
        raise ValueError("车辆名单重复")
    frame = daily[daily.gpsno.isin(vehicle_ids)].sort_values(["gpsno", "day"]).copy()
    if len(frame) != len(vehicle_ids) * 60 or frame.duplicated(["gpsno", "day"]).any():
        raise ValueError("V7 必须有每车 60 个唯一日历日")
    expected = [FIRST_DAY + timedelta(days=day) for day in range(60)]
    for gpsno, part in frame.groupby("gpsno", sort=False):
        if part.day.tolist() != expected:
            raise ValueError(f"车辆 {gpsno} 日索引不连续")
    frame["day_index"] = frame.groupby("gpsno", sort=False).cumcount() + 1
    km = pd.to_numeric(frame.distance_km, errors="coerce").to_numpy(dtype=float)
    valid_km = np.isfinite(km) & (km >= 0)
    for code in EVENT_CODES:
        counts_name = f"event_{code}_count"
        episodes_name = f"event_{code}_episodes"
        counts = pd.to_numeric(frame[counts_name], errors="coerce").fillna(0).to_numpy(dtype=float)
        episodes = pd.to_numeric(frame[episodes_name], errors="coerce").fillna(0).to_numpy(dtype=float)
        if not np.isfinite(counts).all() or not np.isfinite(episodes).all() or (counts < 0).any() or (episodes < 0).any():
            raise ValueError(f"事件 {code} 次数或段数无效")
        frame[counts_name] = counts
        frame[episodes_name] = episodes
        frame[f"event_{code}_rate100_smooth"] = np.divide(
            100 * counts, km + 100, out=np.zeros_like(counts), where=valid_km,
        )
    return frame[["gpsno", "day", "day_index", *EVENT_COLUMNS, *NUMERIC_COLUMNS, *QUALITY_COLUMNS]]


class DayScaler:
    """Fit each unique training vehicle/day once; never see outer validation rows."""

    def fit(self, day_table: pd.DataFrame, ids: set[str]) -> "DayScaler":
        train = day_table[day_table.gpsno.isin(ids) & day_table.day_index.le(53)]
        if len(train) != len(ids) * 53 or not ids:
            raise ValueError("缩放器必须使用完整训练车辆的第 1..53 天")
        raw = self._raw(train)
        with np.errstate(all="ignore"):
            self.low = np.nanpercentile(raw, 1, axis=0)
            self.high = np.nanpercentile(raw, 99, axis=0)
        self.low = np.nan_to_num(self.low, nan=0.0)
        self.high = np.nan_to_num(self.high, nan=0.0)
        clipped = np.clip(raw, self.low, self.high)
        self.median = np.nan_to_num(np.nanmedian(clipped, axis=0), nan=0.0)
        filled = np.where(np.isfinite(clipped), clipped, self.median)
        std = filled.std(axis=0)
        self.scale = np.where(std > 1e-6, std, 1.0)
        self.fit_vehicle_count = len(ids)
        return self

    @staticmethod
    def _raw(frame: pd.DataFrame) -> np.ndarray:
        events = frame[list(EVENT_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(events).all() or (events < 0).any():
            raise ValueError("事件 token 必须为非负有限值")
        numeric = frame[list(NUMERIC_COLUMNS)].to_numpy(dtype=float)
        numeric[~np.isfinite(numeric)] = np.nan
        return np.concatenate((np.log1p(events), np.sign(numeric) * np.log1p(np.abs(numeric))), axis=1)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = np.clip(self._raw(frame), self.low, self.high)
        observed = np.isfinite(raw)
        filled = np.where(observed, raw, self.median)
        quality = frame[list(QUALITY_COLUMNS)].fillna(False).to_numpy(dtype=np.float32)
        numeric_missing = (~observed[:, len(EVENT_COLUMNS):]).astype(np.float32)
        token = np.concatenate(((filled - self.median) / self.scale, quality, numeric_missing), axis=1)
        if token.shape[1] != len(TOKEN_COLUMNS) or not np.isfinite(token).all():
            raise ValueError("V7 token 宽度或数值无效")
        return token.astype(np.float32)

    def describe(self) -> dict:
        return {"token_columns": list(TOKEN_COLUMNS), "event_columns": list(EVENT_COLUMNS),
                "numeric_columns": list(NUMERIC_COLUMNS), "quality_columns": list(QUALITY_COLUMNS),
                "low": self.low.tolist(), "high": self.high.tolist(),
                "median": self.median.tolist(), "scale": self.scale.tolist(),
                "fit_vehicle_count": self.fit_vehicle_count,
                "fit_policy": "unique training vehicle days 1..53"}
