"""Structured daily event-node inputs for V10 RiskChainNet."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ie_alpaca.features.daily_v7 import (
    NUMERIC_COLUMNS, QUALITY_COLUMNS, build_day_table as build_v7_day_table,
)
from ie_alpaca.features.landmark_v4 import EVENT_CODES


EVENT_FEATURE_NAMES = (
    "log_count", "log_episodes", "log_rate_100km", "log_rate_hour",
    "occurred", "km_observed", "hours_observed", "night_share",
)
CONTEXT_COLUMNS = (*NUMERIC_COLUMNS, *QUALITY_COLUMNS,
                   *(f"{name}_missing" for name in NUMERIC_COLUMNS))


def build_day_table(daily: pd.DataFrame, vehicle_ids: list[str]) -> pd.DataFrame:
    """Reuse the audited 60-day grid and retain only prefix-safe daily fields."""
    return build_v7_day_table(daily, vehicle_ids)


class NodeDayScaler:
    """Fit V10 event and context transforms on training vehicles through day 53."""

    def _event_raw(self, frame: pd.DataFrame) -> np.ndarray:
        counts = np.column_stack([
            pd.to_numeric(frame[f"event_{code}_count"], errors="coerce").fillna(0).to_numpy(float)
            for code in EVENT_CODES
        ])
        episodes = np.column_stack([
            pd.to_numeric(frame[f"event_{code}_episodes"], errors="coerce").fillna(0).to_numpy(float)
            for code in EVENT_CODES
        ])
        rate_km = np.column_stack([
            pd.to_numeric(frame[f"event_{code}_rate100_smooth"], errors="coerce").fillna(0).to_numpy(float)
            for code in EVENT_CODES
        ])
        km = pd.to_numeric(frame.distance_km, errors="coerce").to_numpy(float)
        hours = pd.to_numeric(frame.drive_hours, errors="coerce").to_numpy(float)
        night = pd.to_numeric(frame.night_distance_km, errors="coerce").to_numpy(float)
        km_ok = np.isfinite(km) & (km >= 0)
        hour_ok = np.isfinite(hours) & (hours >= 0)
        rate_hour = counts / (np.where(hour_ok, hours, 0)[:, None] + 1.0)
        night_share = np.divide(np.maximum(night, 0), np.maximum(km, 0),
                                out=np.zeros_like(night), where=km_ok & (km > 0))
        night_share = np.clip(night_share, 0, 1)
        features = np.stack((
            np.log1p(counts), np.log1p(episodes), np.log1p(rate_km), np.log1p(rate_hour),
            (counts > 0).astype(float), np.broadcast_to(km_ok[:, None], counts.shape),
            np.broadcast_to(hour_ok[:, None], counts.shape),
            np.broadcast_to(night_share[:, None], counts.shape),
        ), axis=-1)
        if not np.isfinite(features).all() or (features < 0).any():
            raise ValueError("V10 event nodes contain invalid values")
        return features

    @staticmethod
    def _context_raw(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numeric = frame[list(NUMERIC_COLUMNS)].to_numpy(float)
        numeric[~np.isfinite(numeric)] = np.nan
        transformed = np.sign(numeric) * np.log1p(np.abs(numeric))
        quality = frame[list(QUALITY_COLUMNS)].fillna(False).to_numpy(np.float32)
        return transformed, quality

    def fit(self, day_table: pd.DataFrame, ids: set[str]) -> "NodeDayScaler":
        train = day_table[day_table.gpsno.isin(ids) & day_table.day_index.le(53)]
        if not ids or len(train) != len(ids) * 53:
            raise ValueError("V10 scaler requires every training vehicle day 1..53")
        event = self._event_raw(train)
        self.event_scale = np.maximum(np.percentile(event[..., :4], 99, axis=0), .05)
        numeric, _ = self._context_raw(train)
        with np.errstate(all="ignore"):
            self.low = np.nan_to_num(np.nanpercentile(numeric, 1, axis=0), nan=0.0)
            self.high = np.nan_to_num(np.nanpercentile(numeric, 99, axis=0), nan=0.0)
        clipped = np.clip(numeric, self.low, self.high)
        self.median = np.nan_to_num(np.nanmedian(clipped, axis=0), nan=0.0)
        filled = np.where(np.isfinite(clipped), clipped, self.median)
        self.scale = np.where(filled.std(axis=0) > 1e-6, filled.std(axis=0), 1.0)
        self.fit_vehicle_count = len(ids)
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        event = self._event_raw(frame)
        event[..., :4] = np.minimum(event[..., :4], self.event_scale) / self.event_scale
        numeric, quality = self._context_raw(frame)
        numeric = np.clip(numeric, self.low, self.high)
        observed = np.isfinite(numeric)
        filled = np.where(observed, numeric, self.median)
        context = np.concatenate(((filled - self.median) / self.scale, quality,
                                  (~observed).astype(np.float32)), axis=1)
        if context.shape[1] != len(CONTEXT_COLUMNS) or not np.isfinite(context).all():
            raise ValueError("V10 context transform is invalid")
        return event.astype(np.float32), context.astype(np.float32)

    def describe(self) -> dict:
        return {"event_codes": list(EVENT_CODES), "event_feature_names": list(EVENT_FEATURE_NAMES),
                "context_columns": list(CONTEXT_COLUMNS), "event_scale": self.event_scale.tolist(),
                "context_low": self.low.tolist(), "context_high": self.high.tolist(),
                "context_median": self.median.tolist(), "context_scale": self.scale.tolist(),
                "fit_vehicle_count": self.fit_vehicle_count,
                "fit_policy": "unique outer/inner training vehicles, days 1..53"}

    @classmethod
    def from_description(cls, values: dict) -> "NodeDayScaler":
        """Restore the exact fold transform saved inside a V10 checkpoint."""
        if values.get("event_codes") != list(EVENT_CODES):
            raise ValueError("saved V10 scaler uses a different event order")
        if values.get("event_feature_names") != list(EVENT_FEATURE_NAMES):
            raise ValueError("saved V10 scaler uses different event features")
        if values.get("context_columns") != list(CONTEXT_COLUMNS):
            raise ValueError("saved V10 scaler uses different context columns")
        scaler = cls()
        scaler.event_scale = np.asarray(values["event_scale"], dtype=float)
        scaler.low = np.asarray(values["context_low"], dtype=float)
        scaler.high = np.asarray(values["context_high"], dtype=float)
        scaler.median = np.asarray(values["context_median"], dtype=float)
        scaler.scale = np.asarray(values["context_scale"], dtype=float)
        scaler.fit_vehicle_count = int(values["fit_vehicle_count"])
        return scaler


AUX_GROUPS = {
    "control": (30002, 30003, 30017, 41002, 41004, 41005, 41009),
    "proximal": (30000, 30005),
    "severe": (11803, 11804),
}


def future_group_targets(day_table: pd.DataFrame, vehicle_ids: list[str], horizon: int = 7) -> np.ndarray:
    """For day d, mark whether each downstream group occurs in d+1..d+horizon."""
    ordered = day_table[day_table.gpsno.isin(vehicle_ids)].sort_values(["gpsno", "day_index"])
    if len(ordered) != len(vehicle_ids) * 60:
        raise ValueError("V10 auxiliary targets require complete 60-day sequences")
    result = np.zeros((len(vehicle_ids), 60, len(AUX_GROUPS)), dtype=np.float32)
    for vehicle_index, gpsno in enumerate(vehicle_ids):
        frame = ordered[ordered.gpsno.eq(gpsno)]
        for group_index, codes in enumerate(AUX_GROUPS.values()):
            occurred = np.column_stack([
                pd.to_numeric(frame[f"event_{code}_episodes"], errors="coerce").fillna(0).to_numpy(float)
                for code in codes
            ]).sum(axis=1) > 0
            for day in range(60):
                result[vehicle_index, day, group_index] = occurred[day + 1:min(60, day + horizon + 1)].any()
    return result
