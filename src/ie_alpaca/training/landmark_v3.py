"""Compressed stationary next-event hazard likelihood and fold-only preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def make_outcomes(daily: pd.DataFrame, landmarks: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(key): part.sort_values("day") for key, part in daily.groupby("gpsno", sort=False)}
    rows = []
    for gpsno, t in landmarks[["gpsno", "anchor_day"]].itertuples(index=False, name=None):
        c = min(40, 60 - int(t))
        if c < 1:
            continue
        future = lookup[str(gpsno)].iloc[t:t + c]
        counts = (pd.to_numeric(future.event_11803_count, errors="coerce").fillna(0)
                  + pd.to_numeric(future.event_11804_count, errors="coerce").fillna(0))
        hits = np.flatnonzero(counts.to_numpy() > 0)
        j = int(hits[0] + 1) if len(hits) else None
        rows.append({"gpsno": str(gpsno), "anchor_day": int(t), "horizon_days": c,
                     "first_event_day": j, "label": int(j is not None)})
    return pd.DataFrame(rows)


def compress_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    """At most two rows per anchor, exactly matching daily survival log loss."""
    counts = outcomes.groupby("gpsno").size().to_dict()
    rows = []
    for row in outcomes.itertuples(index=False):
        unit = 1.0 / counts[row.gpsno]
        negative_days = row.first_event_day - 1 if pd.notna(row.first_event_day) else row.horizon_days
        if negative_days > 0:
            rows.append({"gpsno": row.gpsno, "anchor_day": row.anchor_day,
                         "y": 0, "weight": float(negative_days * unit)})
        if pd.notna(row.first_event_day):
            rows.append({"gpsno": row.gpsno, "anchor_day": row.anchor_day,
                         "y": 1, "weight": float(unit)})
    return pd.DataFrame(rows)


class HazardLogistic:
    def __init__(self, columns: list[str], *, c: float, seed: int):
        self.columns, self.c, self.seed = columns, c, seed
        self.imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.2, C=c,
            max_iter=5000, tol=1e-4, random_state=seed,
        )

    def _base(self, landmarks: pd.DataFrame) -> np.ndarray:
        x = landmarks[self.columns].to_numpy(dtype=float)
        x[~np.isfinite(x)] = np.nan
        return np.sign(x) * np.log1p(np.abs(x))

    def _transform(self, landmarks: pd.DataFrame) -> np.ndarray:
        x = np.clip(self._base(landmarks), self.low, self.high)
        return self.scaler.transform(self.imputer.transform(x))

    def fit(self, landmarks: pd.DataFrame, compressed: pd.DataFrame) -> "HazardLogistic":
        if landmarks.duplicated(["gpsno", "anchor_day"]).any():
            raise ValueError("训练特征含重复 landmark")
        raw = self._base(landmarks)
        self.low = np.nanpercentile(raw, 1, axis=0)
        self.high = np.nanpercentile(raw, 99, axis=0)
        self.low = np.where(np.isfinite(self.low), self.low, 0)
        self.high = np.where(np.isfinite(self.high), self.high, 0)
        train_x = self.imputer.fit_transform(np.clip(raw, self.low, self.high))
        self.scaler.fit(train_x)
        keyed = compressed.merge(landmarks[["gpsno", "anchor_day", *self.columns]],
                                 on=["gpsno", "anchor_day"], how="left", validate="many_to_one")
        if keyed[self.columns].isna().all(axis=1).any():
            raise ValueError("加权记录缺少训练特征")
        x = self._transform(keyed)
        self.model.fit(x, keyed.y.to_numpy(dtype=int), sample_weight=keyed.weight.to_numpy(dtype=float))
        return self

    def daily_hazard(self, landmarks: pd.DataFrame) -> np.ndarray:
        q = self.model.predict_proba(self._transform(landmarks))[:, 1]
        return np.clip(q, 1e-12, 1 - 1e-12)


def horizon_probability(q: np.ndarray, days: int | np.ndarray) -> np.ndarray:
    return -np.expm1(np.asarray(days) * np.log1p(-np.asarray(q)))
