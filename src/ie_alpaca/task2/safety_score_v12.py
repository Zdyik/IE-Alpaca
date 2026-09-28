"""Transparent driver scorecard built around the selected V12 State-MAE risk model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ie_alpaca.data.task_one import FIRST_DAY
from ie_alpaca.features.daily_v12 import STATE_EVENT_CODES
from ie_alpaca.features.landmark_v4 import EVENT_CODES, EVENT_NAMES


SCORE_STATES = ("model", "upstream", "control", "proximal", "history", "quality")
DEFAULT_MAX_POINTS = {
    "model": 50.0,
    "upstream": 14.0,
    "control": 10.0,
    "proximal": 10.0,
    "history": 12.0,
    "quality": 4.0,
}
STATE_LABELS = {
    "model": "V12未来风险",
    "upstream": "危险驾驶行为",
    "control": "车辆控制稳定性",
    "proximal": "近端险情",
    "history": "事故与未遂历史",
    "quality": "管理配合度",
}


@dataclass(frozen=True)
class DailyArrays:
    gpsno: tuple[str, ...]
    counts: np.ndarray
    episodes: np.ndarray
    distance: np.ndarray
    hours: np.ndarray
    imu: np.ndarray
    trajectory_observed: np.ndarray
    imu_observed: np.ndarray


def prepare_daily_arrays(day_table: pd.DataFrame, vehicle_ids: list[str]) -> DailyArrays:
    ordered = day_table[day_table.gpsno.isin(vehicle_ids)].sort_values(["gpsno", "day_index"])
    if len(ordered) != len(vehicle_ids) * 60 or ordered.gpsno.astype(str).unique().tolist() != vehicle_ids:
        raise ValueError("task2 requires 60 ordered days for every vehicle")

    def matrix(suffix: str) -> np.ndarray:
        values = np.column_stack([
            pd.to_numeric(ordered[f"event_{code}_{suffix}"], errors="coerce").fillna(0).to_numpy(float)
            for code in EVENT_CODES
        ])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"invalid event {suffix} values")
        return values.reshape(len(vehicle_ids), 60, len(EVENT_CODES))

    numeric = lambda name: pd.to_numeric(ordered[name], errors="coerce").to_numpy(float).reshape(len(vehicle_ids), 60)
    imu = np.stack([
        numeric("imu_accel_peak_p99"), numeric("imu_accel_variability_p90"), numeric("imu_gyro_peak")
    ], axis=-1)
    return DailyArrays(
        gpsno=tuple(vehicle_ids),
        counts=matrix("count"),
        episodes=matrix("episodes"),
        distance=numeric("distance_km"),
        hours=numeric("drive_hours"),
        imu=imu,
        trajectory_observed=ordered.trajectory_recorded_today.fillna(False).to_numpy(bool).reshape(len(vehicle_ids), 60),
        imu_observed=ordered.imu_recorded_today.fillna(False).to_numpy(bool).reshape(len(vehicle_ids), 60),
    )


def event_burden(arrays: DailyArrays, anchor: int) -> np.ndarray:
    """Exposure-normalized, trend-aware event burden for each vehicle and event."""
    if not 14 <= anchor <= 60:
        raise ValueError("task2 anchor must be in [14,60]")
    counts = arrays.counts[:, :anchor]
    total = counts.sum(axis=1)
    distance = np.nan_to_num(arrays.distance[:, :anchor], nan=0.0).clip(min=0).sum(axis=1)
    hours = np.nan_to_num(arrays.hours[:, :anchor], nan=0.0).clip(min=0).sum(axis=1)
    active_share = (counts > 0).mean(axis=1)
    recent_days = min(14, anchor)
    recent = counts[:, anchor - recent_days:anchor].sum(axis=1) / recent_days
    past_days = anchor - recent_days
    past = (counts[:, :past_days].sum(axis=1) / past_days if past_days else np.zeros_like(recent))
    positive_trend = np.maximum(recent - past, 0.0) / (past + 0.10)
    positive_trend = np.minimum(positive_trend, 5.0)
    rate_km = 1000.0 * total / (distance[:, None] + 100.0)
    rate_hour = 100.0 * total / (hours[:, None] + 1.0)
    return (np.log1p(rate_km) + 0.40 * np.log1p(rate_hour) +
            0.80 * active_share + 0.25 * np.log1p(positive_trend))


def normalize_event_weights(event_weights: pd.DataFrame, uniform_share: float = .50) -> dict[str, np.ndarray]:
    """Shrink learned sensitivity weights toward uniform within each semantic state."""
    required = {"event_code", "state", "mean_abs_probability_delta"}
    if not required.issubset(event_weights):
        raise ValueError(f"event weight table missing {sorted(required - set(event_weights))}")
    table = event_weights.copy()
    table["event_code"] = table.event_code.astype(int)
    result = {}
    for state, codes in STATE_EVENT_CODES.items():
        values = table.set_index("event_code").reindex(codes).mean_abs_probability_delta.fillna(0).to_numpy(float)
        data = values / values.sum() if values.sum() > 0 else np.full(len(codes), 1.0 / len(codes))
        result[state] = uniform_share * np.full(len(codes), 1.0 / len(codes)) + (1 - uniform_share) * data
        result[state] /= result[state].sum()
    return result


def raw_state_signals(arrays: DailyArrays, anchor: int,
                      learned_weights: dict[str, np.ndarray]) -> tuple[pd.DataFrame, np.ndarray]:
    burden = event_burden(arrays, anchor)
    positions = {code: EVENT_CODES.index(code) for code in EVENT_CODES}
    signals: dict[str, np.ndarray] = {}
    for state in ("upstream", "control", "proximal", "history", "quality"):
        codes = STATE_EVENT_CODES[state]
        weights = learned_weights[state]
        signals[f"{state}_event"] = burden[:, [positions[code] for code in codes]] @ weights

    imu = np.abs(arrays.imu[:, :anchor])
    observed = np.isfinite(imu)
    logged = np.where(observed, np.log1p(imu), 0.0)
    denom = observed.sum(axis=(1, 2)).clip(min=1)
    signals["control_imu"] = logged.sum(axis=(1, 2)) / denom
    signals["trajectory_coverage"] = arrays.trajectory_observed[:, :anchor].mean(axis=1)
    signals["imu_coverage"] = arrays.imu_observed[:, :anchor].mean(axis=1)
    return pd.DataFrame(signals, index=list(arrays.gpsno)), burden


def build_reference(signals: pd.DataFrame, probabilities: np.ndarray,
                    development_ids: set[str]) -> dict[str, list[float]]:
    selected = signals.loc[signals.index.isin(development_ids)]
    if len(selected) != len(development_ids):
        raise ValueError("reference signals do not cover every development vehicle")
    reference = {name: np.sort(selected[name].to_numpy(float)).tolist()
                 for name in ("upstream_event", "control_event", "control_imu",
                              "proximal_event", "history_event", "quality_event")}
    positions = [signals.index.get_loc(gpsno) for gpsno in selected.index]
    reference["model_probability"] = np.sort(np.asarray(probabilities, float)[positions]).tolist()
    return reference


def empirical_percentile(values: np.ndarray, reference: list[float]) -> np.ndarray:
    ref = np.asarray(reference, float)
    if not len(ref) or not np.isfinite(ref).all():
        raise ValueError("invalid task2 score reference")
    return np.searchsorted(ref, np.asarray(values, float), side="right") / len(ref)


def risk_percentiles(signals: pd.DataFrame, probabilities: np.ndarray,
                     reference: dict[str, list[float]], event_feed: np.ndarray,
                     route: np.ndarray) -> pd.DataFrame:
    result = pd.DataFrame(index=signals.index)
    result["model"] = empirical_percentile(probabilities, reference["model_probability"])
    result["upstream"] = empirical_percentile(signals.upstream_event, reference["upstream_event"])
    control_event = empirical_percentile(signals.control_event, reference["control_event"])
    control_imu = empirical_percentile(signals.control_imu, reference["control_imu"])
    result["control"] = .75 * control_event + .25 * control_imu
    result["proximal"] = empirical_percentile(signals.proximal_event, reference["proximal_event"])
    result["history"] = empirical_percentile(signals.history_event, reference["history_event"])
    result["quality"] = empirical_percentile(signals.quality_event, reference["quality_event"])

    unavailable = ~np.asarray(event_feed, bool)
    for name in ("upstream", "proximal", "history", "quality"):
        result.loc[unavailable, name] = .50
    result.loc[unavailable, "control"] = .75 * .50 + .25 * control_imu[unavailable]
    no_evidence = np.asarray(route) == "prior_no_observed_behavior"
    result.loc[no_evidence, list(SCORE_STATES)] = .50
    return result.clip(0, 1)


def score_from_risks(risks: pd.DataFrame, max_points: dict[str, float] | None = None) -> pd.DataFrame:
    points = DEFAULT_MAX_POINTS if max_points is None else max_points
    if set(points) != set(SCORE_STATES) or abs(sum(points.values()) - 100) > 1e-8:
        raise ValueError("task2 maximum deductions must cover six states and sum to 100")
    missing = set(SCORE_STATES) - set(risks)
    if missing:
        raise ValueError(f"task2 risk frame missing {sorted(missing)}")
    output = pd.DataFrame(index=risks.index)
    for name in SCORE_STATES:
        output[f"{name}_deduction"] = float(points[name]) * risks[name].to_numpy(float) ** 2
    output["total_deduction"] = output.sum(axis=1)
    output["safety_score"] = (100.0 - output.total_deduction).clip(0, 100)
    return output


def grade(score: float, confidence_level: str) -> tuple[str, str, str]:
    if confidence_level == "低":
        return "U", "证据不足", "检查事件源、轨迹和IMU设备；补齐数据前暂缓奖惩"
    if score >= 85:
        return "A", "低风险", "保持常规月度复盘，可纳入安全激励"
    if score >= 75:
        return "B", "较低风险", "每周反馈主要扣分项，保持常规管理"
    if score >= 65:
        return "C", "中风险", "安排针对性培训，两周后复评"
    if score >= 50:
        return "D", "高风险", "24小时内人工复核，进行专项培训和重点监控"
    return "E", "极高风险", "立即人工核查，暂停高风险任务并制定整改计划"


def confidence(event_feed: np.ndarray, signals: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    value = (.60 * np.asarray(event_feed, float) + .30 * signals.trajectory_coverage.to_numpy(float) +
             .10 * signals.imu_coverage.to_numpy(float))
    level = np.where(value >= .80, "高", np.where(value >= .50, "中", "低"))
    return value.clip(0, 1), level


def route_for_anchor(event_feed: np.ndarray, signals: pd.DataFrame) -> np.ndarray:
    behavior = (signals.trajectory_coverage.to_numpy() > 0) | (signals.imu_coverage.to_numpy() > 0)
    return np.where(event_feed, "full", np.where(behavior, "context_only_no_matched_event",
                                                  "prior_no_observed_behavior"))


def top_dimension(row: pd.Series) -> str:
    name = max(SCORE_STATES[1:], key=lambda item: float(row[f"{item}_deduction"]))
    return STATE_LABELS[name]


def top_events_for_vehicle(burden: np.ndarray, learned_weights: dict[str, np.ndarray], limit: int = 3) -> str:
    positions = {code: EVENT_CODES.index(code) for code in EVENT_CODES}
    scored = []
    for state in ("upstream", "control", "proximal", "history", "quality"):
        for code, weight in zip(STATE_EVENT_CODES[state], learned_weights[state]):
            value = float(burden[positions[code]] * weight)
            if value > 0:
                scored.append((value, code))
    scored.sort(reverse=True)
    return "；".join(f"{code}{EVENT_NAMES[code]}" for _, code in scored[:limit]) or "无明显行为事件"


def scorecards(signals: pd.DataFrame, burden: np.ndarray, probabilities: np.ndarray,
               reference: dict[str, list[float]], event_feed: np.ndarray,
               learned_weights: dict[str, np.ndarray], max_points: dict[str, float],
               anchor: int, fallback_probability: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    route = route_for_anchor(event_feed, signals)
    probabilities = np.asarray(probabilities, float).copy()
    probabilities[route == "prior_no_observed_behavior"] = float(fallback_probability)
    risks = risk_percentiles(signals, probabilities, reference, event_feed, route)
    deductions = score_from_risks(risks, max_points)
    evidence, evidence_level = confidence(event_feed, signals)
    output = pd.concat((risks.add_prefix("risk_percentile_"), deductions), axis=1)
    output.insert(0, "gpsno", signals.index.astype(str))
    output["anchor_day"] = anchor
    output["as_of_date"] = date.isoformat(FIRST_DAY + timedelta(days=anchor - 1))
    output["v12_risk_probability"] = probabilities
    output["evidence_confidence"] = evidence
    output["confidence_level"] = evidence_level
    output["route"] = route
    labels = [grade(score, level) for score, level in zip(output.safety_score, evidence_level)]
    output["grade"] = [item[0] for item in labels]
    output["risk_level"] = [item[1] for item in labels]
    output["management_action"] = [item[2] for item in labels]
    output["top_risk_dimension"] = output.apply(top_dimension, axis=1)
    output["top_risk_events"] = [top_events_for_vehicle(values, learned_weights) for values in burden]
    output["safety_rank"] = output.safety_score.rank(method="min", ascending=False).astype(int)
    output["risk_rank"] = output.safety_score.rank(method="min", ascending=True).astype(int)
    output["fleet_safety_percentile"] = output.safety_score.rank(pct=True, method="average")
    return output.sort_values("gpsno").reset_index(drop=True), risks

