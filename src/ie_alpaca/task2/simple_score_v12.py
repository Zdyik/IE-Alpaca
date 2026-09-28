"""Minimal human-readable V12 scorecard: events -> dimensions -> safety score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ie_alpaca.features.landmark_v4 import EVENT_CODES, EVENT_NAMES
from ie_alpaca.task2.safety_score_v12 import DailyArrays


DIMENSION_CODES = {
    "history": (11803, 11804),
    "collision": (30000, 30005, 60292, 60294),
    "attention": (41001, 41002, 41003, 41004, 41005, 41009, 41023, 41029),
    "speed": (11401, 11402, 11403, 11405, 11406),
    "control": (30002, 30003, 30017),
}
DIMENSION_LABELS = {
    "history": "事故与未遂事故",
    "collision": "碰撞与周边冲突",
    "attention": "疲劳与注意力",
    "speed": "超速行为",
    "control": "车辆控制稳定性",
    "trend": "风险持续与恶化",
}


def _columnwise_percentile(values: np.ndarray, reference_mask: np.ndarray,
                           zero_is_no_risk: bool) -> np.ndarray:
    values = np.asarray(values, float)
    one_dimensional = values.ndim == 1
    matrix = values[:, None] if one_dimensional else values
    output = np.zeros_like(matrix, float)
    for column in range(matrix.shape[1]):
        current = matrix[:, column]
        if zero_is_no_risk:
            ref = np.sort(current[reference_mask & (current > 0)])
            selected = current > 0
        else:
            ref = np.sort(current[reference_mask])
            selected = np.isfinite(current)
        if not len(ref):
            continue
        left = np.searchsorted(ref, current[selected], side="left")
        right = np.searchsorted(ref, current[selected], side="right")
        output[selected, column] = (left + right) / (2.0 * len(ref))
    return output[:, 0] if one_dimensional else output


def event_risk_values(arrays: DailyArrays, anchor: int, reference_mask: np.ndarray,
                      mix: dict[str, float]) -> np.ndarray:
    if not 14 <= anchor <= 60:
        raise ValueError("simple score anchor must be between 14 and 60")
    if abs(sum(mix.values()) - 1.0) > 1e-8:
        raise ValueError("event-risk mixture must sum to one")
    counts = arrays.counts[:, :anchor]
    total = counts.sum(axis=1)
    distance = np.nan_to_num(arrays.distance[:, :anchor], nan=0.0).clip(min=0).sum(axis=1)
    hours = np.nan_to_num(arrays.hours[:, :anchor], nan=0.0).clip(min=0).sum(axis=1)
    rate_km = 1000.0 * total / (distance[:, None] + 100.0)
    rate_hour = 100.0 * total / (hours[:, None] + 1.0)
    active_share = (counts > 0).mean(axis=1)
    recency = np.zeros_like(total, float)
    for vehicle in range(len(total)):
        for event in range(total.shape[1]):
            occurred = np.flatnonzero(counts[vehicle, :, event] > 0)
            if len(occurred):
                recency[vehicle, event] = np.exp(-(anchor - 1 - occurred[-1]) / 14.0)
    return (
        mix["rate_per_1000km"] * _columnwise_percentile(np.log1p(rate_km), reference_mask, True)
        + mix["rate_per_100hours"] * _columnwise_percentile(np.log1p(rate_hour), reference_mask, True)
        + mix["active_day_share"] * _columnwise_percentile(active_share, reference_mask, True)
        + mix["recency"] * recency
    )


def learned_event_weights(sensitivity: pd.DataFrame, uniform_share: float) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if not 0 <= uniform_share <= 1:
        raise ValueError("uniform share must be in [0,1]")
    table = sensitivity.copy()
    table["event_code"] = table.event_code.astype(int)
    indexed = table.set_index("event_code")
    rows, weights = [], {}
    for dimension, codes in DIMENSION_CODES.items():
        importance = indexed.reindex(codes).mean_abs_probability_delta.fillna(0).to_numpy(float)
        learned = importance / importance.sum() if importance.sum() else np.full(len(codes), 1 / len(codes))
        value = uniform_share * np.full(len(codes), 1 / len(codes)) + (1 - uniform_share) * learned
        value /= value.sum()
        weights[dimension] = value
        for code, weight, raw in zip(codes, value, importance):
            rows.append({"dimension": dimension, "dimension_name": DIMENSION_LABELS[dimension],
                         "event_code": int(code), "event_name": EVENT_NAMES[code],
                         "v12_mean_abs_probability_delta": float(raw),
                         "event_weight_within_dimension": float(weight)})
    return weights, pd.DataFrame(rows)


def _trend_risk(arrays: DailyArrays, anchor: int, reference_mask: np.ndarray) -> np.ndarray:
    positions = {code: EVENT_CODES.index(code) for code in EVENT_CODES}
    selected = [positions[code] for codes in DIMENSION_CODES.values() for code in codes]
    any_event = (arrays.counts[:, :anchor, selected].sum(axis=2) > 0).astype(float)
    recent_days = min(14, max(7, anchor // 3))
    recent = any_event[:, -recent_days:].mean(axis=1)
    previous = any_event[:, :-recent_days].mean(axis=1) if anchor > recent_days else np.zeros(len(any_event))
    increase = np.maximum(recent - previous, 0.0)
    streak = []
    for row in any_event:
        best = current = 0
        for value in row:
            current = current + 1 if value else 0
            best = max(best, current)
        streak.append(best)
    return (
        .50 * _columnwise_percentile(recent, reference_mask, True)
        + .30 * _columnwise_percentile(increase, reference_mask, True)
        + .20 * _columnwise_percentile(np.asarray(streak, float), reference_mask, True)
    )


def build_simple_scorecards(arrays: DailyArrays, anchor: int, development_ids: set[str],
                            event_feed: np.ndarray, sensitivity: pd.DataFrame,
                            config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = list(arrays.gpsno)
    reference_mask = np.asarray([gpsno in development_ids for gpsno in ids])
    if reference_mask.sum() != len(development_ids):
        raise ValueError("development reference IDs are incomplete")
    points = {str(k): float(v) for k, v in config["dimension_points"].items()}
    if set(points) != {*DIMENSION_CODES, "trend"} or abs(sum(points.values()) - 100) > 1e-8:
        raise ValueError("six simple-score dimensions must sum to 100 points")
    event_weights, weight_table = learned_event_weights(
        sensitivity, float(config["event_weight_uniform_share"]))
    event_values = event_risk_values(arrays, anchor, reference_mask, config["event_risk_mix"])
    positions = {code: EVENT_CODES.index(code) for code in EVENT_CODES}
    dimension_risk: dict[str, np.ndarray] = {}
    event_contributions: list[list[tuple[float, int]]] = [[] for _ in ids]
    for dimension, codes in DIMENSION_CODES.items():
        selected = [positions[code] for code in codes]
        risk = event_values[:, selected] @ event_weights[dimension]
        if dimension == "control":
            imu = np.abs(arrays.imu[:, :anchor])
            observed = np.isfinite(imu)
            logged = np.where(observed, np.log1p(imu), 0.0)
            raw_imu = logged.sum(axis=(1, 2)) / observed.sum(axis=(1, 2)).clip(min=1)
            imu_risk = _columnwise_percentile(raw_imu, reference_mask, False)
            event_share = float(config["control_event_share"])
            risk = event_share * risk + float(config["control_imu_share"]) * imu_risk
        dimension_risk[dimension] = risk
        event_multiplier = float(config["control_event_share"]) if dimension == "control" else 1.0
        for event_column, (code, weight) in enumerate(zip(codes, event_weights[dimension])):
            contribution = points[dimension] * event_multiplier * weight * event_values[:, selected[event_column]]
            for vehicle, value in enumerate(contribution):
                if value > 0:
                    event_contributions[vehicle].append((float(value), int(code)))
    dimension_risk["trend"] = _trend_risk(arrays, anchor, reference_mask)

    trajectory_coverage = arrays.trajectory_observed[:, :anchor].mean(axis=1)
    imu_coverage = arrays.imu_observed[:, :anchor].mean(axis=1)
    evidence = (.50 * np.asarray(event_feed, float) + .30 * trajectory_coverage + .20 * imu_coverage)
    confidence = np.where(evidence >= .80, "高", np.where(evidence >= .50, "中", "低"))
    output = pd.DataFrame({"gpsno": ids, "anchor_day": anchor,
                           "evidence_confidence": evidence, "confidence_level": confidence})
    low_evidence = output.confidence_level.eq("低").to_numpy()
    for dimension in points:
        output[f"risk_{dimension}"] = dimension_risk[dimension]
        # 低证据车辆使用开发集该维度的典型风险，避免把设备缺测误判为安全。
        # 各维度扣分仍由同一条公开公式计算，因此总分能够逐项复核。
        fallback_risk = float(np.median(dimension_risk[dimension][reference_mask]))
        output.loc[low_evidence, f"risk_{dimension}"] = fallback_risk
        output[f"deduction_{dimension}"] = points[dimension] * dimension_risk[dimension]
        output.loc[low_evidence, f"deduction_{dimension}"] = points[dimension] * fallback_risk
    deduction_columns = [f"deduction_{name}" for name in points]
    output["total_deduction"] = output[deduction_columns].sum(axis=1)
    output["safety_score"] = (100 - output.total_deduction).clip(0, 100)

    thresholds = config["grade_thresholds"]
    grade, level, action = [], [], []
    for score, confidence_level in zip(output.safety_score, output.confidence_level):
        if confidence_level == "低":
            values = ("U", "证据不足", "检查设备和数据链路，补齐证据前暂缓奖惩")
        elif score >= thresholds["A"]:
            values = ("A", "低风险", "正常月度复盘，可进入安全驾驶奖励候选")
        elif score >= thresholds["B"]:
            values = ("B", "较低风险", "每周反馈主要扣分项，保持常规管理")
        elif score >= thresholds["C"]:
            values = ("C", "中风险", "安排针对性培训，14天后重新评价")
        elif score >= thresholds["D"]:
            values = ("D", "高风险", "24小时内人工复核，专项培训并加强监控")
        else:
            values = ("E", "极高风险", "立即人工核查，必要时暂停高风险任务")
        grade.append(values[0]); level.append(values[1]); action.append(values[2])
    output["grade"], output["risk_level"], output["management_action"] = grade, level, action
    output["top_risk_dimension"] = [
        DIMENSION_LABELS[max(points, key=lambda name: float(row[f"deduction_{name}"]))]
        for _, row in output.iterrows()
    ]
    top_event_strings = []
    for contributions in event_contributions:
        contributions.sort(reverse=True)
        top_event_strings.append("；".join(
            f"{code}{EVENT_NAMES[code]}({value:.2f}分)" for value, code in contributions[:3]
        ) or "无明显风险事件")
    output["top_risk_events"] = top_event_strings
    return output, weight_table
