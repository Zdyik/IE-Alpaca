"""六组特征的构建（与数据源解耦，输入是日粒度聚合表）。

## 架构：为什么先做日粒度聚合

数据里最重的两类是轨迹（数十 GB、轨迹点级）与 IMU（更高频）。而特征需要按
**任意时间窗口**计算（多起点增广要求 `[1..t]` 这种可变长度），如果每次都重扫
原始数据，代价无法接受。因此采用两级架构：

1. :mod:`.daily` 只扫一遍原始数据，产出 ``(gpsno, date)`` 粒度的聚合表；
2. 本模块在这些小表上做窗口切片与聚合，任意窗口都很便宜。

这也让「保险箱」与「增广样本」共用同一份聚合表，口径天然一致。

## 日粒度表的 schema（本模块的输入契约）

``daily_exposure``（来自轨迹）::

    gpsno | date | km | hours | night_km | highway_km | speed_sum | speed_max | n_points

``daily_events``（来自风险事件）::

    gpsno | date | n_events | c_<事件码>... | n_events_night | n_events_dawn | n_ultra

``daily_imu``（来自 IMU，可选）::

    gpsno | date | imu_harsh_brake | imu_harsh_accel | imu_sharp_turn
          | imu_swerve | imu_suspected_crash | imu_suspected_rollover

## 核心建模思想：暴露量归一化

原始事件计数与里程天然正相关 —— 跑了 10000 km 的车，什么事件都比别人多。
**直接把计数喂给模型，模型学到的是「里程」而不是「风险」**，最后交出「跑得多
风险高」的结论，AUC 平庸，任务二的可解释性更是没法看。

所以所有风险特征的主轴一律是**速率**：

* ``每千公里事件率 = 计数 / 里程 × 1000``
* ``每百小时事件率 = 计数 / 行驶小时 × 100``

速率还有一个决定性好处：**对窗口长度不敏感**。20 天特征窗口训练、60 天窗口推理
这种长度错配，用绝对计数会直接崩，用速率基本消除 —— 这是「60 = 20 + 40」切分
能成立的前提。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config

EPS = 1e-9

EXPOSURE_COLUMNS = ["km", "hours", "night_km", "highway_km", "speed_sum", "speed_max", "n_points"]
IMU_EVENT_KEYS = (
    "imu_harsh_brake",
    "imu_harsh_accel",
    "imu_sharp_turn",
    "imu_swerve",
    "imu_suspected_crash",
    "imu_suspected_rollover",
)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _gini(values: np.ndarray) -> float:
    """日粒度事件数的 Gini 系数，用于区分「偶发一次」与「长期慢性风险」。"""
    v = np.sort(np.asarray(values, dtype=float))
    v = v[~np.isnan(v)]
    if v.size == 0 or v.sum() <= 0:
        return 0.0
    n = v.size
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * v).sum()) / (n * v.sum()) - (n + 1.0) / n)


def _safe_div(a, b, scale: float = 1.0):
    """安全除法：分母为 0 或 NaN 时返回 NaN（而不是 inf），交给模型当缺失处理。"""
    a_s = pd.Series(a) if not isinstance(a, pd.Series) else a
    b_s = pd.Series(b) if not isinstance(b, pd.Series) else b
    b_s = b_s.replace(0, np.nan)
    return a_s.astype(float) / b_s.astype(float) * scale


def _naive_date(x) -> pd.Timestamp:
    """把标量时间戳转成朴素本地日期（与日粒度表的 ``date`` 列口径一致）。"""
    t = pd.Timestamp(x)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t.normalize()


def _window_slice(df: Optional[pd.DataFrame], start, end) -> pd.DataFrame:
    """按 ``(start, end]`` 切片（左开右闭，避免相邻窗口重复计入边界日）。

    比较前统一时区口径：日粒度表的 ``date`` 是本地朴素日期，但调用方可能传入
    tz-aware 的窗口边界，直接比较会抛 ``Invalid comparison``。
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["gpsno", "date"])
    d = df["date"]
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    s, e = _naive_date(start), _naive_date(end)
    return df[(d > s) & (d <= e)]


# ---------------------------------------------------------------------------
# G1 暴露量
# ---------------------------------------------------------------------------
def _exposure_features(exp_win: pd.DataFrame, profile: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    if len(exp_win):
        g = exp_win.groupby("gpsno")
        out = pd.DataFrame(
            {
                "km": g["km"].sum(min_count=1),
                "hours": g["hours"].sum(min_count=1),
                "active_days": g["date"].nunique(),
                "night_km": g["night_km"].sum(min_count=1),
                "highway_km": g["highway_km"].sum(min_count=1),
                "speed_sum": g["speed_sum"].sum(min_count=1),
                "speed_max": g["speed_max"].max(),
                "n_traj_points": g["n_points"].sum(min_count=1),
            }
        ).reindex(index)
    else:
        out = pd.DataFrame(index=index)

    for col in ("km", "hours", "active_days", "night_km", "highway_km", "speed_sum", "n_traj_points"):
        if col not in out.columns:
            out[col] = np.nan

    out["km_per_active_day"] = _safe_div(out["km"], out["active_days"])
    out["night_km_share"] = _safe_div(out["night_km"], out["km"])
    out["highway_km_share"] = _safe_div(out["highway_km"], out["km"])
    out["speed_mean"] = _safe_div(out["speed_sum"], out["n_traj_points"])

    prof = profile.set_index("gpsno") if "gpsno" in profile.columns else profile
    for col in (
        "month_km",
        "month_hours",
        "month_stops",
        "highway_km_ratio",
        "morning_hours_ratio",
        "dusk_hours_ratio",
        "night_km_ratio",
        "night_hours_ratio",
    ):
        if col in prof.columns:
            out[f"profile_{col}"] = prof[col].reindex(index)

    if "energy_type" in prof.columns:
        et = prof["energy_type"].reindex(index).fillna("未知").astype(str)
        for val in et.value_counts().head(5).index.tolist():
            out[f"energy_{val}"] = (et == val).astype(int)

    # 轨迹里程与画像月均里程的交叉校验：用于发现暴露量数据本身有问题
    if "month_km" in prof.columns:
        out["km_vs_profile_ratio"] = _safe_div(out["km"], prof["month_km"].reindex(index))
    return out


# ---------------------------------------------------------------------------
# G2 事件族速率
# ---------------------------------------------------------------------------
def _family_counts_from_daily(ev_daily: pd.DataFrame, cfg: Config, index: pd.Index) -> pd.DataFrame:
    """把日粒度事件表按车聚合成各族的计数与严重度加权分。"""
    weights = {int(k): float(v["weight"]) for k, v in cfg.code_catalog.items()}
    code_cols = [c for c in ev_daily.columns if c.startswith("c_")]

    if len(ev_daily):
        per_veh = ev_daily.groupby("gpsno")[code_cols].sum().reindex(index).fillna(0.0) if code_cols else pd.DataFrame(index=index)
        n_events = ev_daily.groupby("gpsno")["n_events"].sum().reindex(index).fillna(0.0)
    else:
        per_veh = pd.DataFrame(0.0, index=index, columns=code_cols)
        n_events = pd.Series(0.0, index=index)

    out = pd.DataFrame(index=index)
    for fam in cfg.all_family_names():
        cols = [f"c_{c}" for c in cfg.family_codes(fam) if f"c_{c}" in per_veh.columns]
        out[f"count_{fam}"] = per_veh[cols].sum(axis=1) if cols else 0.0
        fam_w = np.mean([weights.get(int(c), 1.0) for c in cfg.family_codes(fam)]) if cfg.family_codes(fam) else 1.0
        out[f"sev_{fam}"] = out[f"count_{fam}"] * fam_w

    out["total_events"] = n_events

    sev_total = pd.Series(0.0, index=index)
    for c in code_cols:
        code = int(c[2:])
        if code in weights:
            sev_total = sev_total + per_veh[c] * weights[code]
    out["severity_total"] = sev_total
    return out


def _add_rate_features(g2: pd.DataFrame, g1: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    km = g1["km"] if "km" in g1.columns else pd.Series(np.nan, index=g1.index)
    hours = g1["hours"] if "hours" in g1.columns else pd.Series(np.nan, index=g1.index)
    per_km = float(cfg.exposure["per_km"])
    per_hour = float(cfg.exposure["per_hour"])

    for fam in cfg.all_family_names():
        g2[f"rate_per_1kkm_{fam}"] = _safe_div(g2[f"count_{fam}"], km, per_km)
        g2[f"rate_per_100h_{fam}"] = _safe_div(g2[f"count_{fam}"], hours, per_hour)
        g2[f"share_{fam}"] = _safe_div(g2[f"count_{fam}"], g2["total_events"])

    g2["rate_per_1kkm_total"] = _safe_div(g2["total_events"], km, per_km)
    g2["rate_per_100h_total"] = _safe_div(g2["total_events"], hours, per_hour)
    g2["rate_severity_per_1kkm"] = _safe_div(g2["severity_total"], km, per_km)
    return g2


# ---------------------------------------------------------------------------
# G3 时间结构
# ---------------------------------------------------------------------------
def _temporal_features(ev_win: pd.DataFrame, cfg: Config, index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    if not len(ev_win):
        return out

    g = ev_win.groupby("gpsno")
    total = g["n_events"].sum().reindex(index)
    night = g["n_events_night"].sum().reindex(index) if "n_events_night" in ev_win.columns else None
    dawn = g["n_events_dawn"].sum().reindex(index) if "n_events_dawn" in ev_win.columns else None

    if night is not None:
        out["night_event_share"] = _safe_div(night.fillna(0.0), total)
    if dawn is not None:
        out["dawn_event_share"] = _safe_div(dawn.fillna(0.0), total)

    if "weekday" in ev_win.columns:
        wk = ev_win[ev_win["weekday"].isin(cfg.temporal["weekend_days"])]
        out["weekend_event_share"] = _safe_div(
            wk.groupby("gpsno")["n_events"].sum().reindex(index).fillna(0.0), total
        )
    return out


# ---------------------------------------------------------------------------
# G4 趋势与集中度
# ---------------------------------------------------------------------------
def _trend_features(
    ev_win: pd.DataFrame, index: pd.Index, window_start: pd.Timestamp, window_end: pd.Timestamp
) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    if not len(ev_win):
        return out

    mid = pd.Timestamp(window_start).normalize() + (pd.Timestamp(window_end).normalize() - pd.Timestamp(window_start).normalize()) / 2
    early = ev_win[ev_win["date"] <= mid].groupby("gpsno")["n_events"].sum().reindex(index).fillna(0.0)
    late = ev_win[ev_win["date"] > mid].groupby("gpsno")["n_events"].sum().reindex(index).fillna(0.0)
    # 用对数比而非差值：对量级差异稳健
    out["log_rate_ratio_late_early"] = np.log((late + 1.0) / (early + 1.0))

    daily_cnt = ev_win.groupby(["gpsno", "date"])["n_events"].sum()
    total = daily_cnt.groupby(level=0).sum().reindex(index)
    out["max_day_share"] = _safe_div(daily_cnt.groupby(level=0).max().reindex(index).fillna(0.0), total)

    by_veh = daily_cnt.groupby(level=0)
    gini = by_veh.apply(lambda s: _gini(s.values))
    out["day_gini"] = gini.reindex(index).fillna(0.0)

    start = pd.Timestamp(window_start).normalize()
    wk = ev_win.assign(w=((ev_win["date"] - start).dt.days // 7)).groupby(["gpsno", "w"])["n_events"].sum()

    def _slope(s: pd.Series) -> float:
        if len(s) < 2:
            return 0.0
        x = np.asarray([t[1] for t in s.index], dtype=float)
        y = s.values.astype(float)
        if y.std() == 0:
            return 0.0
        return float(np.polyfit(x, y, 1)[0] / (y.mean() + EPS))

    out["weekly_rate_slope"] = wk.groupby(level=0).apply(_slope).reindex(index).fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# G5 高危 recency
# ---------------------------------------------------------------------------
def _ultra_features(
    ev_win: pd.DataFrame, cfg: Config, index: pd.Index, window_end: pd.Timestamp
) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    for code in cfg.family_codes("ultra"):
        col = f"c_{code}"
        if len(ev_win) and col in ev_win.columns:
            out[f"count_code_{code}"] = ev_win.groupby("gpsno")[col].sum().reindex(index).fillna(0.0)
        else:
            out[f"count_code_{code}"] = 0.0

    if len(ev_win) and "n_ultra" in ev_win.columns:
        rows = ev_win[ev_win["n_ultra"] > 0]
        if len(rows):
            last = rows.groupby("gpsno")["date"].max().reindex(index)
            out["days_since_last_ultra"] = (pd.Timestamp(window_end).normalize() - last).dt.days
        else:
            out["days_since_last_ultra"] = np.nan
    else:
        out["days_since_last_ultra"] = np.nan

    # 有/无超高危事件的显式标记：比靠计数为 0 更稳健（区分「没有」与「没记录」）
    ultra_cols = [c for c in out.columns if c.startswith("count_code_")]
    out["has_ultra_in_window"] = (out[ultra_cols].sum(axis=1) > 0).astype(int) if ultra_cols else 0
    return out


# ---------------------------------------------------------------------------
# G6 覆盖度与删失
# ---------------------------------------------------------------------------
def _coverage_features(
    ev_win: pd.DataFrame,
    g1: pd.DataFrame,
    cfg: Config,
    index: pd.Index,
    window_days: int,
    daily_imu: Optional[pd.DataFrame],
    km: pd.Series,
    hours: pd.Series,
) -> pd.DataFrame:
    out = pd.DataFrame(index=index)

    compliance_cols = [f"c_{c}" for c in cfg.compliance_codes]
    if len(ev_win):
        present = [c for c in compliance_cols if c in ev_win.columns]
        out["compliance_count"] = (
            ev_win.groupby("gpsno")[present].sum().sum(axis=1).reindex(index).fillna(0.0) if present else 0.0
        )
        out["event_days"] = ev_win.groupby("gpsno")["date"].nunique().reindex(index).fillna(0.0)
    else:
        out["compliance_count"] = 0.0
        out["event_days"] = 0.0

    out["compliance_rate_per_100h"] = _safe_div(
        out["compliance_count"], hours, float(cfg.exposure["per_hour"])
    )
    total_with_compliance = out["compliance_count"] + (
        ev_win.groupby("gpsno")["n_events"].sum().reindex(index).fillna(0.0) if len(ev_win) else 0.0
    )
    out["compliance_share"] = _safe_div(out["compliance_count"], total_with_compliance)

    out["trajectory_days"] = g1["active_days"]
    out["trajectory_coverage_ratio"] = (g1["active_days"] / max(window_days, 1)).clip(0, 1)
    out["low_coverage_flag"] = (
        g1["active_days"].fillna(0) < float(cfg.censoring["min_trajectory_days"])
    ).astype(int)

    if daily_imu is not None and len(daily_imu):
        imu_win = daily_imu
        if len(imu_win):
            gg = imu_win.groupby("gpsno")
            out["imu_coverage_days"] = gg["date"].nunique().reindex(index)
            out["imu_coverage_ratio"] = (out["imu_coverage_days"] / max(window_days, 1)).clip(0, 1)
            for key in IMU_EVENT_KEYS:
                if key in imu_win.columns:
                    out[f"count_{key}"] = gg[key].sum().reindex(index)
                    out[f"rate_per_1kkm_{key}"] = _safe_div(out[f"count_{key}"], km, float(cfg.exposure["per_km"]))
            out["has_imu"] = out["imu_coverage_days"].notna().astype(int)
        else:
            out["has_imu"] = 0
    else:
        out["has_imu"] = 0
    return out


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def build_features(
    daily_exposure: pd.DataFrame,
    daily_events: pd.DataFrame,
    profile: pd.DataFrame,
    cfg: Config,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    vehicles: Sequence[str],
    daily_imu: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """构建六组特征。

    参数
    ----
    window_start, window_end
        特征窗口 ``(window_start, window_end]``。
    vehicles
        需要输出的车辆全集。不在窗口内出现的车辆会得到 NaN 暴露量与 0 事件计数，
        **不会**被静默丢弃 —— 「这辆车数据缺失」本身就是重要特征。

    关于目标泄漏
    ------------
    本函数只接受 ``(window_start, window_end]`` 内的事件。调用方必须保证
    ``window_end`` 不越过结果窗口起点；构造增广样本时这一点尤其关键。泄漏防护的
    第二道闸门在 :mod:`ie_safety.models.leakage`。
    """
    index = pd.Index(sorted(map(str, vehicles)), name="gpsno")
    window_start = _naive_date(window_start)
    window_end = _naive_date(window_end)
    window_days = max(int((window_end - window_start).days), 1)

    exp_win = _window_slice(daily_exposure, window_start, window_end)
    ev_win = _window_slice(daily_events, window_start, window_end)
    if len(ev_win) and "weekday" not in ev_win.columns:
        ev_win = ev_win.assign(weekday=ev_win["date"].dt.weekday)

    g1 = _exposure_features(exp_win, profile, index)
    km = g1["km"]
    hours = g1["hours"]

    imu_win = _window_slice(daily_imu, window_start, window_end) if daily_imu is not None else None

    g2 = _add_rate_features(_family_counts_from_daily(ev_win, cfg, index), g1, cfg)
    g3 = _temporal_features(ev_win, cfg, index)
    g4 = _trend_features(ev_win, index, window_start, window_end)
    g5 = _ultra_features(ev_win, cfg, index, window_end)
    g6 = _coverage_features(ev_win, g1, cfg, index, window_days, imu_win, km, hours)

    out = pd.concat([g1, g2, g3, g4, g5, g6], axis=1)
    out.index.name = "gpsno"
    return out


# ---------------------------------------------------------------------------
# 特征字典
# ---------------------------------------------------------------------------
def feature_dictionary(cfg: Config) -> pd.DataFrame:
    """产出特征字典（组 / 名称 / 定义 / 时间来源），用于文档与泄漏审计。"""
    rows: List[Dict[str, str]] = []

    def add(group: str, name: str, desc: str, source: str) -> None:
        rows.append({"group": group, "feature": name, "description": desc, "time_source": source})

    add("G1 暴露量", "km", "窗口内总里程（轨迹 distance 累加）", "特征窗口")
    add("G1 暴露量", "hours", "窗口内总行驶时长（轨迹 run_time 累加）", "特征窗口")
    add("G1 暴露量", "active_days", "窗口内有轨迹数据的天数", "特征窗口")
    add("G1 暴露量", "km_per_active_day", "日均里程", "特征窗口")
    add("G1 暴露量", "night_km_share", "夜间里程占比（轨迹实算）", "特征窗口")
    add("G1 暴露量", "highway_km_share", "高速里程占比（轨迹速度阈值近似）", "特征窗口")
    add("G1 暴露量", "speed_mean / speed_max", "速度均值与窗口内最高速度", "特征窗口")
    add("G1 暴露量", "profile_*", "车辆画像近半年月均值（长期行为基准）", "画像（半年，早于窗口）")
    add("G1 暴露量", "km_vs_profile_ratio", "轨迹里程 / 画像月均里程（暴露量交叉校验）", "特征窗口")
    for fam in cfg.all_family_names():
        label = cfg.families[fam]["name"]
        add("G2 事件速率", f"count_{fam}", f"{label} 事件计数（仅窗口内）", "特征窗口")
        add("G2 事件速率", f"rate_per_1kkm_{fam}", f"{label} 每千公里事件率（暴露量归一化）", "特征窗口")
        add("G2 事件速率", f"rate_per_100h_{fam}", f"{label} 每百小时事件率", "特征窗口")
        add("G2 事件速率", f"share_{fam}", f"{label} 占全部事件的比例", "特征窗口")
    add("G2 事件速率", "severity_total / rate_severity_per_1kkm", "严重度加权总分与其速率", "特征窗口")
    add("G3 时间结构", "night_event_share", "夜间事件占比", "特征窗口")
    add("G3 时间结构", "dawn_event_share", "凌晨 0-5 点事件占比", "特征窗口")
    add("G3 时间结构", "weekend_event_share", "周末事件占比", "特征窗口")
    add("G4 趋势与集中度", "log_rate_ratio_late_early", "后半段 vs 前半段事件数对数比", "特征窗口")
    add("G4 趋势与集中度", "weekly_rate_slope", "周粒度事件数归一化斜率", "特征窗口")
    add("G4 趋势与集中度", "max_day_share", "单日最大事件数占比", "特征窗口")
    add("G4 趋势与集中度", "day_gini", "日粒度事件数 Gini（区分偶发与慢性）", "特征窗口")
    add("G5 高危 recency", "count_code_11803", "窗口内事故计数", "特征窗口")
    add("G5 高危 recency", "count_code_11804", "窗口内未遂事故计数", "特征窗口")
    add("G5 高危 recency", "days_since_last_ultra", "距最近一次超高危事件的天数", "特征窗口")
    add("G5 高危 recency", "has_ultra_in_window", "窗口内是否出现过高危事件", "特征窗口")
    add("G6 覆盖与删失", "compliance_count / compliance_share", "摄像头遮挡与角度异常（设备可见性）", "特征窗口")
    add("G6 覆盖与删失", "trajectory_days / trajectory_coverage_ratio", "轨迹覆盖天数与覆盖率", "特征窗口")
    add("G6 覆盖与删失", "event_days", "有事件记录的天数", "特征窗口")
    add("G6 覆盖与删失", "low_coverage_flag", "轨迹覆盖不足标记（防「沉默即安全」）", "特征窗口")
    add("G6 覆盖与删失", "has_imu", "窗口内是否有 IMU 数据（缺失保持 NaN，不填 0）", "特征窗口")

    return pd.DataFrame(rows)
