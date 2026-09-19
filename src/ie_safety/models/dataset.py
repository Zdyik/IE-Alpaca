"""窗口切分与监督样本构造 —— 「60 = 20 + 40」的具体实现。

## 监督信号从哪来

赛题要求「用过去 60 天预测未来 40 天」，但给出的数据只有 60 天，未来 40 天的
真实标签在赛方手里。也就是说**我们没有带标注的样本**。而事件数据里包含
``11803 事故`` 与 ``11804 未遂事故``，覆盖整个观测期。

于是用**时间切分自监督**：

    特征窗口 = 第 1..t 天
    结果窗口 = 第 t+1 .. t+40 天
    标签     = 该车在结果窗口内是否出现事故或未遂事故

``t = 20`` 时结果窗口刚好落在 60 天观测期内（20 + 40 = 60）。这不是巧合，而是
唯一能让 40 天结果窗口与赛题前瞻期对齐的整数切分。

## 多起点增广与它的代价

只要 ``t <= 观测期 - outcome_days``，结果窗口就始终在观测期内，因此
``t = 1..20`` 都能构造合法样本，总计 ``20 × 500 = 10000`` 条。

代价是样本**高度重叠**（同一台车、结果窗口互相包含）。因此：

* ``t = 20`` 的样本构成**保险箱**，全程锁死，只在最终评估时开一次；
* ``t ∈ {5, 10, 15}`` 用于开发，交叉验证一律 ``GroupKFold by gpsno``。

如果不做这个区分，同一台车的多个窗口会同时出现在训练与验证折里，指标会虚高
到离谱 —— 这是这类自监督构造最容易踩的坑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..features.build import build_features


@dataclass
class WindowSpec:
    """一次时间切分的完整定义。"""

    origin_day: int               # t：特征窗口长度（天）
    feature_start: pd.Timestamp
    feature_end: pd.Timestamp
    outcome_end: pd.Timestamp
    outcome_days: int

    @property
    def label(self) -> str:
        return f"origin{self.origin_day}"


def make_window(
    t0: pd.Timestamp, origin_day: int, outcome_days: int, observation_days: int
) -> WindowSpec:
    """构造窗口并**强制校验结果窗口不越界**。

    越界意味着结果窗口超出了观测期 —— 要么标签会系统性缺失（把「没数据」当成
    「没出险」），要么根本构造不出标签。这里选择快速失败。
    """
    t0 = pd.Timestamp(t0).normalize()
    if origin_day <= 0:
        raise ValueError(f"origin_day 必须为正，收到 {origin_day}")
    if origin_day + outcome_days > observation_days:
        raise ValueError(
            f"窗口越界：origin_day({origin_day}) + outcome_days({outcome_days}) "
            f"> 观测期({observation_days}) 天。结果窗口会超出观测期，标签不可靠。"
        )
    feature_end = t0 + pd.Timedelta(days=origin_day)
    outcome_end = feature_end + pd.Timedelta(days=outcome_days)
    return WindowSpec(
        origin_day=origin_day,
        feature_start=t0,
        feature_end=feature_end,
        outcome_end=outcome_end,
        outcome_days=outcome_days,
    )


def label_from_events(
    daily_events: pd.DataFrame,
    window: WindowSpec,
    label_codes: Sequence[int],
    vehicles: Sequence[str],
) -> pd.Series:
    """从日粒度事件表构造标签：结果窗口内是否出现标签事件码。

    **只在结果窗口内统计**，这是标签与特征之间的硬边界。
    """
    index = pd.Index(sorted(map(str, vehicles)), name="gpsno")
    y = pd.Series(0, index=index, dtype=int)

    cols = [f"c_{int(c)}" for c in label_codes]
    present = [c for c in cols if c in daily_events.columns]
    if not present or not len(daily_events):
        return y

    sel = daily_events[
        (daily_events["date"] > window.feature_end.normalize())
        & (daily_events["date"] <= window.outcome_end.normalize())
    ]
    if not len(sel):
        return y
    hits = sel.groupby("gpsno")[present].sum().sum(axis=1)
    y.loc[y.index.intersection(hits.index)] = (hits.reindex(y.index).fillna(0) > 0).astype(int)
    return y


def build_window_dataset(
    daily_exposure: pd.DataFrame,
    daily_events: pd.DataFrame,
    daily_imu: Optional[pd.DataFrame],
    profile: pd.DataFrame,
    cfg: Config,
    t0: pd.Timestamp,
    window: WindowSpec,
    label_codes: Sequence[int],
    vehicles: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """构造一个窗口下的 ``(X, y)``。"""
    X = build_features(
        daily_exposure=daily_exposure,
        daily_events=daily_events,
        profile=profile,
        cfg=cfg,
        window_start=window.feature_start,
        window_end=window.feature_end,
        vehicles=vehicles,
        daily_imu=daily_imu,
    )
    y = label_from_events(daily_events, window, label_codes, vehicles)
    y = y.reindex(X.index)
    return X, y


def make_augmented_dataset(
    daily_exposure: pd.DataFrame,
    daily_events: pd.DataFrame,
    daily_imu: Optional[pd.DataFrame],
    profile: pd.DataFrame,
    cfg: Config,
    t0: pd.Timestamp,
    origins: Sequence[int],
    label_codes: Sequence[int],
    vehicles: Sequence[str],
    outcome_days: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """多起点增广：把若干 ``origin_day`` 的样本纵向拼起来。

    返回 ``(X, y, groups)``，其中 ``groups`` 是 ``gpsno`` —— **交叉验证必须按它
    分组**，否则同一台车的多个窗口会同时落在训练与验证折里。
    """
    od = int(outcome_days if outcome_days is not None else cfg.outcome_days)
    obs_days = int(cfg.observation_days)
    xs: List[pd.DataFrame] = []
    ys: List[pd.Series] = []
    gs: List[pd.Series] = []

    for origin in origins:
        win = make_window(t0, int(origin), od, obs_days)
        X, y = build_window_dataset(
            daily_exposure, daily_events, daily_imu, profile, cfg, t0, win, label_codes, vehicles
        )
        X = X.copy()
        X["origin_day"] = int(origin)
        xs.append(X)
        ys.append(y)
        gs.append(pd.Series(X.index.astype(str), index=X.index, name="gpsno"))

    if not xs:
        raise ValueError("origins 为空，无法构造增广数据集")

    X_all = pd.concat(xs, axis=0)
    y_all = pd.concat(ys, axis=0)
    groups = pd.concat(gs, axis=0)
    return X_all, y_all, groups


def describe_windows(
    t0: pd.Timestamp, origins: Sequence[int], outcome_days: int, observation_days: int
) -> pd.DataFrame:
    """产出窗口说明表，用于文档与复核。"""
    rows = []
    for o in origins:
        w = make_window(t0, int(o), outcome_days, observation_days)
        rows.append(
            {
                "origin_day": w.origin_day,
                "feature_window": f"{w.feature_start.date()} ~ {w.feature_end.date()}",
                "outcome_window": f"{w.feature_end.date()} ~ {w.outcome_end.date()}",
                "feature_days": w.origin_day,
                "outcome_days": w.outcome_days,
            }
        )
    return pd.DataFrame(rows)
