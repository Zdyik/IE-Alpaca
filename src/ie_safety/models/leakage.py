"""泄漏审计 —— 把「我的结果可不可信」变成可执行的三项检查。

## 为什么单独成模块

这类任务里最贵的错误不是 AUC 低，而是把一个泄漏导致的 0.95 交上去，然后在复现
测试里被打回原形。而泄漏**不会自己暴露**：它表现为一个漂亮得可疑的数字。

因此 ``> 0.85`` 的 AUC 在本项目里**默认判定为 bug**，必须先通过下面三项审计。

## 三项审计

1. **截断不变性**（本模块最有价值的一项）
   用「截至特征窗口结束」的数据重建特征，与用全量数据重建的特征逐格比对。
   如果实现里不小心看到了结果窗口的数据，两者必然不同。
   这比人工逐特征检查时间来源可靠得多 —— 它检查的是**行为**而不是**声明**。

2. **标签置换**
   打乱标签后重跑完整流水线，AUC 必须回到 0.45–0.55。
   关键细节：要跑**完整流水线（含特征选择与模型选择）**，而不是只打乱标签跑一次
   模型 —— 后者测不出选择偏差，而那正是最可能骗到自己的力量。

3. **逐特征时间来源核对**
   把特征字典里声明的 ``time_source`` 与允许的集合比对，作为人可复核的兜底。
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..features.build import build_features
from .validation import auc_score, bootstrap_auc_ci

logger = logging.getLogger(__name__)

#: 允许的特征时间来源
ALLOWED_TIME_SOURCES = {"特征窗口", "画像（半年，早于窗口）"}


def truncation_invariance_test(
    daily_exposure: pd.DataFrame,
    daily_events: pd.DataFrame,
    profile: pd.DataFrame,
    cfg: Config,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    vehicles: Sequence[str],
    daily_imu: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """截断不变性：特征不得依赖特征窗口之后的数据。

    做法：把日粒度表裁到 ``window_end`` 之后清零（模拟「未来数据根本不存在」），
    再重建特征。两次结果必须**逐格完全一致**。
    """
    def _truncate(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or not len(df):
            return df
        return df[df["date"] <= pd.Timestamp(window_end).normalize()].copy()

    def _blank_future(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """更狠的版本：保留行但把窗口之后的数据抹成 0/NaN。"""
        if df is None or not len(df):
            return df
        d = df.copy()
        mask = d["date"] > pd.Timestamp(window_end).normalize()
        num = [c for c in d.columns if c not in ("gpsno", "date")]
        d.loc[mask, num] = 0.0
        return d

    kwargs = dict(
        profile=profile, cfg=cfg, window_start=window_start, window_end=window_end, vehicles=vehicles
    )
    base = build_features(daily_exposure, daily_events, **kwargs, daily_imu=daily_imu)
    trunc = build_features(_truncate(daily_exposure), _truncate(daily_events), **kwargs,
                           daily_imu=_truncate(daily_imu))
    blanked = build_features(_blank_future(daily_exposure), _blank_future(daily_events), **kwargs,
                             daily_imu=_blank_future(daily_imu))

    def _diff(a: pd.DataFrame, b: pd.DataFrame) -> Dict[str, object]:
        common = [c for c in a.columns if c in b.columns]
        aa, bb = a[common].astype(float), b[common].astype(float)
        # 用「两者同为 NaN」或「数值近似相等」判定一致
        both_nan = aa.isna() & bb.isna()
        close = np.isclose(aa.fillna(0.0), bb.fillna(0.0), rtol=1e-9, atol=1e-9)
        same = both_nan | close
        bad = (~same).sum().sum()
        differing = [c for c in common if not same[c].all()]
        return {"n_differing_cells": int(bad), "differing_columns": differing[:20]}

    d_trunc = _diff(base, trunc)
    d_blank = _diff(base, blanked)
    passed = d_trunc["n_differing_cells"] == 0 and d_blank["n_differing_cells"] == 0

    return {
        "check": "truncation_invariance",
        "passed": bool(passed),
        "detail_truncated": d_trunc,
        "detail_blanked": d_blank,
        "interpretation": (
            "特征对特征窗口之后的数据完全不敏感 —— 实现层面不存在时间泄漏。"
            if passed
            else "特征随结果窗口数据变化而改变，存在时间泄漏，必须修复后再继续。"
        ),
    }


def timestamp_audit(cfg: Config, dictionary: pd.DataFrame) -> Dict[str, object]:
    """核对特征字典里声明的时间来源是否都在允许集合内。"""
    if dictionary is None or not len(dictionary):
        return {"check": "timestamp_audit", "passed": False, "reason": "特征字典为空"}
    bad = dictionary[~dictionary["time_source"].isin(ALLOWED_TIME_SOURCES)]
    return {
        "check": "timestamp_audit",
        "passed": bool(len(bad) == 0),
        "n_features": int(len(dictionary)),
        "violations": bad[["feature", "time_source"]].to_dict("records") if len(bad) else [],
        "allowed_sources": sorted(ALLOWED_TIME_SOURCES),
    }


def shuffle_label_test(
    run_pipeline: Callable[[pd.DataFrame, np.ndarray, np.ndarray], float],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_shuffles: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    """标签置换检验：打乱标签后重跑**完整流水线**，AUC 必须回到 0.5 附近。

    ``run_pipeline(X, y, groups) -> auc`` 必须是完整流水线（含特征选择与模型
    选择），不能只是「打乱标签再跑一次固定模型」。
    """
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y).astype(int)
    aucs: List[float] = []
    for i in range(int(n_shuffles)):
        perm = rng.permutation(len(y_arr))
        a = run_pipeline(X, y_arr[perm], groups)
        if np.isfinite(a):
            aucs.append(float(a))

    arr = np.asarray(aucs) if aucs else np.asarray([np.nan])
    mean_auc = float(np.nanmean(arr))
    passed = bool(np.isfinite(mean_auc) and 0.45 <= mean_auc <= 0.55)

    return {
        "check": "shuffle_label",
        "passed": passed,
        "mean_auc": mean_auc,
        "per_run": [round(a, 4) for a in aucs],
        "expected_range": [0.45, 0.55],
        "interpretation": (
            "打乱标签后回到随机水平，说明流水线本身没有系统性泄漏。"
            if passed
            else "打乱标签后仍显著偏离 0.5，说明流水线（含选择过程）存在泄漏或选择偏差。"
        ),
    }


def leakage_verdict(audit_results: Sequence[Dict[str, object]], auc_value: float) -> Dict[str, object]:
    """综合给出「这个 AUC 能不能信」的结论。

    规则：``AUC > 0.85`` 时，任何一项审计未过都直接判定为不可信。
    """
    failed = [r.get("check") for r in audit_results if not r.get("passed")]
    suspicious = bool(np.isfinite(auc_value) and auc_value > 0.85)
    trustworthy = (not suspicious) and (len(failed) == 0)
    if suspicious and not failed:
        note = (
            f"AUC={auc_value:.3f} 高于 0.85，但三项审计全部通过。仍建议复核特征定义"
            "与标签窗口是否与赛方口径一致。"
        )
    elif suspicious and failed:
        note = f"AUC={auc_value:.3f} 高于 0.85 且审计未通过（{failed}），判定为泄漏，不得提交。"
    elif failed:
        note = f"审计未通过（{failed}），结果不可信。"
    else:
        note = "三项审计全部通过，结果可信。"
    return {
        "auc": float(auc_value) if np.isfinite(auc_value) else None,
        "suspicious_high_auc": suspicious,
        "failed_checks": failed,
        "trustworthy": bool(trustworthy),
        "note": note,
    }
