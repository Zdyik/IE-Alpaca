"""标签口径探测 —— 整个项目的决策闸门。

## 为什么必须先解决这件事

赛题要求「用过去 60 天的驾驶行为预测未来 40 天是否发生事故」，但给出的数据
**只有 60 天**（观测期 2026-06-01 ~ 07-30）。未来 40 天的真实标签在赛方手里，
也就是说：**我们很可能没有任何带标注的样本**。

同时，风险事件数据里包含事件码 ``11803 事故`` 与 ``11804 未遂事故``，覆盖整个
观测期。也就是说，这 60 天里发生过的出险是被记录下来的。

两件事合起来指向同一个解法，而且数字对得极其整齐：

    60 = 20 + 40

于是用**时间切分自监督**：前 ``feature_days`` 天作特征窗口，其后
``outcome_days`` 天作结果窗口，标签 = 结果窗口内是否出现事故或未遂事故。
这与赛方要求的 40 天前瞻期完全对齐。再进一步可以做**多起点增广**：特征窗口取
``[1..t]``、结果窗口取 ``[t+1..t+40]``，只要 ``t <= 60 - 40 = 20``，结果窗口就
始终落在观测期内。

## 但这个等式有一个前提

它要求 ``11803``/``11804`` 在观测期里**确实有足够多的正例**。如果事故是 0 条
或个位数，监督信号就撑不起来，必须降级。降级路径有三条，按优先级：

1. 事故 + 未遂事故足够 → 用它（默认口径）
2. 事故稀疏但未遂事故充足 → 仍用未遂事故为主，文档显式说明
3. 两者都稀疏 → 并入 ``30000 前碰撞预警`` 作代理标签，并考虑缩短结果窗口
   以提高基率（代价是前瞻期与赛题要求的 40 天错配，必须写进文档）

**这个决策决定后面所有代码怎么写，所以它是闸门，不是普通步骤。**
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config

# ---- 降级判定的阈值（写成常量而不是埋在代码里，方便答辩时解释）-------------
#: 事故条数低于此值即认为「事故信号不足以独立支撑监督」
MIN_ACCIDENTS_FOR_MAIN_SCHEME = 10
#: 事故+未遂事故的正例车辆数低于此值即认为过于稀疏
MIN_POSITIVE_VEHICLES = 15
#: 在保险箱切分下，正例率低于此值则考虑缩短结果窗口
MIN_BASE_RATE_ACCEPTABLE = 0.05
#: 缩短后的备选结果窗口长度
FALLBACK_OUTCOME_DAYS = 20


def label_events(events: pd.DataFrame, cfg: Config, codes: Optional[List[int]] = None) -> pd.DataFrame:
    """筛出与标签相关的事件。

    ``codes`` 为 ``None`` 时使用配置中的当前生效口径。返回带 ``ts`` 列的子集；
    若 ``ts`` 缺失则抛错——**没有时间戳就无法构造时序标签，也就无法做任何泄漏
    防护**，这里选择快速失败而不是静默降级。
    """
    if "ts" not in events.columns:
        raise ValueError("事件表缺少 ts 列；无法构造时序标签。请确认 start_time 已解析。")
    use = codes if codes is not None else cfg.label_codes()
    sub = events[events["event_type"].isin(use)].copy()
    return sub.dropna(subset=["ts"])


def _base_rate_at_origin(
    lab: pd.DataFrame, vehicles: List[str], origin_day: int, outcome_days: int, t0: pd.Timestamp
) -> Dict[str, Any]:
    """给定起点 ``origin_day``，计算结果窗口内的标签基率。

    特征窗口 = 第 ``1..origin_day`` 天，结果窗口 = 第 ``origin_day+1 .. origin_day+outcome_days`` 天。
    """
    start = t0 + pd.Timedelta(days=origin_day)
    end = t0 + pd.Timedelta(days=origin_day + outcome_days)
    win = lab[(lab["ts"] > start) & (lab["ts"] <= end)]
    positive = set(win["gpsno"].unique())
    n = len(vehicles)
    return {
        "origin_day": int(origin_day),
        "feature_window": [str(t0.date()), str((t0 + pd.Timedelta(days=origin_day)).date())],
        "outcome_window": [str(start.date()), str(end.date())],
        "positive_vehicles": len(positive),
        "base_rate": round(len(positive) / n, 4) if n else 0.0,
        "positive_events": int(len(win)),
    }


def probe_label_structure(events: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """探测标签结构：计数、时间分布、覆盖车辆、各起点基率。"""
    acc_codes = cfg.accident_codes
    nm_codes = cfg.near_miss_codes

    acc = events[events["event_type"].isin(acc_codes)]
    nm = events[events["event_type"].isin(nm_codes)]

    vehicles = sorted(events["gpsno"].dropna().unique().tolist())
    lab = label_events(events, cfg)

    t0 = lab["ts"].min().normalize() if len(lab) else (
        pd.Timestamp(cfg["data"]["observation_start"], tz=cfg.timezone)
    )
    # 观测期起点应以全量事件（而非仅标签事件）为准，否则首日无标签事件会前移起点
    t0 = events["ts"].min().normalize() if "ts" in events.columns and events["ts"].notna().any() else t0

    origins = sorted(set(cfg.dev_origins + [cfg.vault_origin]))
    rates = [
        _base_rate_at_origin(lab, vehicles, t, cfg.outcome_days, t0) for t in origins if t >= 0
    ]

    daily = (
        lab.assign(day=lab["ts"].dt.normalize()).groupby("day").size().sort_index()
        if len(lab)
        else pd.Series(dtype=int)
    )

    return {
        "observation_start": str(t0.date()),
        "observation_end": str((events["ts"].max() if "ts" in events.columns and events["ts"].notna().any() else t0).date()),
        "n_vehicles_in_events": len(vehicles),
        "accident_codes": acc_codes,
        "near_miss_codes": nm_codes,
        "accident_events": int(len(acc)),
        "accident_vehicles": int(acc["gpsno"].nunique()) if len(acc) else 0,
        "near_miss_events": int(len(nm)),
        "near_miss_vehicles": int(nm["gpsno"].nunique()) if len(nm) else 0,
        "label_events_total": int(len(lab)),
        "label_vehicles": int(lab["gpsno"].nunique()) if len(lab) else 0,
        "daily_label_event_counts": {str(k.date()): int(v) for k, v in daily.items()},
        "base_rates_by_origin": rates,
    }


def decide_label_scheme(probe: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """依据探测结果决定标签口径与结果窗口长度。

    返回的字典会被写到 ``reports/label_decision.json``，并作为文档中「标签定义」
    一节的依据。**任何降级都必须在文档里显式说明，不允许静默降级。**
    """
    acc = int(probe["accident_events"])
    nm = int(probe["near_miss_events"])
    positives = int(probe["label_vehicles"])

    vault = next(
        (r for r in probe["base_rates_by_origin"] if r["origin_day"] == cfg.vault_origin),
        None,
    )
    vault_base = float(vault["base_rate"]) if vault else 0.0

    scheme = "accident+near_miss"
    outcome_days = cfg.outcome_days
    codes = sorted(set(cfg.accident_codes + cfg.near_miss_codes))
    reasons: List[str] = []

    if acc == 0:
        reasons.append(
            f"观测期内事故（{cfg.accident_codes}）事件为 0 条，事故信号无法独立支撑监督。"
        )
        scheme = "near_miss_only"
    elif acc < MIN_ACCIDENTS_FOR_MAIN_SCHEME:
        reasons.append(
            f"事故事件仅 {acc} 条，低于阈值 {MIN_ACCIDENTS_FOR_MAIN_SCHEME}；"
            "仍纳入标签但以未遂事故为主要正例来源。"
        )
        scheme = "accident+near_miss"

    if positives < MIN_POSITIVE_VEHICLES:
        reasons.append(
            f"事故+未遂事故覆盖车辆仅 {positives} 台，低于阈值 {MIN_POSITIVE_VEHICLES}；"
            "启用代理标签（并入前碰撞预警）。"
        )
        scheme = "proxy"
        codes = sorted(set(codes + cfg.proxy_extra_codes))

    if vault_base < MIN_BASE_RATE_ACCEPTABLE:
        reasons.append(
            f"保险箱切分（origin={cfg.vault_origin}，outcome={cfg.outcome_days} 天）"
            f"的标签基率仅 {vault_base:.4f}，低于 {MIN_BASE_RATE_ACCEPTABLE}；"
            f"建议把结果窗口缩短到 {FALLBACK_OUTCOME_DAYS} 天以提高基率，"
            "代价是前瞻期与赛题要求的 40 天错配（必须写进文档）。"
        )
        outcome_days = FALLBACK_OUTCOME_DAYS

    if not reasons:
        reasons.append(
            f"事故 {acc} 条 / 未遂事故 {nm} 条，覆盖 {positives} 台车；"
            f"保险箱基率 {vault_base:.4f}。默认口径（事故+未遂事故，40 天结果窗口）成立。"
        )

    return {
        "scheme": scheme,
        "label_codes": codes,
        "accident_codes": cfg.accident_codes,
        "near_miss_codes": cfg.near_miss_codes,
        "proxy_extra_codes": cfg.proxy_extra_codes if scheme == "proxy" else [],
        "feature_days": cfg.feature_days,
        "outcome_days": outcome_days,
        "outcome_days_original": cfg.outcome_days,
        "horizon_mismatch": outcome_days != cfg.outcome_days,
        "vault_origin": cfg.vault_origin,
        "vault_base_rate": vault_base,
        "degraded": scheme != "accident+near_miss" or outcome_days != cfg.outcome_days,
        "reasons": reasons,
        "probe": probe,
    }


def write_decision(decision: Dict[str, Any], path: Path) -> Path:
    """把标签口径决策写入 JSON（含完整探测证据，便于复核）。"""
    from ..textio import write_json_lf

    return write_json_lf(path, decision)


def load_decision(path: Path) -> Optional[Dict[str, Any]]:
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
