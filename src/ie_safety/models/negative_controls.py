"""四类负对照 —— 真正防止自欺的一层。

普通做法是「打乱标签跑一次模型，看 AUC 是否回到 0.5」。这**不够**：它测不出
**选择偏差**，而选择偏差恰恰是最可能骗到自己的力量。

因此本模块的四个对照都跑**完整流水线**（含特征准备与模型选择），并且都要求
在**与真实流程完全相同的协议**下进行：

============  ==========================================  ==============================
对照           做法                                        不看会怎样
============  ==========================================  ==============================
标签置换       打乱标签，重跑完整流水线                     协议存在泄漏或选择偏差
影子标签       用同基率的随机 0/1 当标签，重跑完整流水线    特征选择在挖噪声
时间倒置       特征取自结果窗口、标签取自特征窗口           特征是「事后可见」的伪特征
随机特征注入   加入与真实特征等量的随机噪声列               特征选择在贪心过拟合
============  ==========================================  ==============================

参考：跑 200 组实验挑最好的，CV AUC 期望虚高约 +0.08（见
:func:`ie_safety.models.validation.selection_bias_table`）。四个对照就是在
检查「我们的协议会不会自己制造出这种虚高」。
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from .pipeline import pipeline_auc

logger = logging.getLogger(__name__)

#: 随机特征注入允许的 AUC 波动上限（超过即认为协议在贪心过拟合）
RANDOM_FEATURE_TOLERANCE = 0.02
#: 负对照的期望区间
NULL_AUC_RANGE = (0.45, 0.55)


def _verdict(name: str, auc: float, expected: str, passed: bool) -> Dict[str, object]:
    return {
        "check": name,
        "passed": bool(passed),
        "observed_auc": round(float(auc), 4) if np.isfinite(auc) else None,
        "expected": expected,
    }


def control_label_permutation(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: Config,
    n_shuffles: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    """对照 1：标签置换。"""
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y).astype(int)
    aucs: List[float] = []
    for _ in range(int(n_shuffles)):
        perm = rng.permutation(len(y_arr))
        a = pipeline_auc(X, y_arr[perm], groups, cfg)
        if np.isfinite(a):
            aucs.append(float(a))
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    out = _verdict("label_permutation", mean_auc, f"AUC ∈ {NULL_AUC_RANGE}", NULL_AUC_RANGE[0] <= mean_auc <= NULL_AUC_RANGE[1])
    out["per_run"] = [round(a, 4) for a in aucs]
    out["interpretation"] = (
        "打乱标签后回到随机水平，流水线本身无系统性泄漏。"
        if out["passed"]
        else "打乱标签后仍显著偏离 0.5：流水线（含选择过程）存在泄漏或选择偏差。"
    )
    return out


def control_shadow_label(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: Config,
    n_shadows: int = 3,
    seed: int = 42,
) -> Dict[str, object]:
    """对照 2：影子标签。

    用与真实标签**同基率**的随机 0/1 作为标签，跑完整流水线（含特征选择）。
    与标签置换的区别：置换保留了原本的正例集合（打乱后每个窗口的正例数不变），
    而影子标签是完全独立生成的噪声 —— 因此它对「特征选择在挖噪声」更敏感。
    """
    rng = np.random.default_rng(seed + 1000)
    y_arr = np.asarray(y).astype(int)
    base = float(y_arr.mean())
    aucs: List[float] = []
    for _ in range(int(n_shadows)):
        shadow = (rng.random(len(y_arr)) < base).astype(int)
        if shadow.sum() < 3 or shadow.sum() > len(shadow) - 3:
            continue
        a = pipeline_auc(X, shadow, groups, cfg)
        if np.isfinite(a):
            aucs.append(float(a))
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    out = _verdict("shadow_label", mean_auc, f"AUC ∈ {NULL_AUC_RANGE}", NULL_AUC_RANGE[0] <= mean_auc <= NULL_AUC_RANGE[1])
    out["per_run"] = [round(a, 4) for a in aucs]
    out["base_rate"] = round(base, 4)
    out["interpretation"] = (
        "同基率随机标签下没有跑出信号，特征选择没有在挖噪声。"
        if out["passed"]
        else "同基率随机标签下仍跑出信号：特征选择过程在挖噪声，必须收紧。"
    )
    return out


def control_time_reversal(
    build_reversed: Callable[[], Tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    cfg: Config,
) -> Dict[str, object]:
    """对照 3：时间倒置。

    把「特征窗口」与「结果窗口」对调：特征取自后 40 天，标签取自前 20 天。
    如果模型这时仍能拿到明显高于 0.5 的 AUC，说明特征里含有**事后可见**的信息
    （典型的如「出险后的处罚记录」「事故后的维修里程」）—— 这些在真实前瞻预测
    里是不存在的。

    期望：AUC ≈ 0.5。略微高于 0.5 是正常的，因为驾驶行为本身有持续性，
    前 20 天的风险水平确实与后 40 天相关 —— 但那属于「真实信号」而非泄漏。
    因此判据放宽到 0.60。
    """
    X_rev, y_rev, g_rev = build_reversed()
    a = pipeline_auc(X_rev, np.asarray(y_rev).astype(int), g_rev, cfg)
    passed = bool(np.isfinite(a) and a <= 0.60)
    out = _verdict("time_reversal", a, "AUC ≤ 0.60（时间倒置后不应有强预测力）", passed)
    out["interpretation"] = (
        "时间倒置后预测力消失，特征不含事后可见信息。"
        if passed
        else "时间倒置后仍有强预测力，特征可能含事后可见信息（如出险后记录），需逐个核对。"
    )
    return out


def control_random_features(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: Config,
    n_random: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, object]:
    """对照 4：随机特征注入。

    注入与真实特征**等量**的随机噪声列。若 CV AUC 因此变好，说明特征选择在贪心
    过拟合噪声（在 N=500 上这非常容易发生）。
    """
    rng = np.random.default_rng(seed + 2000)
    base_auc = pipeline_auc(X, y, groups, cfg)

    n_rand = int(n_random if n_random is not None else max(X.shape[1], 5))
    Xr = X.copy()
    for i in range(n_rand):
        Xr[f"_random_{i}"] = rng.normal(size=len(Xr))

    inj_auc = pipeline_auc(Xr, y, groups, cfg)
    delta = float(inj_auc - base_auc) if np.isfinite(inj_auc) and np.isfinite(base_auc) else float("nan")
    passed = bool(np.isfinite(delta) and delta <= RANDOM_FEATURE_TOLERANCE)

    out = _verdict("random_feature_injection", inj_auc, f"ΔAUC ≤ {RANDOM_FEATURE_TOLERANCE}", passed)
    out["base_auc"] = round(float(base_auc), 4) if np.isfinite(base_auc) else None
    out["delta"] = round(delta, 4) if np.isfinite(delta) else None
    out["n_injected"] = n_rand
    out["interpretation"] = (
        "注入等量随机特征后指标未提升，协议没有贪心过拟合。"
        if passed
        else f"注入随机特征后 AUC 提升 {delta:+.4f}，协议在贪心过拟合噪声，必须收紧特征选择。"
    )
    return out


def run_all_negative_controls(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: Config,
    build_reversed: Optional[Callable[[], Tuple[pd.DataFrame, np.ndarray, np.ndarray]]] = None,
    seed: int = 42,
    include_time_reversal: bool = True,
) -> Dict[str, object]:
    """跑完全部四类对照并汇总。"""
    results: List[Dict[str, object]] = []
    results.append(control_label_permutation(X, y, groups, cfg, seed=seed))
    results.append(control_shadow_label(X, y, groups, cfg, seed=seed))
    if include_time_reversal and build_reversed is not None:
        try:
            results.append(control_time_reversal(build_reversed, cfg))
        except Exception as exc:  # pragma: no cover - 数据不足时不应中断整体流程
            results.append(
                {
                    "check": "time_reversal",
                    "passed": False,
                    "observed_auc": None,
                    "expected": "AUC ≤ 0.60",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    results.append(control_random_features(X, y, groups, cfg, seed=seed))

    return {
        "n_checks": len(results),
        "n_passed": sum(1 for r in results if r.get("passed")),
        "all_passed": all(bool(r.get("passed")) for r in results),
        "results": results,
    }
