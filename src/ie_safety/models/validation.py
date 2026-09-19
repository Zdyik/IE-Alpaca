"""验证协议与统计功效 —— 回答「多大的差距才算真差距」。

## 为什么这一层必须存在

N=500、正例率约 20% 时，AUC 的抽样标准误约为 **0.031**（Hanley-McNeil），也就是
说 ``AUC = 0.72`` 的 95% 置信区间是 ``[0.66, 0.78]``。**CV 上 0.72 与 0.76 的
差距在统计上不可区分。**如果不把这件事写进协议，就会陷进「加特征 → CV 涨 0.01
→ 再试一个」的循环，最后把噪声当成增益交上去。

因此本模块提供三件事：

1. **置信区间**：分层 bootstrap（正负例分别重抽样，保持基率）。
2. **配对差异检验**：同一批样本上比较两个模型，报 ΔAUC 的 95% CI。
   **CI 含 0 就判为无效**，不管点估计涨了多少。
3. **并列值诊断**：AUC 公式里并列计 0.5，若正负样本对的并列比例为 ``f``，
   则 ``AUC ≤ 1 − 0.5f``。浅层 GBDT 与分档评分是重灾区 —— 排序不差，却因输出
   只有几档而白丢 AUC。

## 关于效应量门槛

``configs/base.yml`` 里的 ``validation.min_effect_delta_auc = 0.04`` 就是据此
定的：低于它的改动一律记入「无效清单」，不作为「有效」处理。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# 基础指标
# ---------------------------------------------------------------------------
def auc_score(y: Sequence[int], p: Sequence[float]) -> float:
    """AUC。样本只有单一类别时返回 ``nan`` 而不是抛错（便于批量评估）。"""
    y_arr = np.asarray(y)
    if len(np.unique(y_arr)) < 2:
        return float("nan")
    return float(roc_auc_score(y_arr, np.asarray(p, dtype=float)))


def hanley_mcneil_se(y: Sequence[int], auc: float) -> float:
    """AUC 的 Hanley-McNeil 标准误。

    ``SE(A) = sqrt([A(1−A) + (n1−1)(Q1−A²) + (n0−1)(Q2−A²)] / (n1·n0))``，
    其中 ``Q1 = A/(2−A)``、``Q2 = 2A²/(1+A)``。
    """
    y_arr = np.asarray(y)
    n1 = int((y_arr == 1).sum())
    n0 = int((y_arr == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    a = float(auc)
    q1 = a / (2.0 - a) if a != 2.0 else 0.0
    q2 = 2.0 * a * a / (1.0 + a)
    num = a * (1.0 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)
    return float(np.sqrt(max(num, 0.0) / (n1 * n0)))


def _stratified_resample(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """分层重抽样：正负例分别有放回抽样，保持基率不变。"""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    idx = np.concatenate(
        [rng.choice(pos, size=pos.size, replace=True), rng.choice(neg, size=neg.size, replace=True)]
    )
    return idx


def bootstrap_auc_ci(
    y: Sequence[int], p: Sequence[float], n_boot: int = 1000, seed: int = 42, alpha: float = 0.05
) -> Dict[str, float]:
    """分层 bootstrap 的 AUC 置信区间。"""
    y_arr = np.asarray(y).astype(int)
    p_arr = np.asarray(p, dtype=float)
    point = auc_score(y_arr, p_arr)
    if not np.isfinite(point):
        return {"auc": point, "lo": float("nan"), "hi": float("nan"), "se": float("nan")}

    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(max(int(n_boot), 1)):
        idx = _stratified_resample(y_arr, rng)
        v = auc_score(y_arr[idx], p_arr[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"auc": point, "lo": float("nan"), "hi": float("nan"), "se": float("nan")}
    arr = np.asarray(vals)
    return {
        "auc": point,
        "lo": float(np.quantile(arr, alpha / 2)),
        "hi": float(np.quantile(arr, 1 - alpha / 2)),
        "se": float(arr.std(ddof=1)),
        "hanley_se": hanley_mcneil_se(y_arr, point),
    }


def paired_bootstrap_delta(
    y: Sequence[int],
    p_a: Sequence[float],
    p_b: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """配对 bootstrap 的 ΔAUC = AUC(a) − AUC(b)。

    **判据是 CI 是否含 0**，而不是点估计的大小。这正是「多大的差距才算真差距」
    这个问题的可执行答案。
    """
    y_arr = np.asarray(y).astype(int)
    pa = np.asarray(p_a, dtype=float)
    pb = np.asarray(p_b, dtype=float)
    delta = auc_score(y_arr, pa) - auc_score(y_arr, pb)
    if not np.isfinite(delta):
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "significant": False}

    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(max(int(n_boot), 1)):
        idx = _stratified_resample(y_arr, rng)
        d = auc_score(y_arr[idx], pa[idx]) - auc_score(y_arr[idx], pb[idx])
        if np.isfinite(d):
            vals.append(d)
    if not vals:
        return {"delta": delta, "lo": float("nan"), "hi": float("nan"), "significant": False}
    arr = np.asarray(vals)
    lo, hi = float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))
    return {
        "delta": float(delta),
        "lo": lo,
        "hi": hi,
        "se": float(arr.std(ddof=1)),
        "significant": bool(lo > 0 or hi < 0),
    }


# ---------------------------------------------------------------------------
# 并列值诊断
# ---------------------------------------------------------------------------
def tie_diagnostics(p: Sequence[float]) -> Dict[str, float]:
    """并列率与由此推出的 AUC 上限。

    AUC 中并列计 0.5，因此若正负样本对的并列比例是 ``f``，则
    ``AUC ≤ 1 − 0.5 f``。输出只有几档的模型（浅层 GBDT、分档评分）会被这条
    上限卡住，而且是**白丢**——排序能力其实没变差。
    """
    p_arr = np.asarray(p, dtype=float)
    n = p_arr.size
    if n == 0:
        return {"n_unique": 0, "unique_ratio": 0.0, "tie_rate_upper": 0.0, "auc_ceiling": 1.0}
    uniq, counts = np.unique(p_arr, return_counts=True)
    # 并列样本对占比的上界：sum over distinct values of (cnt^2 - cnt) / (n^2 - n)
    tied_pairs = float((counts.astype(float) ** 2 - counts).sum())
    total_pairs = float(n * (n - 1))
    tie_upper = tied_pairs / total_pairs if total_pairs > 0 else 0.0
    return {
        "n_unique": int(uniq.size),
        "unique_ratio": float(uniq.size / n),
        "tie_rate_upper": tie_upper,
        "auc_ceiling": float(1.0 - 0.5 * tie_upper),
    }


# ---------------------------------------------------------------------------
# 交叉验证
# ---------------------------------------------------------------------------
def repeated_grouped_cv(
    X: pd.DataFrame,
    y: Sequence[int],
    groups: Sequence[str],
    model_factory: Callable[[], object],
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 42,
) -> List[float]:
    """重复分层分组交叉验证，返回每一折的 AUC。

    **必须按 ``gpsno`` 分组**：多起点增广后同一台车有多个窗口样本，若不分组，
    同车的样本会同时出现在训练与验证折里，指标会虚高。
    """
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    X_arr = X.reset_index(drop=True)
    y_arr = np.asarray(y).astype(int)
    g_arr = np.asarray(groups).astype(str)

    aucs: List[float] = []
    for rep in range(int(n_repeats)):
        rs = int(seed) + rep
        n_groups = len(np.unique(g_arr))
        k = int(min(n_splits, max(n_groups, 2)))
        try:
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=rs)
            splits = list(splitter.split(X_arr, y_arr, groups=g_arr))
        except Exception:
            splitter = GroupKFold(n_splits=k)
            splits = list(splitter.split(X_arr, y_arr, groups=g_arr))

        for tr, te in splits:
            if len(np.unique(y_arr[tr])) < 2 or len(np.unique(y_arr[te])) < 2:
                continue
            model = model_factory()
            model.fit(X_arr.iloc[tr], y_arr[tr])
            proba = _positive_proba(model, X_arr.iloc[te])
            a = auc_score(y_arr[te], proba)
            if np.isfinite(a):
                aucs.append(a)
    return aucs


def _positive_proba(model: object, X: pd.DataFrame) -> np.ndarray:
    """统一取正类概率，兼容 predict_proba 与 decision_function 两类模型。"""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba)[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))
    return np.asarray(model.predict(X), dtype=float)


def summarize_cv(aucs: Sequence[float]) -> Dict[str, float]:
    """把逐折 AUC 汇总成可报告的统计量。"""
    arr = np.asarray([a for a in aucs if np.isfinite(a)], dtype=float)
    if arr.size == 0:
        return {"n_folds": 0, "mean": float("nan"), "std": float("nan"),
                "lo": float("nan"), "hi": float("nan"), "min": float("nan"), "max": float("nan")}
    se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
    return {
        "n_folds": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "lo": float(arr.mean() - 1.96 * se) if np.isfinite(se) else float("nan"),
        "hi": float(arr.mean() + 1.96 * se) if np.isfinite(se) else float("nan"),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def selection_bias_table(max_n: int = 200) -> pd.DataFrame:
    """「跑 N 次实验取最好」的期望虚高量。

    如果真实 AUC 的抽样标准误是 ``σ``，那么 N 次独立估计的最大值的期望约为
    ``μ + σ·E[max Z_N]``。以 ``σ = 0.031``（N=500、正例率 20%）为例：

    ==========  ==========
    实验次数 N   期望虚高
    ==========  ==========
    5           +0.035
    20          +0.056
    50          +0.067
    200         +0.082
    ==========  ==========

    也就是说，**什么都不做，跑 200 次也能「得到」0.80 的 CV AUC**。
    这正是保险箱（vault）+ 预注册 + 效应量门槛三道防线存在的理由。
    （实验之间通常正相关，会让实际偏差打折，但量级如此。）
    """
    # E[max of N standard normals] 的常用近似
    ns = [5, 10, 20, 50, 100, 200]
    rows = []
    for n in ns:
        if n > max_n:
            continue
        e_max = (1 - np.euler_gamma) * _norm_ppf(1 - 1.0 / n) + np.euler_gamma * _norm_ppf(
            1 - 1.0 / (n * np.e)
        )
        rows.append({"n_experiments": n, "expected_max_z": round(float(e_max), 3),
                     "inflation_at_se_0.031": round(float(e_max) * 0.031, 4)})
    return pd.DataFrame(rows)


def _norm_ppf(q: float) -> float:
    """标准正态分位数（避免额外依赖 scipy.stats，虽然 scipy 已装）。"""
    from scipy.stats import norm

    return float(norm.ppf(q))
