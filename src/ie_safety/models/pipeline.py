"""把「特征 → 训练 → 评估」串成一条可复用的流水线。

单独抽出来的原因：负对照与置换检验要求**重跑完整流水线**（含特征准备与模型
选择），而不是「换掉标签再跑一次固定模型」。只做后者测不出选择偏差，而那正是
最可能骗到自己的力量。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from .train import make_model_factory, prepare_matrix, rank_average
from .validation import _positive_proba, auc_score, summarize_cv

logger = logging.getLogger(__name__)


def _splits_for(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    n_groups = len(np.unique(groups))
    k = int(min(n_splits, max(n_groups, 2)))
    try:
        sp = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=int(seed))
        return list(sp.split(X, y, groups=groups))
    except Exception:
        sp = GroupKFold(n_splits=k)
        return list(sp.split(X, y, groups=groups))


def cross_val_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: Config,
    kinds: Sequence[str] = ("logistic", "lightgbm"),
    n_splits: Optional[int] = None,
    n_repeats: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """重复 K 折的**袋外预测**（out-of-fold）。

    对每个重复，把每一折的验证段预测拼回整表，得到一份完整 OOF 预测。集成在
    OOF 上做秩平均后统一评估 —— 这样集成的 AUC 是公平的，不会因为「用同一批
    数据既拟合又评估」而虚高。
    """
    v = cfg.validation
    n_splits = int(n_splits or v["n_splits"])
    n_repeats = int(n_repeats or v["n_repeats"])
    seed = int(seed if seed is not None else cfg.seed)

    Xp, _ = prepare_matrix(X)
    Xp = Xp.reset_index(drop=True)
    y_arr = np.asarray(y).astype(int)
    g_arr = np.asarray(groups).astype(str)

    oof = {k: np.full(len(y_arr), np.nan) for k in kinds}
    for rep in range(n_repeats):
        for tr, te in _splits_for(Xp, y_arr, g_arr, n_splits, seed + rep):
            if len(np.unique(y_arr[tr])) < 2:
                continue
            for kind in kinds:
                model = make_model_factory(cfg, kind)()
                model.fit(Xp.iloc[tr], y_arr[tr])
                oof[kind][te] = _positive_proba(model, Xp.iloc[te])
    return oof


def run_pipeline_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: Config,
    kinds: Sequence[str] = ("logistic", "lightgbm"),
    n_splits: Optional[int] = None,
    n_repeats: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """完整流水线：特征准备 → 重复分组 CV → 各模型与秩平均集成的 AUC。

    返回结构可直接序列化进报告。
    """
    v = cfg.validation
    n_splits = int(n_splits or v["n_splits"])
    n_repeats = int(n_repeats or v["n_repeats"])
    seed = int(seed if seed is not None else cfg.seed)

    oof = cross_val_oof(X, y, groups, cfg, kinds=kinds, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    y_arr = np.asarray(y).astype(int)

    out: Dict[str, object] = {"n_samples": int(len(y_arr)), "base_rate": float(y_arr.mean())}
    per_model: Dict[str, float] = {}
    for kind, p in oof.items():
        mask = np.isfinite(p)
        a = auc_score(y_arr[mask], p[mask])
        per_model[kind] = a
        out[kind] = {"oof_auc": a}

    if len(kinds) >= 2:
        mask = np.all([np.isfinite(oof[k]) for k in kinds], axis=0)
        ens = rank_average([oof[k][mask] for k in kinds])
        a_ens = auc_score(y_arr[mask], ens)
        out["ensemble"] = {"oof_auc": a_ens, "method": "rank_mean"}
    else:
        out["ensemble"] = {"oof_auc": per_model.get(kinds[0], float("nan")), "method": "single"}

    out["features"] = {"n_features": int(prepare_matrix(X)[0].shape[1])}
    return out


def pipeline_auc(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg: Config) -> float:
    """便捷入口：只取集成 OOF AUC（供负对照使用）。"""
    res = run_pipeline_cv(X, y, groups, cfg)
    return float(res["ensemble"]["oof_auc"])
