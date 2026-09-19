"""任务一：基线、模型、校准、保险箱。

## 训练信号是什么

标签 = 结果窗口内是否出现 ``11803`` / ``11804``，用**二元 logloss** 训练。
两个刻意的选择：

* **不用 ``scale_pos_weight`` 去「调准」。** AUC 与阈值无关，调类别权重不会提升
  AUC，只会破坏概率校准 —— 而交付要求输出概率值。
* **不调 ``pred_label`` 的阈值。** AUC 是阈值无关的，该列只为满足交付格式，
  按训练基率取值并写明即可。把时间花在调阈值上不会让 AUC 动一分。

## 模型选择

* **逻辑回归**（L2，输入 ``log1p`` + 标准化）：在 N=500 上最稳，系数直接可读，
  也是任务二权重的来源。
* **LightGBM 小容量**：``num_leaves 8–15``、``max_depth 3``、
  ``min_child_samples 25``、强 L1/L2。刻意压容量，因为 500 行样本上任何深树
  学到的都是噪声。它原生支持缺失值，所以 IMU 覆盖不全时不需要插补。
* **rank 平均集成**：AUC 只关心序，秩平均比概率平均更稳健。

## 保险箱

``t = 20``（真正的 20+40 切分）的 500 条样本**全程锁死**，只在最终评估时开一次。
理由见 :func:`ie_safety.models.validation.selection_bias_table`：跑 200 组实验
挑最好的，CV 期望虚高约 +0.08 —— 什么都不做也能「得到」0.80。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from .validation import _positive_proba, auc_score, bootstrap_auc_ci, summarize_cv

logger = logging.getLogger(__name__)

#: 这些列名不参与建模（元信息 / 标识）
NON_FEATURE_COLUMNS = {"origin_day", "gpsno"}


# ---------------------------------------------------------------------------
# 特征矩阵准备
# ---------------------------------------------------------------------------
def select_feature_columns(X: pd.DataFrame) -> List[str]:
    """挑出数值型特征列。"""
    cols = []
    for c in X.columns:
        if c in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(X[c]):
            cols.append(c)
    return cols


def prepare_matrix(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """对重尾的非负计数/速率列做 ``log1p``，其余保持不变。

    只对「非负且量级跨度大」的列做变换：``log_rate_ratio_late_early`` 这类可正
    可负的列必须原样保留，``share_*`` 这类 0–1 的列做不做都一样。
    """
    Xn = X[select_feature_columns(X)].copy()
    transformed: List[str] = []
    for c in Xn.columns:
        s = pd.to_numeric(Xn[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        vmin, vmax = float(s.min()), float(s.max())
        if vmin >= 0 and vmax > 10:
            Xn[c] = np.log1p(s)
            transformed.append(c)
    return Xn, transformed


# ---------------------------------------------------------------------------
# 模型工厂
# ---------------------------------------------------------------------------
def make_logistic(cfg: Config):
    """逻辑回归管道：中位数插补 → 标准化 → L2 逻辑回归。"""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    p = cfg.models["logistic"]
    # 注意：scikit-learn 1.8 起 penalty= 已废弃，默认即 L2，因此这里不再显式传。
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(p.get("C", 1.0)),
                    max_iter=int(p.get("max_iter", 5000)),
                    class_weight=None,  # 刻意不用：AUC 与阈值无关，调权重只破坏校准
                ),
            ),
        ]
    )


def make_lightgbm(cfg: Config):
    """小容量 LightGBM。"""
    from lightgbm import LGBMClassifier

    p = dict(cfg.models["lightgbm"])
    return LGBMClassifier(
        n_estimators=int(p.get("n_estimators", 400)),
        learning_rate=float(p.get("learning_rate", 0.03)),
        num_leaves=int(p.get("num_leaves", 12)),
        max_depth=int(p.get("max_depth", 3)),
        min_child_samples=int(p.get("min_child_samples", 25)),
        subsample=float(p.get("subsample", 0.8)),
        subsample_freq=int(p.get("subsample_freq", 1)),
        colsample_bytree=float(p.get("colsample_bytree", 0.6)),
        reg_alpha=float(p.get("reg_alpha", 0.5)),
        reg_lambda=float(p.get("reg_lambda", 1.0)),
        random_state=int(cfg.seed),
        n_jobs=1,
        verbose=-1,
    )


def make_model_factory(cfg: Config, kind: str):
    if kind == "logistic":
        return lambda: make_logistic(cfg)
    if kind == "lightgbm":
        return lambda: make_lightgbm(cfg)
    raise ValueError(f"未知模型类型: {kind}")


# ---------------------------------------------------------------------------
# 基线
# ---------------------------------------------------------------------------
def baseline_b1(X: pd.DataFrame) -> np.ndarray:
    """B1 基线：仅用「特征窗口内的事故 + 未遂事故计数」。

    这几乎必然是最强的单特征（行为复发效应），因此它是**必须超越的对象**。
    任何「改进」若连它都赢不了，就不算改进。
    """
    cols = [c for c in ("count_code_11804", "count_code_11803") if c in X.columns]
    if not cols:
        return np.zeros(len(X))
    return X[cols].fillna(0.0).sum(axis=1).to_numpy(dtype=float)


def baseline_b2(X: pd.DataFrame) -> np.ndarray:
    """B2 基线：仅用车辆画像特征（长期行为画像，不含任何事件信息）。"""
    cols = [c for c in X.columns if c.startswith("profile_") or c.startswith("energy_")]
    if not cols:
        return np.zeros(len(X))
    return X[cols].fillna(0.0).mean(axis=1).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# 集成与校准
# ---------------------------------------------------------------------------
def rank_average(probabilities: Sequence[np.ndarray], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """秩平均集成：AUC 只关心序，秩平均比概率平均稳健。"""
    from scipy.stats import rankdata

    ranks = [rankdata(np.asarray(p, dtype=float)) for p in probabilities]
    w = np.asarray(weights, dtype=float) if weights is not None else np.ones(len(ranks))
    w = w / w.sum()
    out = np.zeros_like(ranks[0], dtype=float)
    for wi, r in zip(w, ranks):
        out += wi * (r / r.size)
    return out


def platt_calibrate(p_train: np.ndarray, y_train: np.ndarray, p_test: np.ndarray) -> np.ndarray:
    """Platt 校准（在 logit 上做一维逻辑回归）。

    500 行样本上**优先 Platt 而不是 Isotonic**：Isotonic 是分段常数，在小样本上
    容易过拟合，还会制造大量并列值从而压低 AUC 上限。
    """
    from sklearn.linear_model import LogisticRegression

    eps = 1e-6
    p_tr = np.clip(np.asarray(p_train, dtype=float), eps, 1 - eps)
    p_te = np.clip(np.asarray(p_test, dtype=float), eps, 1 - eps)
    z_tr = np.log(p_tr / (1 - p_tr)).reshape(-1, 1)
    z_te = np.log(p_te / (1 - p_te)).reshape(-1, 1)

    if len(np.unique(y_train)) < 2:
        return p_test
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(z_tr, np.asarray(y_train).astype(int))
    return lr.predict_proba(z_te)[:, 1]


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def evaluate_predictions(y: Sequence[int], predictions: Dict[str, np.ndarray], seed: int = 42,
                         n_boot: int = 1000) -> Dict[str, Dict[str, float]]:
    """对若干候选预测同时算 AUC + 置信区间。"""
    out: Dict[str, Dict[str, float]] = {}
    for name, p in predictions.items():
        out[name] = bootstrap_auc_ci(y, p, n_boot=n_boot, seed=seed)
    return out


# ---------------------------------------------------------------------------
# 保险箱
# ---------------------------------------------------------------------------
class Vault:
    """保险箱：锁住 ``t=20`` 的主样本，只在最终评估时开一次。

    每次解锁都会追加一条审计记录到 ``reports/vault_audit.jsonl``。如果审计记录
    里已经有历史条目，说明保险箱**已经被开过**，此时再报告的数字不再是无偏的 ——
    这不是可以忽略的形式主义，而是「跑 200 次取最好会虚高 0.08」的直接对策。
    """

    def __init__(self, audit_path: Path) -> None:
        self.audit_path = Path(audit_path)

    def history(self) -> List[Dict[str, object]]:
        if not self.audit_path.exists():
            return []
        rows = []
        with open(self.audit_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def is_unlocked(self) -> bool:
        return len(self.history()) > 0

    def unlock(self, note: str = "", payload: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        rec = {
            "unlocked_at": datetime.now(timezone.utc).isoformat(),
            "note": note,
            "previous_unlocks": len(self.history()),
            "payload": payload or {},
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if rec["previous_unlocks"] > 0:
            logger.warning(
                "保险箱此前已被打开 %d 次，本次结果不再是无偏估计。", rec["previous_unlocks"]
            )
        return rec
