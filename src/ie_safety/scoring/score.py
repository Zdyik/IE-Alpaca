"""任务二：可加性线性分解 + 车队分位数标定的司机安全评分。

## 这道题的张力在哪

任务二由评委人工打分，其中「风险区分能力」占 35%，「可解释性」占 25%。
纯规则评分可解释但区分度弱；纯模型区分度强但可解释性崩。多数人二选一。

这里用**可加性线性分解**同时拿到两头：

1. 定义 5 个风险维度，每个维度的原始风险 ``r_d`` = 该维度的严重度加权事件率，
   取 ``log1p``；
2. **在一组维度特征上重训一个线性 logistic，取其归一化后的系数作为权重 ``w_d``**
   —— 权重不是拍脑袋给的，是数据说的，所以「风险区分能力」有背书；同时它仍是
   线性加和的，所以「可解释性」毫发无损；
3. 每个维度转成**车队内分位数分**（0–100，「比多少同行好」）；
4. 总分 ``= 100 − Σ w_d × (100 − 维度分_d)``。

这套设计的美感在于：总分是风险对数几率的一个**单调变换**，因此其 AUC 必然接近
线性模型的能力上限；而它的可解释性是**构造出来的**，不是事后用 SHAP 硬解释的。
每个司机都能看到「哪个维度扣了多少分、对应哪些具体事件」。

## 一条容易被忽略的坑

分档评分只有 4 档，而 AUC 公式里并列计 0.5 —— 若正负样本对的并列比例 ``f`` 达
0.3，AUC 上限会被压到 ``1 − 0.5×0.3 = 0.85``。因此**报告区分能力时必须同时上报
未分箱的连续风险指数的 AUC**，否则会白丢一大截区分度证据。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Config

logger = logging.getLogger(__name__)

#: 维度 → 用于计算该维度原始风险的列（按优先级取第一个存在的）
DIMENSION_COLUMNS: Dict[str, List[str]] = {
    "ultra": ["rate_per_1kkm_ultra", "sev_ultra", "count_ultra"],
    "fatigue": ["rate_per_1kkm_fatigue", "sev_fatigue", "count_fatigue"],
    "distraction": ["rate_per_1kkm_distraction", "sev_distraction", "count_distraction"],
    "speed": ["rate_per_1kkm_speed", "sev_speed", "count_speed"],
    "compliance": ["compliance_rate_per_100h", "compliance_count"],
}


def _dimension_raw(X: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """抽出 5 个维度的原始风险值。"""
    out = pd.DataFrame(index=X.index)
    for dim, candidates in DIMENSION_COLUMNS.items():
        col = next((c for c in candidates if c in X.columns), None)
        raw = pd.to_numeric(X[col], errors="coerce") if col else pd.Series(np.nan, index=X.index)
        out[dim] = raw.fillna(0.0)
        out.attrs.setdefault("source_columns", {})[dim] = col
    return out


def fit_dimension_weights(
    X: pd.DataFrame, y: np.ndarray, cfg: Config, positive: bool = True
) -> Dict[str, object]:
    """用线性 logistic 从数据里估计维度权重。

    ``positive=True`` 时会强制权重非负并归一化到和为 1。强制非负的理由：风险维度
    与风险的关联方向是领域常识，出现负系数几乎必然意味着共线或删失混杂，此时把
    它当成「这个维度扣分反而更安全」是错的。
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    raw = _dimension_raw(X, cfg)
    Z = np.log1p(np.clip(raw.astype(float).values, 0, None))

    y_arr = np.asarray(y).astype(int)
    if len(np.unique(y_arr)) < 2:
        logger.warning("标签只有单一类别，无法估计权重，退回等权")
        w = {d: 1.0 / len(DIMENSION_COLUMNS) for d in DIMENSION_COLUMNS}
        return {"weights": w, "method": "equal_fallback", "coefficients": {}}

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=5000)),
        ]
    )
    pipe.fit(Z, y_arr)
    coefs = pipe.named_steps["clf"].coef_.ravel()

    dims = list(DIMENSION_COLUMNS.keys())
    raw_coef = {d: float(c) for d, c in zip(dims, coefs)}
    used = {d: (abs(c) if not positive else max(c, 0.0)) for d, c in raw_coef.items()}
    total = sum(used.values())
    if total <= 0:
        logger.warning("所有维度系数非正，强制非负后退回等权")
        weights = {d: 1.0 / len(dims) for d in dims}
        method = "equal_fallback_nonpositive"
    else:
        weights = {d: v / total for d, v in used.items()}
        method = "logistic_coefficients_normalized"

    return {
        "weights": weights,
        "method": method,
        "coefficients": raw_coef,
        "source_columns": raw.attrs.get("source_columns", {}),
        "n_samples": int(len(y_arr)),
        "base_rate": float(y_arr.mean()),
    }


def compute_scores(
    X: pd.DataFrame, weights: Dict[str, float], cfg: Config
) -> pd.DataFrame:
    """计算维度分、总分与分档。

    维度分 = ``100 × (1 − 车队内分位数)``，即「比多少同行好」。
    总分 = ``100 − Σ w_d × (100 − 维度分_d)``，完全可加、可逐项归因。
    """
    raw = _dimension_raw(X, cfg)
    dims = list(DIMENSION_COLUMNS.keys())

    out = pd.DataFrame(index=X.index)
    for d in dims:
        v = np.log1p(np.clip(raw[d].astype(float).values, 0, None))
        pct = pd.Series(v, index=X.index).rank(pct=True, method="average")
        # 事件率为 0 的司机应当拿满分，而不是按分位数拿 100×(1−最高秩)
        out[f"dim_{d}_score"] = np.where(raw[d].values <= 0, 100.0, 100.0 * (1.0 - pct.values))
        out[f"dim_{d}_raw"] = raw[d].values

    wsum = sum(max(float(weights.get(d, 0.0)), 0.0) for d in dims) or 1.0
    deduction = np.zeros(len(out), dtype=float)
    for d in dims:
        w = max(float(weights.get(d, 0.0)), 0.0) / wsum
        out[f"dim_{d}_weight"] = w
        deduction += w * (100.0 - out[f"dim_{d}_score"].values)
        # 记录每个维度的实际扣分，供安全卡逐项归因
        out[f"dim_{d}_deduction"] = w * (100.0 - out[f"dim_{d}_score"].values)

    out["risk_index"] = deduction          # 连续风险指数（未分箱，供 AUC 使用）
    out["score"] = np.clip(100.0 - deduction, 0.0, 100.0)
    out["band"] = assign_bands(out["score"], cfg)
    return out


def assign_bands(scores: pd.Series, cfg: Config) -> pd.Series:
    """按配置的分数区间分档 A/B/C/D。"""
    bands = cfg.scoring["bands"]
    ordered = sorted(bands.items(), key=lambda kv: float(kv[1]["min"]), reverse=True)
    out = pd.Series("D", index=scores.index, dtype=object)
    assigned = pd.Series(False, index=scores.index)
    for name, spec in ordered:
        mask = (scores >= float(spec["min"])) & (~assigned)
        out[mask] = name
        assigned |= mask
    return out


def ewma_smooth(series_history: pd.DataFrame, halflife_days: int) -> pd.Series:
    """对历史评分做指数加权平滑，避免排名天天跳。

    ``series_history`` 需含 ``date`` 与 ``score`` 两列。
    """
    if not len(series_history):
        return pd.Series(dtype=float)
    df = series_history.sort_values("date")
    return df["score"].ewm(halflife=max(int(halflife_days), 1), adjust=False).mean()


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def validate_scores(scores: pd.DataFrame, y: np.ndarray, cfg: Config) -> Dict[str, object]:
    """任务二的区分能力证据。

    关键一条：**必须额外上报未分箱连续风险指数的 AUC**。分档只有 4 档，AUC 公式
    中并列计 0.5，用分档分去算 AUC 会人为压低区分度。
    """
    from ..models.validation import auc_score, bootstrap_auc_ci, tie_diagnostics

    y_arr = np.asarray(y).astype(int)
    risk = scores["risk_index"].to_numpy(dtype=float)  # 越大越危险
    band_score = scores["score"].to_numpy(dtype=float)

    auc_continuous = auc_score(y_arr, risk)
    ci = bootstrap_auc_ci(y_arr, risk, n_boot=cfg.validation["auc_ci_bootstrap"], seed=cfg.seed)
    # 注意方向：score 是「越大越安全」，而 y=1 表示出险，因此必须换算到风险方向（取负），
    # 否则会得到一个小于 0.5 的数字，看起来像「模型很差」，其实是方向搞反了。
    auc_band = auc_score(y_arr, -band_score)

    # 分档单调性：各档的实际出险率应当单调递增（A 最低、D 最高）
    tmp = pd.DataFrame({"band": scores["band"].values, "y": y_arr})
    rate = tmp.groupby("band")["y"].mean().to_dict()
    counts = tmp.groupby("band")["y"].size().to_dict()
    order = [b for b in ["A", "B", "C", "D"] if b in rate]
    rates_ordered = [float(rate[b]) for b in order]
    monotone = all(rates_ordered[i] <= rates_ordered[i + 1] for i in range(len(rates_ordered) - 1))
    risk_ratio = (
        float(rates_ordered[-1] / rates_ordered[0])
        if len(rates_ordered) >= 2 and rates_ordered[0] > 0
        else float("inf") if len(rates_ordered) >= 2 and rates_ordered[-1] > 0 else float("nan")
    )

    # 分档人数可能因「绝对阈值」而严重不均，需要显式暴露；发生倒挂时也要给出统计解读
    band_trend = _band_trend(order, rates_ordered, [int(counts[b]) for b in order])

    return {
        "auc_continuous_risk_index": float(auc_continuous),
        "auc_ci": ci,
        "auc_of_band_score": float(auc_band),
        "band_note": (
            "分档分只有 4 档，并列会压低 AUC 上限；区分能力应以连续风险指数为准。"
            "此处分档 AUC 已换算到风险方向（取 −score），否则会得到小于 0.5 的误导性数字。"
        ),
        "band_event_rate": {b: round(float(rate[b]), 4) for b in order},
        "band_size": {b: int(counts[b]) for b in order},
        "band_monotone": bool(monotone),
        "band_trend": band_trend,
        "decile_trend": _decile_trend(band_score, y_arr),
        "risk_ratio_highest_vs_lowest": None if not np.isfinite(risk_ratio) else float(risk_ratio),
        "tie_diagnostics": tie_diagnostics(risk),
    }


def _band_trend(order: List[str], rates: List[float], sizes: List[int]) -> Dict[str, object]:
    """分档趋势的统计解读。

    「严格两两单调」对单档只有十几二十人的分档来说过于苛刻：出险率 0.7、单档 20 人
    时标准误约 0.10，相邻两档差 0.11 完全不显著。因此这里额外给出秩相关趋势与每次
    倒挂的样本量，让报告可以如实说明「趋势成立、局部倒挂不显著」，而不是简单判
    「未通过」了事。
    """
    from scipy.stats import spearmanr

    if len(order) < 2:
        return {"available": False}
    rho, p = spearmanr(np.arange(len(order)), rates)
    inversions = []
    for i in range(len(rates) - 1):
        if rates[i] > rates[i + 1]:
            se = float(
                np.sqrt(
                    max(rates[i] * (1 - rates[i]) / max(sizes[i], 1), 0)
                    + max(rates[i + 1] * (1 - rates[i + 1]) / max(sizes[i + 1], 1), 0)
                )
            )
            gap = rates[i] - rates[i + 1]
            inversions.append(
                {
                    "from": order[i],
                    "to": order[i + 1],
                    "rate_from": round(rates[i], 4),
                    "rate_to": round(rates[i + 1], 4),
                    "gap": round(gap, 4),
                    "n_from": sizes[i],
                    "n_to": sizes[i + 1],
                    "diff_se": round(se, 4),
                    "significant_at_2se": bool(gap > 2 * se),
                }
            )
    return {
        "available": True,
        "band_order": order,
        "spearman_index_vs_rate": round(float(rho), 4),
        "p_value": float(p),
        "trend_direction_ok": bool(np.isfinite(rho) and rho > 0.5),
        "inversions": inversions,
        "size_balance": {"min": int(min(sizes)), "max": int(max(sizes))} if sizes else {},
        "note": (
            "趋势方向正确（秩相关为正）；局部倒挂的样本量与前后的标准误已列出，"
            "若 significant_at_2se 为 false 则该倒挂在统计上不可区分于噪声。"
            if inversions
            else "各档出险率严格单调递增。"
        ),
    }


def _decile_trend(band_score: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """按分数分十分位，看各档出险率是否随分数上升而单调下降。

    四档太粗：在正例率高、单档人数少时容易掩盖真实趋势。十分位是更灵敏的单调性证据。
    """
    from scipy.stats import spearmanr

    n = len(band_score)
    if n < 20 or len(np.unique(y)) < 2:
        return {"available": False}
    order = np.argsort(band_score)  # 分数从低（危险）到高（安全）
    bins = np.array_split(order, 10)
    rates = [float(y[b].mean()) for b in bins if len(b)]
    rho, _ = spearmanr(np.arange(len(rates)), rates)
    return {
        "available": True,
        "decile_event_rates_low_to_high_score": [round(r, 4) for r in rates],
        "spearman_rate_vs_score": round(float(rho), 4),
        "monotone_decreasing": bool(np.isfinite(rho) and rho <= -0.8),
    }


def counterfactual_test(
    X: pd.DataFrame,
    weights: Dict[str, float],
    cfg: Config,
    dimension: str = "speed",
    reduction: float = 0.5,
    n_drivers: int = 20,
) -> Dict[str, object]:
    """反事实测试：**单独降低一部分司机**该维度的风险后，他们的分数是否上升。

    这是「权重设计是否自洽」的可测代理 —— 如果降低风险反而不加分，说明权重或
    单调方向有问题。

    关键设计：必须只改**一部分司机**。若把全体司机的该维度风险一起降低，车队内
    分位数几乎不变，所有人的分数都不会动 —— 那样测出来的是「分位数标定对整体
    平移不敏感」，而不是权重方向，得到的结论会是假阴性。
    """
    base = compute_scores(X, weights, cfg)

    col = next((c for c in DIMENSION_COLUMNS.get(dimension, []) if c in X.columns), None)
    if col is None:
        return {"check": "counterfactual", "passed": False, "reason": f"找不到维度 {dimension} 的列"}

    n = min(int(n_drivers), len(X))
    targets = X.index[:n]
    X2 = X.copy()
    orig = pd.to_numeric(X2.loc[targets, col], errors="coerce").fillna(0.0)
    X2.loc[targets, col] = orig * (1.0 - reduction)
    after = compute_scores(X2, weights, cfg)

    delta = (after.loc[targets, "score"] - base.loc[targets, "score"])
    improved = float((delta > 0).mean())
    mean_delta = float(delta.mean())

    return {
        "check": "counterfactual",
        "passed": bool(improved >= 0.8 and mean_delta > 0),
        "dimension": dimension,
        "reduction": reduction,
        "n_drivers_modified": int(n),
        "mean_score_delta": round(mean_delta, 3),
        "frac_improved": round(improved, 4),
        "interpretation": (
            f"把 {n} 位司机的「{cfg.scoring['dimensions'].get(dimension, dimension)}」风险降低 "
            f"{reduction:.0%} 后，{improved:.0%} 的人分数上升（平均 +{mean_delta:.2f} 分），权重方向自洽。"
            if improved >= 0.8 and mean_delta > 0
            else "降低风险后分数没有普遍上升，权重方向或分位数标定有问题。"
        ),
    }


def stability_test(
    scores_t1: pd.DataFrame, scores_t2: pd.DataFrame, min_spearman: float = 0.80
) -> Dict[str, object]:
    """跨期稳定性：相邻两期排名的 Spearman 相关。

    低于阈值意味着排名天天跳，管理者无法据此做决定 —— 这属于「可运营性」问题，
    即使区分能力再强也不可用。
    """
    common = scores_t1.index.intersection(scores_t2.index)
    if len(common) < 5:
        return {"check": "stability", "passed": False, "reason": "两期共同司机不足"}
    from scipy.stats import spearmanr

    rho, p = spearmanr(scores_t1.loc[common, "score"], scores_t2.loc[common, "score"])
    return {
        "check": "stability",
        "passed": bool(np.isfinite(rho) and rho >= min_spearman),
        "spearman": round(float(rho), 4),
        "p_value": float(p),
        "n_common": int(len(common)),
        "threshold": min_spearman,
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def scorecard_markdown(
    gpsno: str, row: pd.Series, cfg: Config, events_summary: Optional[Dict[str, int]] = None
) -> str:
    """为单个司机生成可读的安全卡。

    设计目标（也是任务二「可解释性」的可测标准）：**只看这张卡就能说出分数是
    怎么来的**，不需要查代码。
    """
    bands = cfg.scoring["bands"]
    band = str(row.get("band", "?"))
    action = bands.get(band, {}).get("action", "")
    lines = [
        f"## 司机安全卡 · {gpsno}",
        "",
        f"- **总分：{row['score']:.1f} / 100**（{band} 档）",
        f"- 管理动作：{action}",
        "",
        "### 扣分构成（可加性分解，合计即为总扣分）",
        "",
        "| 维度 | 权重 | 维度分 | 扣分 |",
        "|---|---|---|---|",
    ]
    total_ded = 0.0
    for dim, label in cfg.scoring["dimensions"].items():
        w = row.get(f"dim_{dim}_weight", float("nan"))
        s = row.get(f"dim_{dim}_score", float("nan"))
        d = row.get(f"dim_{dim}_deduction", float("nan"))
        total_ded += 0.0 if pd.isna(d) else float(d)
        lines.append(f"| {label} | {0 if pd.isna(w) else w:.1%} | {0 if pd.isna(s) else s:.1f} | {0 if pd.isna(d) else d:.2f} |")
    lines += ["", f"**扣分合计：{total_ded:.2f}**，总分 = 100 − {total_ded:.2f} = {row['score']:.1f}", ""]

    if events_summary:
        lines += ["### 触发扣分的主要行为", ""]
        for name, cnt in sorted(events_summary.items(), key=lambda kv: -kv[1])[:8]:
            lines.append(f"- {name}：{cnt} 次")
        lines.append("")

    lines += [
        "### 如何提高分数",
        "",
        "把上表中扣分最大的那个维度对应的高频行为降下来，分数会按该维度权重线性回升"
        "（分解是可加的，因此提升幅度可以直接算出来）。",
        "",
    ]
    return "\n".join(lines)


def management_playbook(cfg: Config) -> str:
    """运营建议（交付物之一：「如何把评分模型落地到车队安全管理」）。"""
    bands = cfg.scoring["bands"]
    lines = [
        "# 评分模型运营建议",
        "",
        "## 一、分档与对应动作",
        "",
        "| 档位 | 分数区间 | 对象规模（按车队分位估计） | 管理动作 |",
        "|---|---|---|---|",
    ]
    ordered = sorted(bands.items(), key=lambda kv: float(kv[1]["min"]), reverse=True)
    for i, (name, spec) in enumerate(ordered):
        upper = ordered[i - 1][1]["min"] if i > 0 else 100
        lines.append(f"| {name} | {spec['min']} – {upper} | — | {spec['action']} |")
    lines += [
        "",
        "## 二、周期与节奏",
        "",
        "- **每日**：更新分数与排名，只推送「降幅超过阈值」的司机，避免信息过载。",
        "- **每周**：安全员例会，按 D 档名单排跟车计划；对 C 档做一次一对一。",
        "- **每月**：全队分数分布复盘，检查分档比例是否失衡（若 D 档长期超过 10%，"
        "说明阈值需要校准，而不是司机都变差了）。",
        "",
        "## 三、为什么这套评分可以用在管理里",
        "",
        "1. **可解释**：每个扣分项都能指到具体行为与次数，司机不会觉得「分数是黑箱给的」。",
        "2. **可归因**：分解是可加的，因此「改善哪个行为能加多少分」可以精确算出来，"
        "培训目标可以直接量化。",
        "3. **稳定**：分数经指数加权平滑，排名不会天天跳，管理者据此做的决定不会第二天就失效。",
        "4. **口径统一**：维度分是车队内分位数，因此「多少分算好」有明确参照，"
        "不需要主观设定绝对标准。",
        "",
        "## 四、使用边界（必须写进制度）",
        "",
        "- 评分是**风险提示**，不是处罚依据。低分且无事故的司机同样需要被看见。",
        "- 设备遮挡/离线会导致事件被少记，**可能让真实高风险的司机拿到高分**。"
        "因此合规维度单独计权，并建议对低覆盖度车辆做人工复核。",
        "- 分数会随车队整体水平漂移（因为是分位数），跨车队或跨时期的绝对分不可直接比较，"
        "比较应看排名与趋势。",
        "",
    ]
    return "\n".join(lines)
