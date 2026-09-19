"""免费锚点校验 —— 无需标签就能验证「特征抽对了没有」。

这是一条被大多数人浪费掉的反馈通道。出题方在《IMU 数据风险分析参考资料》里
给出了六类 IMU 风险场景的**定量频次量级**（每 100 辆车每月）：

    急加速 / 急刹车      数千次
    急转弯 / 猛打方向    数百次
    疑似碰撞             < 100 次
    疑似侧翻             < 10 次

如果我们从原始 IMU 抽出来的事件频次与这些量级差了一两个数量级，那一定是阈值
算错了（阈值过高漏报、过低误报），与模型好坏无关。**先证明特征抽对了，再谈模型。**

同样的思路用在风险事件数据上：事件族的频次应当满足序关系

    疲劳 / 分心 / 超速   ≫   跟车 / 盲区   ≫   未遂事故   ≫   事故

如果序关系被违反（例如「未遂事故」比「急加速」还多），说明数据或标签定义出了
问题，此时应该停下来查数据，而不是继续调模型。

第三条是**符号约定**：训练完成后，各风险族的系数应为正。若某个族显著为负，
先怀疑混杂（典型是摄像头遮挡造成的删失 —— 设备被挡住的车事件数少，会被误判
成「安全」），**而不是庆祝模型发现了新规律**。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config


def check_frequency_order(events: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """校验事件族的频次序关系。

    返回 ``{"passed": bool, "family_counts": {...}, "violations": [...]}``。
    序关系取配置里的 ``audit.frequency_order``；未配置时使用内置默认。
    """
    counts = {
        fam: int(events["event_type"].isin(cfg.family_codes(fam)).sum())
        for fam in cfg.all_family_names()
    }
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(counts.values())

    spec = cfg.audit.get("frequency_order", {}) or {}
    higher = spec.get("higher", ["fatigue", "distraction", "speed"])
    lower = spec.get("lower", ["ultra"])

    violations: List[str] = []
    for h in higher:
        for l in lower:
            if h in counts and l in counts and counts[h] < counts[l]:
                violations.append(
                    f"频次序关系违反：{cfg.families[h]['name']}({counts[h]}) < "
                    f"{cfg.families[l]['name']}({counts[l]})"
                )

    return {
        "check": "event_family_frequency_order",
        "passed": len(violations) == 0,
        "family_counts": counts,
        "ordering_desc": " > ".join(f"{cfg.families[k]['name']}({v})" for k, v in ordered),
        "total_events": total,
        "expected_relation": f"{higher} >> {lower}",
        "violations": violations,
        "interpretation": (
            "序关系成立，事件数据与领域预期一致。"
            if not violations
            else "序关系被违反。先查数据与标签定义，不要继续调模型。"
        ),
    }


def check_imu_magnitude(
    derived_counts: Dict[str, int],
    n_vehicles: int,
    observation_days: float,
    cfg: Config,
) -> Dict[str, Any]:
    """把 IMU 派生事件数折算成「每 100 车每月」，与出题方给的量级对比。

    ``derived_counts`` 形如 ``{"harsh_brake": 1234, "sharp_turn": 88, ...}``。
    """
    bands = cfg.audit.get("imu_magnitude_per_100_vehicles_month", {}) or {}
    months = max(observation_days / 30.0, 1e-6)
    scale = (100.0 / max(n_vehicles, 1)) / months

    rows: List[Dict[str, Any]] = []
    violations: List[str] = []
    for key, count in derived_counts.items():
        band = bands.get(key)
        per100 = count * scale
        if not band:
            rows.append({"scene": key, "per_100_veh_month": round(per100, 1), "band": None, "ok": None})
            continue
        lo, hi = float(band[0]), float(band[1])
        ok = lo <= per100 <= hi
        rows.append(
            {
                "scene": key,
                "per_100_veh_month": round(per100, 1),
                "band": [lo, hi],
                "ok": ok,
                "ratio_to_band": round(per100 / max(hi, 1e-9), 4),
            }
        )
        if not ok:
            violations.append(
                f"{key}: 折算 {per100:.1f} 次/100车/月，超出出题方给的量级 [{lo}, {hi}]；"
                "阈值很可能算错了。"
            )

    return {
        "check": "imu_magnitude_anchor",
        "passed": len(violations) == 0,
        "n_vehicles": n_vehicles,
        "observation_days": round(observation_days, 1),
        "rows": rows,
        "violations": violations,
        "interpretation": (
            "IMU 派生事件的量级与参考文档一致，特征抽取可信。"
            if not violations
            else "量级明显偏离，先修阈值再谈模型。"
        ),
    }


def check_sign_convention(
    feature_matrix: pd.DataFrame, y: np.ndarray, cfg: Config
) -> Dict[str, Any]:
    """校验各风险族特征与标签的相关方向。

    只看**方向**不看大小：风险族与标签正相关是领域常识，负相关几乎必然意味着
    混杂（如摄像头遮挡造成的删失）而非真实规律。
    """
    label = pd.Series(np.asarray(y), index=feature_matrix.index)
    rows: List[Dict[str, Any]] = []
    violations: List[str] = []

    for fam in cfg.all_family_names():
        col = f"rate_per_1kkm_{fam}"
        if col not in feature_matrix.columns:
            col = f"count_{fam}"
        if col not in feature_matrix.columns:
            continue
        x = pd.to_numeric(feature_matrix[col], errors="coerce")
        mask = x.notna()
        if mask.sum() < 10 or label[mask].nunique() < 2:
            continue
        rho = float(pd.Series(x[mask]).corr(label[mask], method="spearman"))
        rows.append({"family": fam, "feature": col, "spearman": round(rho, 4)})
        if rho < -0.02:
            violations.append(
                f"{cfg.families[fam]['name']} 的 {col} 与标签呈负相关（ρ={rho:.3f}）："
                "先怀疑删失混杂（遮挡/离线导致事件被少记），不要当作真实规律。"
            )

    return {
        "check": "risk_family_sign_convention",
        "passed": len(violations) == 0,
        "rows": rows,
        "violations": violations,
    }


def summarize_anchor_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总锚点校验结论，供报告使用。"""
    return {
        "n_checks": len(results),
        "n_passed": sum(1 for r in results if r.get("passed")),
        "n_failed": sum(1 for r in results if not r.get("passed")),
        "failed_checks": [r.get("check") for r in results if not r.get("passed")],
        "results": results,
    }
