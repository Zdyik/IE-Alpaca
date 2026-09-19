"""05 · 任务二：可加性司机安全评分、档位、安全卡与运营建议。

评分模型的设计要点（详见 `docs/04_任务二评分模型说明.md`）：

* 5 个风险维度，原始风险取 ``log1p(严重度加权事件率)``；
* **维度权重由线性 logistic 从数据估计**（不是拍脑袋），因此「风险区分能力」有
  背书；同时仍是线性加和，因此「可解释性」毫发无损；
* 维度分 = 车队内分位数（「比多少同行好」）；
* 总分 = ``100 − Σ w_d × (100 − 维度分_d)``，完全可加、可逐项归因。

产出（**含车辆级信息的文件一律不入仓库**）：

* ``outputs/task2_scores.csv``         —— 提交物（总分 / 档位 / 各维度分）
* ``outputs/task2_scorecard.md``       —— 每位司机一张可读安全卡
* ``reports/task2_validation.md``      —— 区分能力、单调性、稳定性、反事实
* ``docs/05_运营建议.md``              —— 如何落地到车队管理

用法::

    python scripts/05_task2.py [--raw ...]
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from _common import (
    find_raw_dir,
    get_config,
    load_daily_tables,
    markdown_table,
    setup_logging,
    synthetic_notice,
    write_text_lf,
)

from ie_safety.features.build import build_features
from ie_safety.io import discover_datasets, load_profile
from ie_safety.models.dataset import build_window_dataset, make_augmented_dataset, make_window
from ie_safety.models.validation import tie_diagnostics
from ie_safety.scoring import (
    compute_scores,
    counterfactual_test,
    fit_dimension_weights,
    management_playbook,
    scorecard_markdown,
    stability_test,
    validate_scores,
)

logger = logging.getLogger("task2")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="任务二：司机安全评价模型")
    ap.add_argument("--raw", default=None)
    args = ap.parse_args(argv)
    setup_logging()
    cfg = get_config()

    raw_dir = find_raw_dir(cfg, args.raw)
    found = discover_datasets(raw_dir)
    profile = load_profile(found["profile"])
    tables = load_daily_tables(cfg)
    exp, dev, imu = tables["exposure"], tables["events"], tables["imu"]

    import json

    decision = json.loads((cfg.resolve("reports") / "label_decision.json").read_text(encoding="utf-8"))
    label_codes, outcome_days = decision["label_codes"], int(decision["outcome_days"])

    vehicles = sorted(profile["gpsno"].unique().tolist())
    t0 = dev["date"].min()

    # ------------------------------------------------------------------
    # 1) 用带标签的开发样本估计维度权重
    # ------------------------------------------------------------------
    X_dev, y_dev, _ = make_augmented_dataset(
        exp, dev, imu, profile, cfg, t0, cfg.dev_origins, label_codes, vehicles, outcome_days
    )
    fit = fit_dimension_weights(X_dev, y_dev.values, cfg)
    logger.info("维度权重估计方法：%s", fit["method"])
    for d, w in fit["weights"].items():
        logger.info("  %-12s 权重 %.4f（系数 %+.4f）", d, w, fit["coefficients"].get(d, float("nan")))

    # ------------------------------------------------------------------
    # 2) 在完整观测窗口上给所有司机打分（交付要求：对数据集内所有司机输出评分）
    # ------------------------------------------------------------------
    win_full = make_window(t0, cfg.observation_days - outcome_days, outcome_days, cfg.observation_days)
    X_sub = build_features(
        daily_exposure=exp, daily_events=dev, profile=profile, cfg=cfg,
        window_start=t0, window_end=t0 + pd.Timedelta(days=cfg.observation_days),
        vehicles=vehicles, daily_imu=imu,
    )
    scores = compute_scores(X_sub, fit["weights"], cfg)
    logger.info("评分完成：%d 台车，均值 %.1f，分档 %s",
                len(scores), float(scores["score"].mean()),
                scores["band"].value_counts().to_dict())

    # ------------------------------------------------------------------
    # 3) 校验
    # ------------------------------------------------------------------
    X_v, y_v = build_window_dataset(exp, dev, imu, profile, cfg, t0, win_full, label_codes, vehicles)
    scores_v = compute_scores(X_v, fit["weights"], cfg)
    val = validate_scores(scores_v, y_v.values, cfg)
    logger.info("连续风险指数 AUC = %.4f（分档分 AUC = %.4f）",
                val["auc_continuous_risk_index"], val["auc_of_band_score"])

    # 跨期稳定性：两个不同长度窗口的评分排名是否一致
    win_early = make_window(t0, max(cfg.feature_days // 2, 5), outcome_days, cfg.observation_days)
    X_e, _ = build_window_dataset(exp, dev, imu, profile, cfg, t0, win_early, label_codes, vehicles)
    scores_e = compute_scores(X_e, fit["weights"], cfg)
    stab = stability_test(scores_e, scores_v)

    cf = counterfactual_test(X_v, fit["weights"], cfg, dimension="speed", reduction=0.5)
    logger.info("反事实测试：%s", "通过" if cf.get("passed") else "未通过")

    # ------------------------------------------------------------------
    # 4) 输出
    # ------------------------------------------------------------------
    out_cols = ["score", "band", "risk_index"] + [
        c for c in scores.columns if c.startswith("dim_") and c.endswith("_score")
    ] + [c for c in scores.columns if c.startswith("dim_") and c.endswith("_deduction")]
    out_csv = cfg.resolve("outputs", ensure=True) / "task2_scores.csv"
    scores[out_cols].reset_index().rename(columns={"index": "gpsno"}).to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )
    logger.info("评分结果已写入 %s", out_csv)

    # 安全卡：为每台车生成一张，另存前几张作为文档示例
    name_of = cfg.code_catalog
    cards = []
    for gpsno in scores.index:
        ev_summary = {}
        for code, spec in name_of.items():
            col = f"c_{code}"
            if col in X_sub.columns:
                v = float(pd.to_numeric(X_sub.loc[gpsno, col], errors="coerce") or 0)
                if v > 0:
                    ev_summary[spec["name"]] = int(v)
        cards.append(scorecard_markdown(str(gpsno), scores.loc[gpsno], cfg, ev_summary))
    card_path = cfg.resolve("outputs", ensure=True) / "task2_scorecard.md"
    write_text_lf(card_path, "\n---\n\n".join(cards))
    logger.info("安全卡已写入 %s（%d 张）", card_path, len(cards))

    # ------------------------------------------------------------------
    # 5) 报告
    # ------------------------------------------------------------------
    notice = synthetic_notice(cfg, raw_dir)
    reports = cfg.resolve("reports", ensure=True)
    lines = [
        "# 任务二验证报告",
        "",
        notice,
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. 权重来源（可解释性的基础）",
        "",
        f"- 估计方法：`{fit['method']}`（在一组维度特征上重训线性 logistic，取归一化系数）",
        f"- 训练样本：{fit['n_samples']} 条，基率 {fit['base_rate']:.4f}",
        "",
        markdown_table([
            {"维度": cfg.scoring["dimensions"].get(d, d), "原始系数": round(fit["coefficients"].get(d, float("nan")), 4),
             "归一化权重": round(w, 4)}
            for d, w in fit["weights"].items()
        ]),
        "",
        "> 强制权重非负：风险维度与风险的关联方向是领域常识，出现负系数几乎必然意味着",
        "> 共线或删失混杂（典型是摄像头遮挡导致的事件少记），此时把它当成「扣分反而更安全」是错的。",
        "",
        "## 2. 区分能力（对应评审 35%）",
        "",
        f"- **连续风险指数 AUC = {val['auc_continuous_risk_index']:.4f}**"
        f"，95% CI [{val['auc_ci']['lo']:.4f}, {val['auc_ci']['hi']:.4f}]",
        f"- 分档分 AUC = {val['auc_of_band_score']:.4f}",
        f"- {val['band_note']}",
        "",
        f"- 并列诊断：唯一风险指数 {val['tie_diagnostics']['n_unique']} 个，"
        f"AUC 上限 {val['tie_diagnostics']['auc_ceiling']:.4f}",
        "",
        "## 3. 分档单调性（对应评审「可运营性」）",
        "",
        markdown_table([
            {"档位": b, "人数": val["band_size"].get(b, 0), "实际出险率": val["band_event_rate"].get(b, float("nan"))}
            for b in ["A", "B", "C", "D"] if b in val["band_event_rate"]
        ]),
        "",
        f"- 单调递减：**{'是' if val['band_monotone'] else '否'}**",
        f"- 最高档 / 最低档风险比："
        f"{'—' if val['risk_ratio_highest_vs_lowest'] is None else format(val['risk_ratio_highest_vs_lowest'], '.2f')}",
        f"- 分档人数均衡度：最少 {val['band_trend'].get('size_balance', {}).get('min')} / "
        f"最多 {val['band_trend'].get('size_balance', {}).get('max')}"
        "（用绝对阈值分档时人数本就会不均，这本身是可运营性的代价：司机能记住「85 分是 A」）",
        f"- 分档趋势秩相关：**{val['band_trend'].get('spearman_index_vs_rate')}**"
        f"（方向{'正确' if val['band_trend'].get('trend_direction_ok') else '异常'}）",
        "",
        "> 严格两两单调对单档只有十几二十人的分档过于苛刻：出险率 0.7、单档 20 人时",
        "> 标准误约 0.10，相邻档差 0.11 完全不显著。因此下表列出每次倒挂的样本量与标准误：",
        "",
        markdown_table(val["band_trend"].get("inversions", [])) if val["band_trend"].get("inversions")
        else "各档出险率严格单调递增。",
        "",
        "> 四档在正例率高、单档人数少时容易掩盖真实趋势，因此补充十分位趋势作为更灵敏的证据：",
        "",
        f"- 十分位出险率（分数由低到高）：`{val['decile_trend'].get('decile_event_rates_low_to_high_score')}`",
        f"- 出险率与分数的 Spearman：**{val['decile_trend'].get('spearman_rate_vs_score')}**"
        f"（≤ −0.8 视为单调下降：{'是' if val['decile_trend'].get('monotone_decreasing') else '否'}）",
        "",
        "## 4. 跨期稳定性（排名不能天天跳）",
        "",
        f"- Spearman = **{stab.get('spearman', float('nan'))}**（阈值 {stab.get('threshold')}）",
        f"- 结论：**{'通过' if stab.get('passed') else '未通过'}**",
        "",
        "## 5. 反事实测试（权重方向是否自洽）",
        "",
        f"- {cf.get('interpretation', '')}",
        f"- 降低该维度风险 50% 后，平均分数变化 {cf.get('mean_score_delta')} 分",
        "",
        "## 6. 提交物",
        "",
        f"- `{out_csv}`（列：`gpsno, score, band, risk_index, dim_*_score, dim_*_deduction`）",
        f"- `{card_path}`（每位司机一张可读安全卡）",
        "- **这两个文件含车辆级信息，按赛题保密条款不入公开仓库。**",
        "",
    ]
    write_text_lf(reports / "task2_validation.md", "\n".join(lines))

    playbook = management_playbook(cfg)
    write_text_lf(cfg.resolve("docs", ensure=True) / "05_运营建议.md", playbook)
    logger.info("运营建议已写入 docs/05_运营建议.md")

    print(f"\n连续风险指数 AUC = {val['auc_continuous_risk_index']:.4f}"
          f"（分档分 AUC = {val['auc_of_band_score']:.4f}，并列上限 {val['tie_diagnostics']['auc_ceiling']:.4f}）")
    print(f"分档单调：{val['band_monotone']} | 跨期稳定性：{stab.get('passed')} | 反事实：{cf.get('passed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
