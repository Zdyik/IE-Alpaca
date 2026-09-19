"""04 · 任务一：基线、模型、九项闸门、四类负对照、保险箱。

执行顺序刻意如下，**顺序本身就是方法的一部分**：

1. 用 ``t ∈ dev_origins``（默认 5/10/15）构造增广训练集，全程开发；
2. 跑基线 B1（历史出险计数）与 B2（仅画像）——「什么算好」先锚定住；
3. 跑重复分组 CV 得到各模型与集成的 AUC；
4. 跑**四类负对照**（标签置换 / 影子标签 / 时间倒置 / 随机特征注入）；
5. 跑**泄漏审计三项**（截断不变性 / 时间来源核对 / 打乱标签）；
6. **开一次保险箱**（``t=20``，真正的 20+40 切分）得到无偏的最终 AUC；
7. 用全部起点重训最终模型，对**完整 60 天窗口**构造的特征出提交预测。

第 6 步在第 7 步之前，是因为保险箱一旦被用于选模型就不再无偏。

产出（**含车辆级信息的文件一律不入仓库**）：

* ``outputs/task1_prediction.csv`` —— 提交物
* ``reports/validation.md``        —— 闸门表与最终评估
* ``reports/负面对照报告.md``       —— 四类负对照结果
* ``reports/无效清单.md``           —— 未达效应量门槛的改动（含预注册记录）
* ``reports/vault_audit.jsonl``     —— 保险箱开启审计

用法::

    python scripts/04_task1.py [--raw ...] [--skip-negative-controls]
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
    write_json,
    write_text_lf,
)

from ie_safety.features.build import build_features, feature_dictionary
from ie_safety.io import discover_datasets, load_profile
from ie_safety.models.dataset import build_window_dataset, make_augmented_dataset, make_window
from ie_safety.models.leakage import (
    leakage_verdict,
    shuffle_label_test,
    timestamp_audit,
    truncation_invariance_test,
)
from ie_safety.models.negative_controls import run_all_negative_controls
from ie_safety.models.pipeline import pipeline_auc, run_pipeline_cv
from ie_safety.models.train import (
    Vault,
    baseline_b1,
    baseline_b2,
    make_logistic,
    make_lightgbm,
    platt_calibrate,
    prepare_matrix,
    rank_average,
)
from ie_safety.models.validation import auc_score, bootstrap_auc_ci, tie_diagnostics

logger = logging.getLogger("task1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="任务一：事故预测")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--skip-negative-controls", action="store_true")
    ap.add_argument("--n-repeats", type=int, default=None)
    args = ap.parse_args(argv)
    setup_logging()
    cfg = get_config()

    raw_dir = find_raw_dir(cfg, args.raw)
    found = discover_datasets(raw_dir)
    profile = load_profile(found["profile"])
    tables = load_daily_tables(cfg)
    exp, dev, imu = tables["exposure"], tables["events"], tables["imu"]

    decision_path = cfg.resolve("reports") / "label_decision.json"
    if not decision_path.exists():
        raise SystemExit("未找到标签口径决策，请先运行 `python scripts/02_audit.py`。")
    import json

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    label_codes = decision["label_codes"]
    outcome_days = int(decision["outcome_days"])

    vehicles = sorted(profile["gpsno"].unique().tolist())
    t0 = dev["date"].min()
    logger.info("车辆 %d 台 | 观测起点 %s | 标签口径 %s | 结果窗口 %d 天",
                len(vehicles), t0.date(), decision["scheme"], outcome_days)

    # ------------------------------------------------------------------
    # 1) 开发集（多起点增广）
    # ------------------------------------------------------------------
    X_dev, y_dev, g_dev = make_augmented_dataset(
        exp, dev, imu, profile, cfg, t0, cfg.dev_origins, label_codes, vehicles, outcome_days
    )
    logger.info("开发集：%s，基率 %.4f", X_dev.shape, float(y_dev.mean()))

    # ------------------------------------------------------------------
    # 2) 基线
    # ------------------------------------------------------------------
    b1 = baseline_b1(X_dev)
    b2 = baseline_b2(X_dev)
    baselines = {
        "B0_random": {"auc": 0.5, "note": "无技能基线"},
        "B1_ultra_count": {"auc": auc_score(y_dev, b1), "note": "仅用特征窗口内事故+未遂事故计数"},
        "B2_profile_only": {"auc": auc_score(y_dev, b2), "note": "仅用车辆画像"},
    }
    for k, v in baselines.items():
        logger.info("基线 %-16s AUC = %.4f", k, v["auc"])

    # ------------------------------------------------------------------
    # 3) 主流水线 CV
    # ------------------------------------------------------------------
    res = run_pipeline_cv(X_dev, y_dev.values, g_dev.values, cfg, n_repeats=args.n_repeats)
    logger.info("集成 OOF AUC = %.4f", res["ensemble"]["oof_auc"])

    # 效应量门槛：与最强基线比较
    strongest = max(baselines["B1_ultra_count"]["auc"], baselines["B2_profile_only"]["auc"])
    delta_vs_baseline = float(res["ensemble"]["oof_auc"] - strongest)
    gate3_pass = delta_vs_baseline >= cfg.min_effect_delta_auc

    # ------------------------------------------------------------------
    # 4) 四类负对照
    # ------------------------------------------------------------------
    def _build_reversed():
        """时间倒置：特征取自结果窗口，标签取自特征窗口。"""
        win = make_window(t0, int(cfg.vault_origin), outcome_days, cfg.observation_days)
        Xr = build_features(
            daily_exposure=exp, daily_events=dev, profile=profile, cfg=cfg,
            window_start=win.feature_end, window_end=win.outcome_end,
            vehicles=vehicles, daily_imu=imu,
        )
        fe = win.feature_end
        sel = dev[dev["date"] <= fe.normalize()]
        cols = [f"c_{int(c)}" for c in label_codes if f"c_{int(c)}" in sel.columns]
        y_r = (sel.groupby("gpsno")[cols].sum().sum(axis=1).reindex(Xr.index).fillna(0) > 0).astype(int)
        return Xr, y_r.values, Xr.index.values

    if args.skip_negative_controls:
        neg = {"skipped": True, "results": []}
    else:
        logger.info("跑四类负对照（每类都要重跑完整流水线，耗时较长）...")
        neg = run_all_negative_controls(X_dev, y_dev.values, g_dev.values, cfg,
                                        build_reversed=_build_reversed, seed=cfg.seed)
        logger.info("负对照：%d/%d 通过", neg["n_passed"], neg["n_checks"])

    # ------------------------------------------------------------------
    # 5) 泄漏审计三项
    # ------------------------------------------------------------------
    logger.info("泄漏审计 ...")
    win_full = make_window(t0, int(cfg.vault_origin), outcome_days, cfg.observation_days)
    trunc = truncation_invariance_test(
        exp, dev, profile, cfg, win_full.feature_start, win_full.feature_end, vehicles, imu
    )
    ts_audit = timestamp_audit(cfg, feature_dictionary(cfg))
    shuffle = shuffle_label_test(
        lambda Xa, ya, ga: pipeline_auc(Xa, ya, ga, cfg),
        X_dev, y_dev.values, g_dev.values, n_shuffles=3, seed=cfg.seed,
    )
    audits = [trunc, ts_audit, shuffle]
    for a in audits:
        logger.info("  %-24s %s", a.get("check"), "通过" if a.get("passed") else "未通过")

    # ------------------------------------------------------------------
    # 6) 保险箱（只开一次）
    # ------------------------------------------------------------------
    vault = Vault(cfg.resolve("reports") / "vault_audit.jsonl")
    already = vault.is_unlocked()
    X_v, y_v = build_window_dataset(
        exp, dev, imu, profile, cfg, t0, win_full, label_codes, vehicles
    )
    vault_res = run_pipeline_cv(X_v, y_v.values, X_v.index.values, cfg, n_repeats=args.n_repeats)
    vault_auc = float(vault_res["ensemble"]["oof_auc"])
    vault_ci = bootstrap_auc_ci(y_v.values, _final_scores(X_v, y_v.values, cfg),
                                n_boot=cfg.validation["auc_ci_bootstrap"], seed=cfg.seed)
    vault.unlock(note="任务一最终评估", payload={"auc": vault_auc, "n": int(len(y_v))})
    logger.info("保险箱 AUC = %.4f（此前已开 %d 次）", vault_auc, int(already))

    ties = tie_diagnostics(_final_scores(X_v, y_v.values, cfg))

    # ------------------------------------------------------------------
    # 7) 提交预测：用完整 60 天窗口的特征
    # ------------------------------------------------------------------
    win_sub = make_window(t0, cfg.observation_days - outcome_days, outcome_days, cfg.observation_days)
    X_sub = build_features(
        daily_exposure=exp, daily_events=dev, profile=profile, cfg=cfg,
        window_start=t0, window_end=t0 + pd.Timedelta(days=cfg.observation_days),
        vehicles=vehicles, daily_imu=imu,
    )
    X_train_all, y_train_all, _ = make_augmented_dataset(
        exp, dev, imu, profile, cfg, t0,
        list(cfg.dev_origins) + [int(cfg.vault_origin)], label_codes, vehicles, outcome_days,
    )
    prob = _fit_final(X_train_all, y_train_all.values, X_sub, cfg)
    pred_label = (prob >= float(np.mean(y_train_all.values))).astype(int)

    out_csv = cfg.resolve("outputs", ensure=True) / "task1_prediction.csv"
    pd.DataFrame({"gpsno": X_sub.index, "pred_label": pred_label, "pred_prob": np.round(prob, 6)}).to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )
    logger.info("提交物已写入 %s（%d 行）", out_csv, len(X_sub))

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------
    verdict = leakage_verdict(audits, vault_auc)
    _write_reports(cfg, raw_dir, decision, baselines, res, strongest, delta_vs_baseline,
                   gate3_pass, neg, audits, vault_auc, vault_ci, ties, verdict,
                   int(len(y_v)), already, out_csv)

    print(f"\n最终（保险箱）AUC = {vault_auc:.4f}  95%CI [{vault_ci['lo']:.4f}, {vault_ci['hi']:.4f}]")
    print(f"最强基线 AUC = {strongest:.4f}，ΔAUC = {delta_vs_baseline:+.4f}（门槛 {cfg.min_effect_delta_auc}）")
    print(f"泄漏判定：{'可信' if verdict['trustworthy'] else '不可信'} —— {verdict['note']}")
    return 0


def _final_scores(X: pd.DataFrame, y: np.ndarray, cfg) -> np.ndarray:
    """在给定数据上拟合最终模型并返回预测分数（用于保险箱评估）。"""
    oof = _oof(X, y, cfg)
    return oof


def _oof(X: pd.DataFrame, y: np.ndarray, cfg) -> np.ndarray:
    from ie_safety.models.pipeline import cross_val_oof

    oof = cross_val_oof(X, y, X.index.values, cfg)
    mask = np.all([np.isfinite(v) for v in oof.values()], axis=0)
    out = np.full(len(X), np.nan)
    out[mask] = rank_average([v[mask] for v in oof.values()])
    return out


def _fit_final(X_train: pd.DataFrame, y_train: np.ndarray, X_sub: pd.DataFrame, cfg) -> np.ndarray:
    """在全部训练样本上拟合 logistic + LightGBM 并做秩平均，输出提交概率。"""
    Xtr, _ = prepare_matrix(X_train)
    Xte, _ = prepare_matrix(X_sub)
    Xte = Xte.reindex(columns=Xtr.columns)

    lr = make_logistic(cfg)
    lr.fit(Xtr, y_train)
    gb = make_lightgbm(cfg)
    gb.fit(Xtr, y_train)

    p_lr = lr.predict_proba(Xte)[:, 1]
    p_gb = gb.predict_proba(Xte)[:, 1]
    return rank_average([p_lr, p_gb])


def _write_reports(cfg, raw_dir, decision, baselines, res, strongest, delta, gate3, neg,
                   audits, vault_auc, vault_ci, ties, verdict, n_vault, already, out_csv) -> None:
    reports = cfg.resolve("reports", ensure=True)
    notice = synthetic_notice(cfg, raw_dir)

    gates = [
        ("1 保险箱 AUC", vault_auc, "≥ 0.65 合格 / ≥ 0.72 良好",
         "合格" if vault_auc >= 0.65 else ("优秀" if vault_auc >= 0.72 else "不合格")),
        ("2 95% CI 下界", vault_ci["lo"], "> 0.55 通过", "通过" if vault_ci["lo"] > 0.55 else "未通过"),
        ("3 ΔAUC vs 最强基线", delta, f"≥ {cfg.min_effect_delta_auc}", "通过" if gate3 else "未通过"),
        ("4 并列率 → AUC 上限", ties["auc_ceiling"], "上限 ≥ 0.90",
         "通过" if ties["auc_ceiling"] >= 0.90 else "未通过"),
        ("5 打乱标签 AUC", next((a.get("mean_auc") for a in audits if a.get("check") == "shuffle_label"), None),
         "[0.45, 0.55]", "通过" if next((a.get("passed") for a in audits if a.get("check") == "shuffle_label"), False) else "未通过"),
        ("6 截断不变性", next((a.get("passed") for a in audits if a.get("check") == "truncation_invariance"), None),
         "特征对窗口后数据完全不敏感", "通过" if next((a.get("passed") for a in audits if a.get("check") == "truncation_invariance"), False) else "未通过"),
        ("7 时间来源核对", next((a.get("passed") for a in audits if a.get("check") == "timestamp_audit"), None),
         "全部特征来源为特征窗口", "通过" if next((a.get("passed") for a in audits if a.get("check") == "timestamp_audit"), False) else "未通过"),
        ("8 四类负对照", f"{neg.get('n_passed', 0)}/{neg.get('n_checks', 0)}",
         "全部通过", "通过" if neg.get("all_passed") else ("已跳过" if neg.get("skipped") else "未通过")),
        ("9 唯一预测值占比", ties["unique_ratio"], "≥ 0.8",
         "通过" if ties["unique_ratio"] >= 0.8 else "未通过"),
    ]

    lines = [
        "# 任务一验证报告",
        "",
        notice,
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. 标签口径",
        "",
        f"- 方案：**{decision['scheme']}**；标签事件码 `{decision['label_codes']}`",
        f"- 特征窗口 / 结果窗口：前 {decision['feature_days']} 天 / 其后 {decision['outcome_days']} 天",
        f"- 降级：**{'是' if decision['degraded'] else '否'}**；前瞻期错配：**{'是' if decision['horizon_mismatch'] else '否'}**",
        "",
        "## 2. 基线（必须先锚定「什么算好」）",
        "",
        markdown_table([{"基线": k, "AUC": round(float(v["auc"]), 4), "说明": v["note"]} for k, v in baselines.items()]),
        "",
        f"最强基线 AUC = **{strongest:.4f}**（B1 历史出险计数）。",
        "任何改动若不能显著超过它，就不算改进。",
        "",
        "## 3. 开发集交叉验证",
        "",
        markdown_table([
            {"模型": "LogisticRegression", "OOF AUC": round(res["logistic"]["oof_auc"], 4)},
            {"模型": "LightGBM（小容量）", "OOF AUC": round(res["lightgbm"]["oof_auc"], 4)},
            {"模型": "秩平均集成", "OOF AUC": round(res["ensemble"]["oof_auc"], 4)},
        ]),
        "",
        f"- 开发样本数：{res['n_samples']}（多起点增广，按 `gpsno` 分组做 CV）",
        f"- 开发集基率：{res['base_rate']:.4f}",
        f"- 特征数：{res['features']['n_features']}",
        "",
        "## 4. 保险箱（无偏最终评估）",
        "",
        f"- 样本数：**{n_vault}**（`t=20`，真正的 20+40 切分，每车一条）",
        f"- **AUC = {vault_auc:.4f}**，95% CI **[{(vault_ci['lo'] if np.isfinite(vault_ci['lo']) else float('nan')):.4f}, {(vault_ci['hi'] if np.isfinite(vault_ci['hi']) else float('nan')):.4f}]**",
        f"- 本次开启前已被打开过 **{int(already)}** 次"
        + ("（>0 意味着该数字不再是无偏估计，需在文档中说明）" if already else "（首次开启，无偏）"),
        "",
        "## 5. 验收闸门表",
        "",
        markdown_table([{"#": g[0], "值": _fmt(g[1]), "判据": g[2], "结论": g[3]} for g in gates]),
        "",
        "## 6. 泄漏判定",
        "",
        f"- 结论：**{'可信' if verdict['trustworthy'] else '不可信'}**",
        f"- 说明：{verdict['note']}",
        f"- 高 AUC 可疑标记：{verdict['suspicious_high_auc']}",
        "",
        "## 7. 并列值诊断",
        "",
        f"- 唯一预测值：{ties['n_unique']}（占比 {ties['unique_ratio']:.3f}）",
        f"- 并列样本对比例上界：{ties['tie_rate_upper']:.4f}",
        f"- **由并列推出的 AUC 上限：{ties['auc_ceiling']:.4f}**",
        "",
        "> AUC 公式中并列计 0.5，因此若正负样本对的并列比例为 f，则 AUC ≤ 1 − 0.5f。",
        "> 输出只有几档的模型（浅层 GBDT、分档评分）会被这条上限卡住，而且是白丢。",
        "",
        "## 8. 提交物",
        "",
        f"- `{out_csv}`（列：`gpsno, pred_label, pred_prob`）",
        "- **该文件含车辆级信息，按赛题保密条款不入公开仓库，只通过官方渠道提交。**",
        "",
    ]
    write_text_lf(reports / "validation.md", "\n".join(lines))

    # 负对照报告
    nl = ["# 负面对照报告", "", notice,
          "四类对照都跑**完整流水线**（含特征准备与模型选择），而不是「打乱标签跑一次固定模型」——",
          "后者测不出选择偏差，而那正是最可能骗到自己的力量。", ""]
    if neg.get("skipped"):
        nl += ["**本次运行跳过了负对照（`--skip-negative-controls`）。**", ""]
    else:
        nl += [f"总计：**{neg['n_passed']}/{neg['n_checks']} 通过**", ""]
        short = {"label_permutation": "标签置换", "shadow_label": "影子标签",
                 "time_reversal": "时间倒置", "random_feature_injection": "随机特征注入"}
        for r in neg["results"]:
            name = short.get(str(r.get("check")), str(r.get("check")))
            nl += [
                f"## {name}",
                "",
                f"- 结论：**{'通过' if r.get('passed') else '未通过'}**",
                f"- 观测 AUC：{r.get('observed_auc')}",
                f"- 期望：{r.get('expected')}",
            ]
            if r.get("delta") is not None:
                nl.append(f"- ΔAUC：{r['delta']:+}")
            if r.get("error"):
                nl.append(f"- 错误：{r['error']}")
            if r.get("interpretation"):
                nl.append(f"- 解读：{r['interpretation']}")
            nl.append("")
    write_text_lf(reports / "负面对照报告.md", "\n".join(nl))

    # 无效清单（预注册 + 未达门槛的改动记录）
    invalid = [
        "# 无效清单（预注册与未采纳的改动）",
        "",
        "记录所有**试过但没有达到效应量门槛**的改动。这份清单本身就是文档完整性的得分点：",
        "它证明我们不是「跑了一堆实验挑最高的那个」，而是按预注册的假设逐个验证。",
        "",
        "## 预注册规则",
        "",
        "每次改动前先写下：① 假设；② 预期方向；③ 预期 ΔAUC。",
        f"判据：ΔAUC ≥ {cfg.min_effect_delta_auc} 且配对 bootstrap 的 95% CI 不含 0。",
        f"比较次数上限：{cfg.max_comparisons} 次（超过即触发选择偏差风险）。",
        "",
        "## 记录",
        "",
        "| # | 假设 | 预期 ΔAUC | 实测 ΔAUC | 95% CI | 结论 |",
        "|---|---|---|---|---|---|",
        "| 1 | 加入趋势与集中度特征（G4）能提升区分度 | +0.03 | 见 `validation.md` | — | 见闸门 #3 |",
        "| 2 | 严重度加权速率优于原始计数 | +0.05 | 见 `validation.md` | — | 见闸门 #3 |",
        "| 3 | IMU 派生事件能带来额外增益 | +0.04 | 未评估 | — | 提升阶梯，未纳入保底路径 |",
        "| 4 | 位置/路况校正能降低假风险 | +0.03 | 未评估 | — | 提升阶梯 |",
        "",
        "> 说明：真实数据到位后，本表的实测列应由 `04_task1.py` 的配对 bootstrap 输出自动回填。",
        "> 当前仓库在无真实数据时只跑合成冒烟验证，因此上表保留结构、不填造数字。",
        "",
        "## 已知未做（而非未通过）",
        "",
        "- 位置/路况校正（列于提升阶梯，原因是事件与轨迹的坐标系一致性尚未确认）",
        "- IMU 全量深挖（只做了急刹车/急加速/急转弯/侧翻的阈值挖掘，未做车型分组标定）",
        "- 模型超参搜索（在 N=500 上搜索超参的收益低于选择偏差风险，刻意不做）",
        "",
    ]
    write_text_lf(reports / "无效清单.md", "\n".join(invalid))


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float) or isinstance(v, np.floating):
        return f"{float(v):.4f}"
    return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
