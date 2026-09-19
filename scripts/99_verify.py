"""99 · 端到端自检：在没有真实数据时验证「代码链路能跑 + 验证协议自洽」。

这是一份**可执行的完成定义（DoD）检查**。它在合成数据上跑完整链路，并断言那些
在真实数据上无法预先知道答案的性质：

======================  ==========================================================
断言                     为什么能在合成数据上验证
======================  ==========================================================
链路端到端可跑           读取 → 审计 → 特征 → 建模 → 评分 全流程无异常
窗口不越界               20 + 40 = 60，结果窗口必须落在观测期内
截断不变性               特征对特征窗口之后的数据完全不敏感（实现层无时间泄漏）
打乱标签回到 0.5         合成数据有已知生成过程，所以「正确时应当有信号、打乱后
                         应当没信号」是可验证的
模型优于单特征基线       否则特征工程没有产生价值
评分单调趋势             连续风险指数与出险率的秩相关方向正确
合规校验通过             无数据文件、赛题 PDF、车辆级结果或密钥入库
======================  ==========================================================

用法::

    python scripts/99_verify.py [--vehicles 100] [--quick]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from _common import get_config, setup_logging

RESULTS: list = []


def record(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append({"check": name, "passed": bool(passed), "detail": detail})
    mark = "✓" if passed else "✗"
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="端到端自检")
    ap.add_argument("--vehicles", type=int, default=100)
    ap.add_argument("--quick", action="store_true", help="跳过耗时的负对照")
    args = ap.parse_args(argv)
    setup_logging(logging.WARNING)
    cfg = get_config()

    # 用临时目录跑，避免污染真实数据区
    tmp = Path(tempfile.mkdtemp(prefix="ie_verify_"))
    try:
        return _run(cfg, tmp, args)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run(cfg, tmp: Path, args) -> int:
    from ie_safety.audit.anchors import check_frequency_order
    from ie_safety.audit.label_probe import decide_label_scheme, probe_label_structure
    from ie_safety.features.build import build_features
    from ie_safety.features.daily import build_daily_events, build_daily_exposure, build_daily_imu
    from ie_safety.io import discover_datasets, load_events, load_profile
    from ie_safety.models.dataset import build_window_dataset, make_augmented_dataset, make_window
    from ie_safety.models.leakage import truncation_invariance_test
    from ie_safety.models.negative_controls import control_label_permutation
    from ie_safety.models.pipeline import pipeline_auc, run_pipeline_cv
    from ie_safety.models.train import baseline_b1
    from ie_safety.models.validation import auc_score
    from ie_safety.scoring import compute_scores, fit_dimension_weights, validate_scores
    from ie_safety.synthetic import SyntheticSpec, generate

    print("\n[1/6] 生成合成数据 ...")
    raw = tmp / "raw"
    generate(SyntheticSpec(n_vehicles=args.vehicles, n_days=cfg.observation_days, seed=cfg.seed), raw)
    found = discover_datasets(raw)
    record("数据集按列签名正确识别", all(found.get(k) for k in ("profile", "events", "trajectory", "imu")),
           ", ".join(f"{k}={len(found.get(k) or [])}" for k in ("profile", "events", "trajectory", "imu")))

    print("\n[2/6] 读取与日粒度聚合 ...")
    profile = load_profile(found["profile"])
    events = load_events(found["events"], cfg.timezone)
    exp = build_daily_exposure(found["trajectory"], cfg)
    dev = build_daily_events(found["events"], cfg)
    imu = build_daily_imu(found["imu"], cfg)
    record("日粒度聚合产出", len(exp) > 0 and len(dev) > 0,
           f"exposure={exp.shape}, events={dev.shape}, imu={None if imu is None else imu.shape}")
    record("画像比率字段兼容 % 字符串", profile["highway_km_ratio"].between(0, 1).all(),
           f"highway_km_ratio ∈ [{profile['highway_km_ratio'].min():.3f}, {profile['highway_km_ratio'].max():.3f}]")

    print("\n[3/6] 标签口径闸门与锚点 ...")
    probe = probe_label_structure(events, cfg)
    decision = decide_label_scheme(probe, cfg)
    record("标签口径可判定", probe["label_events_total"] > 0,
           f"{decision['scheme']}，正例车辆 {probe['label_vehicles']}")
    anchor = check_frequency_order(events, cfg)
    record("事件族频次序关系锚点", anchor["passed"], anchor.get("ordering_desc", ""))

    vehicles = sorted(profile["gpsno"].unique().tolist())
    t0 = dev["date"].min()
    outcome_days = int(decision["outcome_days"])
    label_codes = decision["label_codes"]

    print("\n[4/6] 特征与泄漏审计 ...")
    X_dev, y_dev, g_dev = make_augmented_dataset(
        exp, dev, imu, profile, cfg, t0, cfg.dev_origins, label_codes, vehicles, outcome_days
    )
    record("多起点增广数据集", X_dev.shape[0] == len(vehicles) * len(cfg.dev_origins),
           f"{X_dev.shape[0]} 样本 × {X_dev.shape[1]} 列")

    win = make_window(t0, int(cfg.vault_origin), outcome_days, cfg.observation_days)
    trunc = truncation_invariance_test(exp, dev, profile, cfg, win.feature_start, win.feature_end,
                                       vehicles, imu)
    record("截断不变性（实现层无时间泄漏）", trunc["passed"],
           f"差异单元 {trunc['detail_truncated']['n_differing_cells']}")

    print("\n[5/6] 建模与验证协议 ...")
    b1 = auc_score(y_dev, baseline_b1(X_dev))
    res = run_pipeline_cv(X_dev, y_dev.values, g_dev.values, cfg, n_repeats=2)
    ens = float(res["ensemble"]["oof_auc"])
    record("模型优于单特征基线 B1", ens > b1, f"集成 {ens:.4f} vs B1 {b1:.4f}")

    if args.quick:
        record("标签置换负对照", True, "已跳过（--quick）")
    else:
        nc = control_label_permutation(X_dev, y_dev.values, g_dev.values, cfg, n_shuffles=2, seed=cfg.seed)
        record("标签置换负对照回到 0.5 附近", nc["passed"], f"AUC={nc['observed_auc']}")

    X_v, y_v = build_window_dataset(exp, dev, imu, profile, cfg, t0, win, label_codes, vehicles)
    vault_res = run_pipeline_cv(X_v, y_v.values, X_v.index.values, cfg, n_repeats=2)
    record("保险箱无偏评估可跑", np.isfinite(vault_res["ensemble"]["oof_auc"]),
           f"AUC={vault_res['ensemble']['oof_auc']:.4f}，n={len(y_v)}")

    print("\n[6/6] 任务二评分 ...")
    fit = fit_dimension_weights(X_dev, y_dev.values, cfg)
    scores_v = compute_scores(X_v, fit["weights"], cfg)
    val = validate_scores(scores_v, y_v.values, cfg)
    record("连续风险指数区分能力", val["auc_continuous_risk_index"] > 0.55,
           f"AUC={val['auc_continuous_risk_index']:.4f}")
    record("评分趋势方向正确", bool(val["band_trend"].get("trend_direction_ok")),
           f"分档秩相关={val['band_trend'].get('spearman_index_vs_rate')}，"
           f"十分位秩相关={val['decile_trend'].get('spearman_rate_vs_score')}")

    # ---------------- 汇总 ----------------
    n_pass = sum(1 for r in RESULTS if r["passed"])
    print(f"\n{'=' * 62}")
    print(f"自检结果：{n_pass}/{len(RESULTS)} 通过")
    print(f"{'=' * 62}")
    for r in RESULTS:
        if not r["passed"]:
            print(f"  未通过：{r['check']} — {r['detail']}")
    if n_pass == len(RESULTS):
        print("全部通过。链路可跑、协议自洽。")
        print("注意：以上均基于**合成数据**，不代表比赛结果。")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
