"""02 · 数据审计、标签口径闸门与免费锚点校验。

这是整个项目的**决策闸门**：`11803`/`11804` 的计数与分布决定监督信号怎么构造，
从而决定后面所有代码怎么写。

产出：

* ``data/raw/data_manifest.json``  —— 数据清单（文件、行数、时间范围、哈希）
* ``data/processed/daily_*.parquet`` —— 日粒度聚合表
* ``reports/label_decision.json``    —— 标签口径决策（含完整探测证据）
* ``reports/anchor_checks.json``     —— 频次序关系等免费锚点结论
* ``docs/01_数据审计.md``            —— 人可读的审计报告

用法::

    python scripts/02_audit.py [--raw data/raw/synthetic]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from _common import (
    write_text_lf,
    find_raw_dir,
    get_config,
    markdown_table,
    rel_or_abs,
    setup_logging,
    synthetic_notice,
    write_json,
)

from ie_safety.audit.anchors import check_frequency_order, summarize_anchor_results
from ie_safety.audit.label_probe import decide_label_scheme, probe_label_structure, write_decision
from ie_safety.features.daily import (
    build_daily_events,
    build_daily_exposure,
    build_daily_imu,
    save_daily,
)
from ie_safety.io import discover_datasets, load_events, load_profile

logger = logging.getLogger("audit")


def _sha256(path: Path, max_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()[:16]


def build_manifest(found: dict, cfg) -> dict:
    """记录文件名、大小与哈希，便于复现时核对数据是否同一份。"""
    manifest = {"data_dir": rel_or_abs(Path(found["profile"][0].path).parent) if found.get("profile") else "",
                "datasets": {}, "n_files": 0}
    for kind, sources in found.items():
        for s in sources:
            entry = {
                "kind": kind,
                "file": rel_or_abs(s.path),
                "member": s.member,
                "size_bytes": s.path.stat().st_size,
                "sha256_head": _sha256(s.path),
                "columns_detected": s.columns[:40],
            }
            manifest["datasets"].setdefault(kind, []).append(entry)
            manifest["n_files"] += 1
    manifest["observation"] = {
        "start": cfg["data"]["observation_start"],
        "end": cfg["data"]["observation_end"],
        "days": cfg.observation_days,
    }
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="数据审计与标签口径闸门")
    ap.add_argument("--raw", default=None, help="原始数据目录（默认自动发现）")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = get_config()

    raw_dir = find_raw_dir(cfg, args.raw)
    logger.info("原始数据目录: %s", raw_dir)

    found = discover_datasets(raw_dir)
    for kind in ("profile", "events", "trajectory", "imu"):
        logger.info("  %-11s %s", kind, [s.label for s in found.get(kind, [])])
    if found.get("?"):
        logger.info("  未归类     %s", [s.label for s in found["?"]])

    manifest = build_manifest(found, cfg)
    write_json(cfg.resolve("raw") / "data_manifest.json", manifest)
    write_json(cfg.resolve("reports") / "data_manifest.json", manifest)

    # ---------------- 日粒度聚合 ----------------
    logger.info("聚合日粒度表 ...")
    daily_exp = build_daily_exposure(found["trajectory"], cfg)
    daily_ev = build_daily_events(found["events"], cfg)
    daily_imu = build_daily_imu(found["imu"], cfg) if found.get("imu") else None

    proc = cfg.resolve("processed")
    save_daily(daily_exp, proc / "daily_exposure.parquet")
    save_daily(daily_ev, proc / "daily_events.parquet")
    save_daily(daily_imu, proc / "daily_imu.parquet")
    logger.info("  exposure %s | events %s | imu %s", daily_exp.shape, daily_ev.shape,
                None if daily_imu is None else daily_imu.shape)

    # ---------------- 标签口径闸门 ----------------
    profile = load_profile(found["profile"])
    events = load_events(found["events"], cfg.timezone)
    vehicles = sorted(profile["gpsno"].unique().tolist())

    probe = probe_label_structure(events, cfg)
    decision = decide_label_scheme(probe, cfg)
    write_decision(decision, cfg.resolve("reports") / "label_decision.json")
    logger.info("标签口径: %s | outcome_days=%s | 降级=%s", decision["scheme"],
                decision["outcome_days"], decision["degraded"])

    # ---------------- 免费锚点 ----------------
    anchors = [check_frequency_order(events, cfg)]
    anchor_summary = summarize_anchor_results(anchors)
    write_json(cfg.resolve("reports") / "anchor_checks.json", anchor_summary)

    # ---------------- 审计报告 ----------------
    rates = probe["base_rates_by_origin"]
    report = [
        "# 数据审计报告",
        "",
        synthetic_notice(cfg, raw_dir),
        "## 1. 数据清单",
        "",
        f"- 原始数据目录：`{rel_or_abs(raw_dir)}`",
        f"- 识别出的数据集：{markdown_table([{'数据集': k, '文件': ', '.join(s.label for s in v)} for k, v in found.items() if k != '?' and v])}",
        "",
        "## 2. 规模与覆盖",
        "",
        markdown_table([
            {"项目": "车辆画像（500 台预期）", "值": f"{len(profile)} 台"},
            {"项目": "风险事件条数", "值": f"{len(events)}"},
            {"项目": "事件时间范围", "值": f"{events['ts'].min()} ~ {events['ts'].max()}"},
            {"项目": "车辆画像字段数", "值": f"{profile.shape[1]}"},
            {"项目": "日粒度暴露量表", "值": f"{daily_exp.shape[0]} 行"},
            {"项目": "日粒度事件表", "值": f"{daily_ev.shape[0]} 行"},
            {"项目": "日粒度 IMU 表", "值": "无" if daily_imu is None else f"{daily_imu.shape[0]} 行"},
        ]),
        "",
        "## 3. 标签口径闸门（最高优先级）",
        "",
        f"- 事故（{cfg.accident_codes}）事件数：**{probe['accident_events']}**，覆盖 {probe['accident_vehicles']} 台车",
        f"- 未遂事故（{cfg.near_miss_codes}）事件数：**{probe['near_miss_events']}**，覆盖 {probe['near_miss_vehicles']} 台车",
        f"- 最终口径：**{decision['scheme']}**，标签事件码 `{decision['label_codes']}`",
        f"- 结果窗口：**{decision['outcome_days']} 天**（赛题要求 {decision['outcome_days_original']} 天，"
        f"前瞻期错配：{'是' if decision['horizon_mismatch'] else '否'}）",
        f"- 是否降级：**{'是' if decision['degraded'] else '否'}**",
        "",
        "### 决策理由",
        "",
        *[f"- {r}" for r in decision["reasons"]],
        "",
        "### 各起点下的标签基率",
        "",
        markdown_table(rates),
        "",
        "> 说明：`60 = 20 + 40` 是唯一能让 40 天结果窗口刚好落在 60 天观测期内的整数切分。",
        "> `origin_day=20` 的这组样本构成**保险箱**，全程锁死，只在最终评估时开一次。",
        "",
        "## 4. 免费锚点校验（无需标签即可验证特征抽取）",
        "",
        markdown_table([
            {"检查": a.get("check"), "通过": a.get("passed"), "说明": a.get("interpretation", "")}
            for a in anchors
        ]),
        "",
        f"事件族频次实测排序：`{anchors[0].get('ordering_desc', '')}`",
        "",
        "## 5. 已知问题与待复核项",
        "",
        "- `gpsno` 全程按字符串处理（防前导零与科学计数法把车 ID 弄坏），并作为四表 join 键。",
        "- 车辆画像的比率字段在文档示例里是 `\"4.21%\"` 字符串但声明为 `double`，"
        "读取层已按「去百分号 + 量级 > 1 时除以 100」兼容解析。",
        "- 事件经纬度与轨迹经纬度的坐标系是否一致**尚未确认**；若不一致，位置校正会让热点错位，"
        "所以位置校正被放在提升阶梯而非保底路径。",
        "- IMU 只有部分车辆覆盖，属正常情况（出题方已说明），特征层对缺失保持 NaN 不填 0。",
        "",
    ]

    docs = cfg.resolve("docs")
    out_path = docs / "01_数据审计.md"
    write_text_lf(out_path, "\n".join(report))
    logger.info("审计报告已写入 %s", out_path)
    print(f"\n标签口径结论：{decision['scheme']}（{decision['reasons'][0]}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
