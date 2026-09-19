"""03 · 特征工程 v1：构建完整观测窗口的特征表与特征字典。

保底路径只使用「风险事件 + 轨迹 + 车辆画像」三类数据（不含 IMU），约 40–90 个
特征。IMU 属于提升阶梯：压缩包、只覆盖部分车辆、需要姿态校准与自适应阈值，
一周内做对的风险高，因此不让它挡住主线。

产出：

* ``data/processed/feature_v1.parquet`` —— 每车一行（索引 ``gpsno``）
* ``reports/feature_dictionary.csv``    —— 特征字典（名称/组/定义/**时间来源**）

用法::

    python scripts/03_features.py [--raw data/raw/synthetic]
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from _common import find_raw_dir, get_config, load_daily_tables, setup_logging

from ie_safety.features.build import build_features, feature_dictionary
from ie_safety.io import discover_datasets, load_profile

logger = logging.getLogger("features")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="构建特征表 v1")
    ap.add_argument("--raw", default=None)
    args = ap.parse_args(argv)
    setup_logging()
    cfg = get_config()

    raw_dir = find_raw_dir(cfg, args.raw)
    found = discover_datasets(raw_dir)
    profile = load_profile(found["profile"])
    tables = load_daily_tables(cfg)

    vehicles = sorted(profile["gpsno"].unique().tolist())
    events = tables["events"]
    t0 = events["date"].min()
    window_end = t0 + pd.Timedelta(days=cfg.observation_days)

    X = build_features(
        daily_exposure=tables["exposure"],
        daily_events=tables["events"],
        profile=profile,
        cfg=cfg,
        window_start=t0,
        window_end=window_end,
        vehicles=vehicles,
        daily_imu=tables["imu"],
    )

    out = cfg.resolve("processed", ensure=True) / "feature_v1.parquet"
    X.reset_index().to_parquet(out, index=False)
    logger.info("特征表已写入 %s，形状 %s", out, X.shape)

    fdict = feature_dictionary(cfg)
    fdict_path = cfg.resolve("reports", ensure=True) / "feature_dictionary.csv"
    fdict.to_csv(fdict_path, index=False, encoding="utf-8-sig")
    logger.info("特征字典已写入 %s（%d 条）", fdict_path, len(fdict))

    print(f"\n特征表：{X.shape[0]} 台车 × {X.shape[1]} 列（其中字典条目 {len(fdict)} 条）")
    print(f"覆盖窗口：{t0.date()} ~ {window_end.date()}（{cfg.observation_days} 天）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
