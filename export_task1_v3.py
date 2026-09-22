"""Export the archived V3 Model 0 candidate as the task-one delivery CSV.

The competition PDF requires one vehicle ID, a binary result and a probability
per row, but does not prescribe column names. This exporter uses
gpsno,prediction,probability and never accesses locked holdout labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import duckdb
import numpy as np


def _read_parquet(path: Path, query: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    with duckdb.connect() as con:
        return con.execute(query, [str(path)]).df()


def export(run_dir: Path, profile_path: Path, output: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    profile_path = profile_path.resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("model") != "landmark_elasticnet_hazard":
        raise ValueError("只能导出已完成的 V3 Model 0 版本")

    source = _read_parquet(
        run_dir / "predictions" / "candidate_500.parquet",
        "SELECT CAST(gpsno AS VARCHAR) AS gpsno, anchor_day, horizon_days, "
        "probability, prediction_at_0_5, route, label_status FROM read_parquet(?)",
    )
    profile = _read_parquet(profile_path, "SELECT CAST(gpsno AS VARCHAR) AS gpsno FROM read_parquet(?)")
    if len(source) != 500 or source.gpsno.isna().any() or source.gpsno.duplicated().any():
        raise ValueError("候选预测必须恰好包含 500 台唯一车辆")
    if len(profile) != 500 or profile.gpsno.duplicated().any() or set(source.gpsno) != set(profile.gpsno):
        raise ValueError("候选车辆 ID 与赛题画像中的 500 台车辆不完全一致")
    if not source.anchor_day.eq(60).all() or not source.horizon_days.eq(40).all():
        raise ValueError("交付表必须是 Day60→未来40天预测")
    if not source.label_status.eq("future_unknown").all():
        raise ValueError("候选预测混入了已知结局")
    p = source.probability.to_numpy(dtype=float)
    y = source.prediction_at_0_5.to_numpy(dtype=int)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("预测概率包含非有限值或超出 [0,1]")
    if not np.isin(y, (0, 1)).all() or not np.array_equal(y, (p >= 0.5).astype(int)):
        raise ValueError("二分类结果不符合预先固定的 0.5 阈值")
    if source.route.eq("prior_no_observed_behavior").sum() != 22:
        raise ValueError("低证据车辆数量与 V3 审计不符")

    output = output.resolve() if output else run_dir / "deliverables" / "task1_v3_model0_predictions.csv"
    if not output.is_relative_to(run_dir):
        raise ValueError("交付 CSV 必须保存在该实验版本目录中")
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = source.sort_values("gpsno")
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("gpsno", "prediction", "probability"))
        for row in ordered.itertuples(index=False):
            writer.writerow((row.gpsno, int(row.prediction_at_0_5), format(float(row.probability), ".15g")))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=Path(
        r"E:\Databases\2026 清华IE亮剑-算法赛道赛题\任务一预处理结果\profile.parquet"
    ))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(export(args.run_dir, args.profile, args.output))


if __name__ == "__main__":
    main()
