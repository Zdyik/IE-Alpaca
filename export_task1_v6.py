"""Export one completed V6 run in the same three-column format as V5."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import duckdb
import numpy as np


def _read(path: Path, query: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    with duckdb.connect() as con:
        return con.execute(query, [str(path)]).df()


def export(run_dir: Path, profile_path: Path, output: Path | None = None) -> Path:
    run_dir, profile_path = run_dir.resolve(), profile_path.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("model") != "torch_all_event_stationary_hazard_v6_width":
        raise ValueError("只能导出已完成的 V6 全事件风险模型")
    source = _read(
        run_dir / "predictions" / "candidate_500.parquet",
        "SELECT CAST(gpsno AS VARCHAR) AS gpsno, anchor_day, horizon_days, "
        "probability, prediction_at_0_5, route, label_status FROM read_parquet(?)",
    )
    profile = _read(profile_path, "SELECT CAST(gpsno AS VARCHAR) AS gpsno FROM read_parquet(?)")
    if len(source) != 500 or source.gpsno.isna().any() or source.gpsno.duplicated().any():
        raise ValueError("候选结果必须恰好有 500 台唯一车辆")
    if len(profile) != 500 or profile.gpsno.duplicated().any() or set(source.gpsno) != set(profile.gpsno):
        raise ValueError("候选设备号与赛题画像不一致")
    if not source.anchor_day.eq(60).all() or not source.horizon_days.eq(40).all():
        raise ValueError("交付表必须为第 60 天后 40 天风险")
    if not source.label_status.eq("future_unknown").all():
        raise ValueError("交付表不能含已知结局")
    p = source.probability.to_numpy(dtype=float)
    prediction = source.prediction_at_0_5.to_numpy(dtype=int)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("概率必须在 [0,1] 且为有限数")
    if not np.array_equal(prediction, (p >= 0.5).astype(int)):
        raise ValueError("二分类结果必须采用固定 0.5 阈值")
    if source.route.eq("prior_no_observed_behavior").sum() != 22:
        raise ValueError("低证据车辆数与 V6 运行记录不一致")

    output = output.resolve() if output else run_dir / "deliverables" / "task1_v6_predictions.csv"
    if not output.is_relative_to(run_dir):
        raise ValueError("交付表必须保存在本版实验文件夹内")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("gpsno", "prediction", "probability"))
        for row in source.sort_values("gpsno").itertuples(index=False):
            writer.writerow((row.gpsno, int(row.prediction_at_0_5), format(float(row.probability), ".15g")))
    shutil.copy2(Path(__file__), run_dir / "source" / "export_task1_v6.py")
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
