"""Export a completed V7 run as gpsno,prediction,probability."""

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
    if manifest.get("status") != "completed" or manifest.get("model") != "torch_daily_transformer_v7":
        raise ValueError("Only a completed V7 transformer run may be exported")
    source = _read(
        run_dir / "predictions" / "candidate_500.parquet",
        "SELECT CAST(gpsno AS VARCHAR) AS gpsno, anchor_day, horizon_days, "
        "probability, prediction_at_0_5, route, label_status FROM read_parquet(?)",
    )
    profile = _read(profile_path, "SELECT CAST(gpsno AS VARCHAR) AS gpsno FROM read_parquet(?)")
    if len(source) != 500 or source.gpsno.isna().any() or source.gpsno.duplicated().any():
        raise ValueError("V7 candidates must cover exactly 500 unique vehicles")
    if len(profile) != 500 or profile.gpsno.duplicated().any() or set(source.gpsno) != set(profile.gpsno):
        raise ValueError("Candidate vehicles differ from the competition profile")
    if not source.anchor_day.eq(60).all() or not source.horizon_days.eq(40).all():
        raise ValueError("Delivery must represent day-60 to future-40 risk")
    if not source.label_status.eq("future_unknown").all():
        raise ValueError("Future labels cannot be known locally")
    p = source.probability.to_numpy(dtype=float)
    prediction = source.prediction_at_0_5.to_numpy(dtype=int)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("All probabilities must be finite and between 0 and 1")
    if not np.array_equal(prediction, (p >= .5).astype(int)):
        raise ValueError("Predictions must use the fixed 0.5 threshold")
    if source.route.eq("prior_no_observed_behavior").sum() != 22:
        raise ValueError("The 22 no-observation vehicles must use the prior route")
    output = output.resolve() if output else run_dir / "deliverables" / "task1_v7_predictions.csv"
    if not output.is_relative_to(run_dir):
        raise ValueError("Delivery must stay inside the immutable V7 run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("gpsno", "prediction", "probability"))
        for row in source.sort_values("gpsno").itertuples(index=False):
            writer.writerow((row.gpsno, int(row.prediction_at_0_5), format(float(row.probability), ".15g")))
    shutil.copy2(Path(__file__), run_dir / "source" / "export_task1_v7.py")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=Path(
        r"E:\Databases\2026 清华IE亮剑-算法赛道赛题\任务一预处理结果\profile.parquet",
    ))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(export(args.run_dir, args.profile, args.output))


if __name__ == "__main__":
    main()
