"""Validate the complete V12 task-two deliverable bundle against its source population."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent
DEFAULT_DB = Path(r"E:\Databases\2026 清华IE亮剑-算法赛道赛题")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_run(run_dir: Path | None, database_root: Path) -> Path:
    if run_dir is not None:
        return run_dir.resolve()
    latest = json.loads((database_root / "任务二安全评价结果" / "latest.json").read_text(encoding="utf-8"))
    return Path(latest["run_dir"]).resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(run_dir: Path, database_root: Path) -> dict[str, object]:
    deliverables = run_dir / "deliverables"
    paths = {
        "scores": deliverables / "task2_driver_safety_scores.csv",
        "scorecards": deliverables / "task2_driver_scorecards.csv",
        "history": deliverables / "task2_daily_score_history.csv",
        "weights": deliverables / "task2_event_weights.csv",
        "fleet": deliverables / "task2_fleet_summary.csv",
        "report": deliverables / "task2_v12_safety_model_report.pdf",
    }
    for name, path in paths.items():
        require(path.exists() and path.stat().st_size > 0, f"missing or empty {name}: {path}")

    scores = pd.read_csv(paths["scores"], encoding="utf-8-sig", dtype={"gpsno": str})
    cards = pd.read_csv(paths["scorecards"], encoding="utf-8-sig", dtype={"gpsno": str})
    history = pd.read_csv(paths["history"], encoding="utf-8-sig", dtype={"gpsno": str})
    weights = pd.read_csv(paths["weights"], encoding="utf-8-sig")
    fleet = pd.read_csv(paths["fleet"], encoding="utf-8-sig")

    profile_path = database_root / "任务一预处理结果" / "profile.parquet"
    with duckdb.connect() as connection:
        profile = connection.execute("SELECT CAST(gpsno AS VARCHAR) gpsno FROM read_parquet(?)",
                                     [str(profile_path)]).df()
    expected_ids = set(profile.gpsno)
    require(len(expected_ids) == 500, "source profile does not contain 500 unique vehicles")
    require(list(scores.columns) == ["gpsno", "safety_score"], "minimal score CSV has wrong columns")
    require(len(scores) == 500 and scores.gpsno.nunique() == 500, "minimal score CSV is not one row per vehicle")
    require(set(scores.gpsno) == expected_ids, "minimal score CSV vehicle population differs from source profile")
    require(scores.safety_score.between(0, 100).all(), "safety score is outside [0,100]")

    require(len(cards) == 500 and cards.gpsno.nunique() == 500, "scorecard is not one row per vehicle")
    require(cards.isna().sum().sum() == 0, "scorecard contains null values")
    joined = scores.merge(cards[["gpsno", "safety_score"]], on="gpsno", suffixes=("_short", "_detail"),
                          validate="one_to_one")
    require(np.allclose(joined.safety_score_short, joined.safety_score_detail, atol=5e-5),
            "minimal and detailed safety scores disagree")
    require(np.allclose(cards.safety_score, 100 - cards.total_deduction, atol=1e-8),
            "detailed deductions do not reconcile to the score")
    require(set(cards.grade).issubset({"A", "B", "C", "D", "E", "U"}), "unexpected grade")
    require(((cards.confidence_level == "低") == (cards.grade == "U")).all(),
            "low-confidence and U-grade assignments differ")

    expected_anchors = set(range(14, 61))
    require(len(history) == 500 * len(expected_anchors), "daily score history has wrong row count")
    require(history.gpsno.nunique() == 500, "daily score history is missing vehicles")
    require(set(history.anchor_day) == expected_anchors, "daily score history has wrong anchor days")
    require((history.groupby("gpsno").anchor_day.nunique() == len(expected_anchors)).all(),
            "one or more vehicles have incomplete daily history")

    require(len(weights) == 24 and weights.event_code.nunique() == 24,
            "event weight file must contain 24 unique events")
    sums = weights.groupby("state").learned_weight_within_state.sum()
    require(np.allclose(sums, 1.0, atol=1e-8), "event weights do not sum to one within state")
    require(int(fleet.vehicles.sum()) == 500, "fleet summary does not sum to 500 vehicles")

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("status") == "completed", "run manifest is not completed")
    require(manifest.get("report_sha256") == sha256(paths["report"]), "report hash differs from manifest")

    report = {
        "status": "passed",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest["run_id"],
        "vehicles": 500,
        "daily_rows": int(len(history)),
        "daily_anchor_range": [int(history.anchor_day.min()), int(history.anchor_day.max())],
        "event_types": int(len(weights)),
        "score_range": [float(scores.safety_score.min()), float(scores.safety_score.max())],
        "grade_counts": {str(k): int(v) for k, v in cards.grade.value_counts().sort_index().items()},
        "checked_files": {name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                          for name, path in paths.items()},
        "checks": [
            "all 500 source vehicle IDs appear exactly once in the minimal score CSV",
            "scores are finite and within [0,100]",
            "six deductions reconcile to the detailed safety score",
            "low evidence is assigned grade U",
            "every vehicle has all 47 daily anchors from day 14 through day 60",
            "24 event weights sum to one within every semantic state",
            "fleet summary totals 500 vehicles",
            "PDF hash matches the completed run manifest",
        ],
    }
    validation_path = run_dir / "validation_report.json"
    validation_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["validation_status"] = "passed"
    manifest["validation_report_sha256"] = sha256(validation_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), run_dir / "source" / Path(__file__).name)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--database-root", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    run_dir = resolve_run(args.run_dir, args.database_root)
    result = validate(run_dir, args.database_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
