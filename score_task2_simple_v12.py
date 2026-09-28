"""Calculate the minimal six-dimension scorecard and compare it with V12 risk predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from ie_alpaca.data.task_one import labeled_reference, load_tables
from ie_alpaca.features.daily_v10 import build_day_table
from ie_alpaca.task2.safety_score_v12 import prepare_daily_arrays
from ie_alpaca.task2.simple_score_v12 import DIMENSION_LABELS, build_simple_scorecards
from ie_alpaca.training.event_v4 import resolve_device
from ie_alpaca.training.finetune_v12 import predict_day60_v12
from ie_alpaca.training.landmark_v3 import horizon_probability
from ie_alpaca.task2.v12_sensitivity import event_sensitivity, load_model


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def overlap(frame: pd.DataFrame, fraction: float) -> float:
    count = int(len(frame) * fraction)
    model = set(frame.nlargest(count, "v12_risk_probability").gpsno)
    score = set(frame.nsmallest(count, "safety_score").gpsno)
    return len(model & score) / count


def run(config_path: Path, database_root: Path) -> Path:
    config_path, database_root = config_path.resolve(), database_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task1_run = database_root / "任务一实验结果" / "runs" / config["source_task1_run_id"]
    checkpoint = task1_run / "models" / "development_state_mae.pt"
    split = json.loads((task1_run / "splits.json").read_text(encoding="utf-8"))
    development_ids = set(map(str, split["development"]))
    root = database_root / "任务二安全评价结果" / "runs"
    token = hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]
    run_id = f"{config['run_prefix']}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{token}"
    run_dir = root / run_id
    for name in ("deliverables", "diagnostics", "source"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)

    settings = config["model"]
    device = resolve_device(settings["device"])
    if device.type == "cpu":
        torch.set_num_threads(int(settings["cpu_threads"]))
    tables = load_tables(database_root / "任务一预处理结果")
    ids = sorted(tables.profile.gpsno.astype(str).tolist())
    day_table = build_day_table(tables.daily, ids)
    arrays = prepare_daily_arrays(day_table, ids)
    event_source = set(labeled_reference(tables.bags).gpsno.astype(str))
    event_feed = np.asarray([gpsno in event_source for gpsno in ids])

    net, scaler, _ = load_model(checkpoint, device)
    ordered = day_table.sort_values(["gpsno", "day_index"])
    events, context = scaler.transform(ordered)
    events = events.reshape(500, 60, events.shape[-2], events.shape[-1])
    context = context.reshape(500, 60, context.shape[-1])
    hazard = predict_day60_v12(net, events, context, device, batch_size=512)
    raw_probability = horizon_probability(hazard, 40)
    behavior = (day_table.groupby("gpsno")[["trajectory_recorded_today", "imu_recorded_today"]]
                .any().loc[ids].any(axis=1).to_numpy())
    probability = np.where(event_feed | behavior, raw_probability, float(config["fallback_probability"]))
    sensitivity = event_sensitivity(net, events, context, raw_probability, arrays.counts,
                                    device, int(settings["batch_size"]))
    cards, event_weights = build_simple_scorecards(
        arrays, 60, development_ids, event_feed, sensitivity, config)
    cards["v12_risk_probability"] = probability
    cards["risk_rank"] = cards.safety_score.rank(method="min", ascending=True).astype(int)
    cards["safety_rank"] = cards.safety_score.rank(method="min", ascending=False).astype(int)

    compact = cards[["gpsno", "safety_score"]].copy()
    compact["safety_score"] = compact.safety_score.round(4)
    compact.to_csv(run_dir / "deliverables" / "task2_safety_scores.csv", index=False, encoding="utf-8-sig")
    cards.to_csv(run_dir / "deliverables" / "task2_driver_scorecards.csv", index=False, encoding="utf-8-sig")
    event_weights.to_csv(run_dir / "deliverables" / "task2_event_weights.csv", index=False, encoding="utf-8-sig")

    evidence = cards[cards.confidence_level.ne("低")].copy()
    risk_score = 100 - cards.safety_score
    evidence_risk = 100 - evidence.safety_score
    pearson = float(risk_score.corr(cards.v12_risk_probability, method="pearson"))
    spearman = float(risk_score.corr(cards.v12_risk_probability, method="spearman"))
    evidence_spearman = float(evidence_risk.corr(evidence.v12_risk_probability, method="spearman"))
    grade_summary = (cards.groupby(["grade", "risk_level"], observed=False)
                     .agg(vehicles=("gpsno", "size"), mean_score=("safety_score", "mean"),
                          mean_v12_risk=("v12_risk_probability", "mean"))
                     .reset_index())
    grade_summary.to_csv(run_dir / "deliverables" / "task2_grade_summary.csv",
                         index=False, encoding="utf-8-sig")
    grade_risk = grade_summary.set_index("grade").mean_v12_risk.to_dict()
    gate = config["alignment_gate"]
    e_minus_a = float(grade_risk.get("E", np.nan) - grade_risk.get("A", np.nan))
    top20 = overlap(cards, .20)
    passed = bool(spearman >= gate["minimum_spearman"] and
                  top20 >= gate["minimum_top20_overlap"] and
                  e_minus_a >= gate["minimum_E_minus_A_probability"])
    metrics = {
        "vehicles": 500,
        "score_min": float(cards.safety_score.min()),
        "score_max": float(cards.safety_score.max()),
        "score_mean": float(cards.safety_score.mean()),
        "score_median": float(cards.safety_score.median()),
        "grade_counts": {str(k): int(v) for k, v in cards.grade.value_counts().sort_index().items()},
        "mean_v12_risk_by_grade": {
            str(k): float(v) for k, v in grade_summary.set_index("grade").mean_v12_risk.items()
        },
        "alignment_with_v12": {
            "pearson_risk_correlation": pearson,
            "spearman_rank_correlation": spearman,
            "spearman_evidence_vehicles": evidence_spearman,
            "top10_risk_overlap": overlap(cards, .10),
            "top20_risk_overlap": top20,
            "top30_risk_overlap": overlap(cards, .30),
            "E_minus_A_mean_probability": e_minus_a,
            "gate_passed": passed,
            "gate": gate,
        },
        "dimension_points": config["dimension_points"],
        "interpretation": ("The simple hierarchical scorecard is sufficiently aligned with V12."
                           if passed else "Alignment is insufficient; do not claim V12 consistency."),
    }
    write_json(run_dir / "metrics.json", metrics)
    manifest = {
        "run_id": run_id, "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_task1_run_id": config["source_task1_run_id"], "device": str(device),
        "vehicles": 500, "alignment_gate_passed": passed,
        "warning": "Alignment measures agreement with V12 Day60 predictions, not independent ground-truth accuracy.",
    }
    write_json(run_dir / "manifest.json", manifest)
    for source in (Path(__file__), config_path, REPO / "src" / "ie_alpaca" / "task2" / "simple_score_v12.py"):
        destination = run_dir / "source" / (source.relative_to(REPO) if source.is_relative_to(REPO) else Path(source.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=REPO / "configs" / "experiments" / "task2_simple_score_v12.json")
    parser.add_argument("--database-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    run(args.config, args.database_root)


if __name__ == "__main__":
    main()
