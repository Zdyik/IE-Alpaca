"""Reproduce both task outputs directly from the selected V12 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from ie_alpaca.data.task_one import labeled_reference, load_tables
from ie_alpaca.features.daily_v10 import build_day_table
from ie_alpaca.task2.safety_score_v12 import (
    build_reference, normalize_event_weights, prepare_daily_arrays, raw_state_signals, scorecards,
)
from ie_alpaca.training.event_v4 import resolve_device
from ie_alpaca.training.finetune_v12 import predict_day60_v12
from ie_alpaca.training.landmark_v3 import horizon_probability
from score_task2_v12 import add_learned_weight, event_sensitivity, load_model, predict_probabilities


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def generate(input_dir: Path, checkpoint_path: Path, split_path: Path,
             task2_config_path: Path, output_dir: Path, device_name: str = "auto") -> Path:
    input_dir, checkpoint_path = input_dir.resolve(), checkpoint_path.resolve()
    split_path, task2_config_path = split_path.resolve(), task2_config_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    config = json.loads(task2_config_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    development_ids = set(map(str, split["development"]))
    if len(development_ids) != 382:
        raise ValueError("portable inference requires the frozen 382-vehicle development reference")
    fallback_probability = float(config["fallback_probability"])
    max_points = {str(k): float(v) for k, v in config["max_deduction_points"].items()}

    device = resolve_device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(int(config["model"]["cpu_threads"]))
    tables = load_tables(input_dir)
    profile_ids = sorted(tables.profile.gpsno.astype(str).tolist())
    if len(profile_ids) != 500 or len(set(profile_ids)) != 500:
        raise ValueError("expected exactly 500 target vehicles")
    day_table = build_day_table(tables.daily, profile_ids)
    arrays = prepare_daily_arrays(day_table, profile_ids)
    event_source_ids = set(labeled_reference(tables.bags).gpsno.astype(str))
    event_feed = np.asarray([gpsno in event_source_ids for gpsno in profile_ids])

    net, scaler, _ = load_model(checkpoint_path, device)
    ordered = day_table.sort_values(["gpsno", "day_index"])
    events, context = scaler.transform(ordered)
    events = events.reshape(500, 60, events.shape[-2], events.shape[-1])
    context = context.reshape(500, 60, context.shape[-1])
    anchors = tuple(range(int(config["history_start_day"]), 61))
    probabilities = predict_probabilities(
        net, events, context, anchors, device, int(config["model"]["batch_size"]))
    day60_probability = probabilities[:, anchors.index(60)]
    task1_hazard = predict_day60_v12(net, events, context, device, batch_size=512)
    task1_day60_probability = horizon_probability(task1_hazard, 40)

    behavior = (day_table.groupby("gpsno")[["trajectory_recorded_today", "imu_recorded_today"]]
                .any().loc[profile_ids].any(axis=1).to_numpy())
    task1_probability = np.where(event_feed | behavior, task1_day60_probability, fallback_probability)
    task1 = pd.DataFrame({
        "gpsno": profile_ids,
        "prediction": (task1_probability >= .5).astype(int),
        "probability": task1_probability,
    }).sort_values("gpsno")
    write_csv(task1, output_dir / "task1_predictions.csv")

    sensitivity = event_sensitivity(
        net, events, context, day60_probability, arrays.counts, device,
        int(config["model"]["batch_size"]))
    learned_weights = normalize_event_weights(
        sensitivity, uniform_share=float(config["event_weight_uniform_share"]))
    sensitivity = add_learned_weight(sensitivity, learned_weights)
    signals60, burden60 = raw_state_signals(arrays, 60, learned_weights)
    reference = build_reference(signals60, day60_probability, development_ids)
    final_cards, _ = scorecards(
        signals60, burden60, day60_probability, reference, event_feed,
        learned_weights, max_points, 60, fallback_probability)
    task2 = final_cards[["gpsno", "safety_score"]].copy()
    task2["safety_score"] = task2.safety_score.round(4)
    write_csv(task2, output_dir / "task2_safety_scores.csv")
    write_csv(final_cards, output_dir / "task2_scorecards.csv")
    write_csv(sensitivity, output_dir / "task2_event_weights.csv")

    history_frames = []
    for column, anchor in enumerate(anchors):
        signals, burden = raw_state_signals(arrays, anchor, learned_weights)
        cards, _ = scorecards(
            signals, burden, probabilities[:, column], reference, event_feed,
            learned_weights, max_points, anchor, fallback_probability)
        history_frames.append(cards[[
            "gpsno", "anchor_day", "as_of_date", "safety_score", "grade", "risk_level",
            "v12_risk_probability", "evidence_confidence", "confidence_level", "route",
            "model_deduction", "upstream_deduction", "control_deduction",
            "proximal_deduction", "history_deduction", "quality_deduction",
        ]])
    history = pd.concat(history_frames, ignore_index=True)
    write_csv(history, output_dir / "task2_daily_score_history.csv")

    fleet = (final_cards.groupby(["grade", "risk_level"], observed=False)
             .agg(vehicles=("gpsno", "size"), mean_score=("safety_score", "mean"),
                  mean_v12_risk=("v12_risk_probability", "mean")).reset_index())
    fleet["share"] = fleet.vehicles / len(final_cards)
    write_csv(fleet, output_dir / "task2_fleet_summary.csv")
    (output_dir / "generation_manifest.json").write_text(json.dumps({
        "checkpoint": str(checkpoint_path),
        "split": str(split_path),
        "task2_config": str(task2_config_path),
        "device": str(device),
        "vehicles": 500,
        "daily_rows": int(len(history)),
        "fallback_probability": fallback_probability,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="directory containing profile.parquet, bag_index.parquet and daily_features.parquet")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--task2-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    print(generate(args.input, args.checkpoint, args.splits, args.task2_config,
                   args.output, args.device))


if __name__ == "__main__":
    main()
