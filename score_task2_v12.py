"""Generate the V12-based task-two driver safety scorecard and dynamic outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from ie_alpaca.data.task_one import labeled_reference, load_tables
from ie_alpaca.features.daily_v10 import CONTEXT_COLUMNS, NodeDayScaler, build_day_table
from ie_alpaca.features.daily_v12 import STATE_EVENT_CODES
from ie_alpaca.features.landmark_v4 import EVENT_CODES, EVENT_NAMES
from ie_alpaca.models.state_encoder_v12 import RiskNetV12
from ie_alpaca.task2.safety_score_v12 import (
    SCORE_STATES, build_reference, normalize_event_weights, prepare_daily_arrays,
    raw_state_signals, scorecards,
)
from ie_alpaca.training.event_v4 import resolve_device
from ie_alpaca.training.landmark_v3 import horizon_probability


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def read_parquet(path: Path) -> pd.DataFrame:
    with duckdb.connect() as connection:
        return connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()


def make_run(root: Path, prefix: str, config_path: Path) -> tuple[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    token = hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]
    run_id = f"{prefix}_{stamp}_{token}"
    run_dir = root / "runs" / run_id
    for name in ("deliverables", "predictions", "diagnostics", "source"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[RiskNetV12, NodeDayScaler, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    settings = checkpoint["model_config"]
    method = checkpoint["method"]
    if method != "state_mae":
        raise ValueError("task2 V12 scorecard requires the selected State-MAE checkpoint")
    net = RiskNetV12(len(CONTEXT_COLUMNS), method, int(settings["width"]),
                     float(settings["dropout"])).to(device)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net, NodeDayScaler.from_description(checkpoint["scaler"]), checkpoint


@torch.inference_mode()
def predict_probabilities(net: RiskNetV12, events: np.ndarray, context: np.ndarray,
                          anchors: tuple[int, ...], device: torch.device,
                          batch_size: int) -> np.ndarray:
    values = []
    for start in range(0, len(events), batch_size):
        stop = start + batch_size
        event = torch.as_tensor(events[start:stop], device=device)
        ctx = torch.as_tensor(context[start:stop], device=device)
        hazards = torch.sigmoid(net(event, ctx, anchors)[0]).cpu().numpy()
        values.append(horizon_probability(hazards, 40))
    return np.concatenate(values).astype(float)


def event_sensitivity(net: RiskNetV12, events: np.ndarray, context: np.ndarray,
                      base_probability: np.ndarray, raw_counts: np.ndarray,
                      device: torch.device, batch_size: int) -> pd.DataFrame:
    state_lookup = {code: state for state, codes in STATE_EVENT_CODES.items() for code in codes}
    rows = []
    for position, code in enumerate(EVENT_CODES):
        ablated = events.copy()
        ablated[:, :, position, :5] = 0.0
        prediction = predict_probabilities(net, ablated, context, (60,), device, batch_size)[:, 0]
        delta = base_probability - prediction
        active = raw_counts[:, :, position].sum(axis=1) > 0
        selected = delta[active] if active.any() else delta
        rows.append({
            "event_code": int(code),
            "event_name": EVENT_NAMES[code],
            "state": state_lookup[code],
            "active_vehicles": int(active.sum()),
            "mean_signed_probability_delta": float(selected.mean()) if len(selected) else 0.0,
            "mean_abs_probability_delta": float(np.abs(selected).mean()) if len(selected) else 0.0,
        })
    return pd.DataFrame(rows)


def add_learned_weight(table: pd.DataFrame, weights: dict[str, np.ndarray]) -> pd.DataFrame:
    lookup = {(state, code): float(value)
              for state, codes in STATE_EVENT_CODES.items()
              for code, value in zip(codes, weights[state])}
    table = table.copy()
    table["learned_weight_within_state"] = [lookup[(state, int(code))]
                                             for state, code in zip(table.state, table.event_code)]
    return table.sort_values(["state", "learned_weight_within_state"], ascending=[True, False])


def ranks_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = pd.Series(left).rank(method="average").to_numpy(float)
    y = pd.Series(right).rank(method="average").to_numpy(float)
    return float(np.corrcoef(x, y)[0, 1])


def compact_scores(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["gpsno", "safety_score"]].assign(
        safety_score=lambda x: x.safety_score.round(4))


def run(config_path: Path, database_root: Path) -> Path:
    config_path = config_path.resolve()
    database_root = database_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    max_points = {name: float(value) for name, value in config["max_deduction_points"].items()}
    if set(max_points) != set(SCORE_STATES) or abs(sum(max_points.values()) - 100) > 1e-8:
        raise ValueError("task2 configured deductions must cover six components and sum to 100")

    task1_run = database_root / "任务一实验结果" / "runs" / config["source_task1_run_id"]
    task2_root = database_root / "任务二安全评价结果"
    input_dir = database_root / "任务一预处理结果"
    manifest_path = task1_run / "manifest.json"
    checkpoint_path = task1_run / "models" / "development_state_mae.pt"
    if not manifest_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError("selected V12 task-one run is incomplete")
    parent_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("status") != "completed" or parent_manifest.get("primary_method") != "state_mae":
        raise ValueError("task2 source run is not the finalized V12 State-MAE")

    run_id, run_dir = make_run(task2_root, config["run_prefix"], config_path)
    manifest = {
        "run_id": run_id,
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_task1_run_id": config["source_task1_run_id"],
        "source_model_sha256": sha256(checkpoint_path),
        "config_sha256": sha256(config_path),
        "locked_task1_labels_used": False,
    }
    write_json(run_dir / "manifest.json", manifest)
    try:
        settings = config["model"]
        device = resolve_device(settings["device"])
        if device.type == "cpu":
            torch.set_num_threads(int(settings["cpu_threads"]))
        tables = load_tables(input_dir)
        profile_ids = sorted(tables.profile.gpsno.astype(str).tolist())
        if len(profile_ids) != 500 or len(set(profile_ids)) != 500:
            raise ValueError("task2 requires exactly 500 unique target vehicles")
        day_table = build_day_table(tables.daily, profile_ids)
        arrays = prepare_daily_arrays(day_table, profile_ids)
        event_source_ids = set(labeled_reference(tables.bags).gpsno.astype(str))
        event_feed = np.asarray([gpsno in event_source_ids for gpsno in profile_ids])

        split = json.loads((task1_run / "splits.json").read_text(encoding="utf-8"))
        development_ids = set(map(str, split["development"]))
        if len(development_ids) != 382 or development_ids & set(map(str, split["holdout"])):
            raise ValueError("invalid frozen vehicle split")

        net, scaler, checkpoint = load_model(checkpoint_path, device)
        ordered = day_table.sort_values(["gpsno", "day_index"])
        events, context = scaler.transform(ordered)
        events = events.reshape(500, 60, len(EVENT_CODES), -1)
        context = context.reshape(500, 60, -1)
        anchors = tuple(range(int(config["history_start_day"]), 61))
        probabilities = predict_probabilities(net, events, context, anchors, device,
                                               int(settings["batch_size"]))
        day60_probability = probabilities[:, anchors.index(60)]

        sensitivity = event_sensitivity(net, events, context, day60_probability,
                                        arrays.counts, device, int(settings["batch_size"]))
        learned_weights = normalize_event_weights(
            sensitivity, uniform_share=float(config["event_weight_uniform_share"]))
        sensitivity = add_learned_weight(sensitivity, learned_weights)
        sensitivity.to_csv(run_dir / "diagnostics" / "event_weights.csv", index=False, encoding="utf-8-sig")

        signals60, burden60 = raw_state_signals(arrays, 60, learned_weights)
        reference = build_reference(signals60, day60_probability, development_ids)
        write_json(run_dir / "diagnostics" / "score_reference.json", reference)

        oof = read_parquet(task1_run / "predictions" / "oof_landmarks.parquet")
        at20 = oof[oof.anchor_day.eq(20)][["gpsno", "label", "probability"]].copy()
        if len(at20) != 382 or at20.gpsno.duplicated().any():
            raise ValueError("V12 OOF anchor-20 predictions are incomplete")
        fallback_probability = float(at20.label.mean())

        final_cards, _ = scorecards(
            signals60, burden60, day60_probability, reference, event_feed,
            learned_weights, max_points, 60, fallback_probability)
        compact_scores(final_cards).to_csv(
            run_dir / "deliverables" / "task2_driver_safety_scores.csv", index=False, encoding="utf-8-sig")
        final_cards.to_csv(
            run_dir / "deliverables" / "task2_driver_scorecards.csv", index=False, encoding="utf-8-sig")
        sensitivity.to_csv(
            run_dir / "deliverables" / "task2_event_weights.csv", index=False, encoding="utf-8-sig")

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
        history.to_csv(run_dir / "deliverables" / "task2_daily_score_history.csv",
                       index=False, encoding="utf-8-sig")

        signals20, burden20 = raw_state_signals(arrays, 20, learned_weights)
        evaluation_probability = probabilities[:, anchors.index(20)].copy()
        id_position = {gpsno: index for index, gpsno in enumerate(profile_ids)}
        for row in at20.itertuples(index=False):
            evaluation_probability[id_position[str(row.gpsno)]] = float(row.probability)
        cards20, _ = scorecards(
            signals20, burden20, evaluation_probability, reference, event_feed,
            learned_weights, max_points, 20, fallback_probability)
        evaluation = cards20.merge(at20, on="gpsno", how="inner", validate="one_to_one")
        risk_score = 100.0 - evaluation.safety_score.to_numpy(float)
        score_auc = float(roc_auc_score(evaluation.label, risk_score))
        model_auc = float(roc_auc_score(evaluation.label, evaluation.probability))
        correlation = ranks_correlation(risk_score, evaluation.label.to_numpy(float))
        grade_outcomes = (evaluation.groupby("grade", observed=False)
                          .agg(vehicles=("gpsno", "size"), future_events=("label", "sum"),
                               mean_safety_score=("safety_score", "mean"))
                          .reset_index())
        grade_outcomes["future_event_rate"] = grade_outcomes.future_events / grade_outcomes.vehicles

        fleet_summary = (final_cards.groupby(["grade", "risk_level"], observed=False)
                         .agg(vehicles=("gpsno", "size"),
                              mean_score=("safety_score", "mean"),
                              mean_v12_risk=("v12_risk_probability", "mean"))
                         .reset_index())
        fleet_summary["share"] = fleet_summary.vehicles / len(final_cards)
        fleet_summary.to_csv(run_dir / "deliverables" / "task2_fleet_summary.csv",
                             index=False, encoding="utf-8-sig")
        grade_outcomes.to_csv(run_dir / "diagnostics" / "oof_grade_outcomes.csv",
                              index=False, encoding="utf-8-sig")

        metrics = {
            "vehicles": 500,
            "score_min": float(final_cards.safety_score.min()),
            "score_max": float(final_cards.safety_score.max()),
            "score_mean": float(final_cards.safety_score.mean()),
            "score_median": float(final_cards.safety_score.median()),
            "grade_counts": {str(k): int(v) for k, v in final_cards.grade.value_counts().sort_index().items()},
            "route_counts": {str(k): int(v) for k, v in final_cards.route.value_counts().items()},
            "evidence_counts": {str(k): int(v) for k, v in final_cards.confidence_level.value_counts().items()},
            "anchor20_oof": {
                "vehicles": int(len(evaluation)),
                "positive": int(evaluation.label.sum()),
                "safety_score_risk_auc": score_auc,
                "v12_probability_auc": model_auc,
                "rank_correlation_with_future_event": correlation,
            },
            "fallback_probability": fallback_probability,
            "score_formula": "100 - sum(max_points[state] * risk_percentile[state]^2)",
            "max_deduction_points": max_points,
            "warning": "Task-two risk differentiation is evaluated only on development OOF; task-one locked labels remain unopened.",
        }
        write_json(run_dir / "metrics.json", metrics)

        source_files = [
            Path(__file__), config_path,
            REPO / "src" / "ie_alpaca" / "task2" / "safety_score_v12.py",
            REPO / "src" / "ie_alpaca" / "features" / "daily_v12.py",
            REPO / "src" / "ie_alpaca" / "models" / "state_encoder_v12.py",
        ]
        for source in source_files:
            relative = source.relative_to(REPO) if source.is_relative_to(REPO) else Path(source.name)
            destination = run_dir / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        manifest.update({
            "status": "scored",
            "completed_scoring_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "vehicles": 500,
            "development_oof_auc": score_auc,
            "output_files": sorted(str(path.relative_to(run_dir))
                                   for path in (run_dir / "deliverables").glob("*.csv")),
        })
        write_json(run_dir / "manifest.json", manifest)
        write_json(task2_root / "latest.json", {"run_id": run_id, "run_dir": str(run_dir)})
        print(str(run_dir), flush=True)
        return run_dir
    except Exception:
        manifest.update({"status": "failed", "failed_utc": datetime.now(timezone.utc).isoformat(),
                         "error": traceback.format_exc()})
        write_json(run_dir / "manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "experiments" / "task2_v12_scorecard.json")
    parser.add_argument("--database-root", type=Path,
                        default=Path(r"E:\Databases\2026 清华IE亮剑-算法赛道赛题"))
    args = parser.parse_args()
    run(args.config, args.database_root)


if __name__ == "__main__":
    main()

