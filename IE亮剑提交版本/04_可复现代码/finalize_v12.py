"""Finalize the V12 method selected by the frozen effect-size stopping rule."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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

from ie_alpaca.data.splits import audit_split, load_or_create_split
from ie_alpaca.data.task_one import labeled_reference, load_tables
from ie_alpaca.evaluation.metrics import auc_interval, score_binary
from ie_alpaca.features.daily_v10 import NodeDayScaler, build_day_table
from ie_alpaca.features.landmark_v3 import ANCHORS
from ie_alpaca.tracking.run_store import create_run, file_manifest, update_leaderboard, verify_source, write_json, write_parquet
from ie_alpaca.training.event_v4 import resolve_device
from ie_alpaca.training.finetune_v12 import fit_risk_v12, predict_day60_v12
from ie_alpaca.training.landmark_v3 import horizon_probability, make_outcomes
from ie_alpaca.training.pretrain_v12 import pretrain_v12
from ie_alpaca.training.risk_chain_v10 import vehicle_pack


def read_parquet(path: Path) -> pd.DataFrame:
    with duckdb.connect() as connection:
        return connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()


def read_reference(root: Path, run_id: str) -> pd.DataFrame:
    frame = read_parquet(root / "runs" / run_id / "predictions" / "oof_landmarks.parquet")
    return frame[frame.anchor_day.eq(20)][["gpsno", "label", "probability"]]


def paired_auc(current: pd.DataFrame, prior: pd.DataFrame, seed: int) -> dict:
    pair = current.merge(prior, on=["gpsno", "label"], suffixes=("_new", "_prior"), validate="one_to_one")
    y = pair.label.to_numpy(int)
    new, old = pair.probability_new.to_numpy(), pair.probability_prior.to_numpy()
    rng = np.random.default_rng(seed); values = []
    for _ in range(2000):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) == 2:
            values.append(roc_auc_score(y[index], new[index]) - roc_auc_score(y[index], old[index]))
    new_auc, old_auc = roc_auc_score(y, new), roc_auc_score(y, old)
    return {"vehicles": len(y), "new_auc": float(new_auc), "prior_auc": float(old_auc),
            "delta_auc": float(new_auc - old_auc),
            "delta_auc_95pct_paired_bootstrap": np.quantile(values, [.025, .975]).tolist()}


def run(config_path: Path, database_root: Path) -> Path:
    config_path, database_root = config_path.resolve(), database_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if tuple(config["anchors"]) != ANCHORS or config["selected_method"] != "state_mae":
        raise ValueError("V12 finalization contract changed")
    input_dir, results_root = database_root / "任务一预处理结果", database_root / "任务一实验结果"
    parent = results_root / "runs" / config["parent_run_id"]
    parent_manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    parent_metrics = json.loads((parent / "metrics.json").read_text(encoding="utf-8"))
    if parent_manifest.get("status") != "completed" or parent_manifest.get("locked_labels_used_in_fit_or_metrics"):
        raise ValueError("invalid V12 parent experiment")
    method = config["selected_method"]
    state_vs_dynamics = parent_metrics["paired_vs_state_mae"]["dynamics"]
    low, high = state_vs_dynamics["delta_auc_95pct_paired_bootstrap"]
    if not (state_vs_dynamics["delta_auc"] < .01 and low <= 0 <= high):
        raise ValueError("parent result no longer satisfies the simpler-model stop rule")

    tables = load_tables(input_dir)
    reference = labeled_reference(tables.bags)
    split_path = results_root / "splits" / f"split_v1_seed{config['split_seed']}.json"
    split = load_or_create_split(reference, split_path, seed=config["split_seed"], folds=5, holdout_fraction=.2)
    audit, development_ids = audit_split(reference, split), set(split["development"])
    profile_ids = sorted(tables.profile.gpsno.astype(str))
    day_table = build_day_table(tables.daily, profile_ids)
    landmarks = pd.MultiIndex.from_product([sorted(development_ids), ANCHORS],
                                           names=["gpsno", "anchor_day"]).to_frame(index=False)
    outcomes = make_outcomes(tables.daily, landmarks)
    settings = config["model"]
    device = resolve_device(settings["device"])
    if device.type == "cpu":
        torch.set_num_threads(int(settings["cpu_threads"]))
    data_files = file_manifest(input_dir)
    fingerprint = hashlib.sha256(json.dumps(data_files, sort_keys=True).encode()).hexdigest()
    if fingerprint != parent_manifest["data_fingerprint"]:
        raise ValueError("parent data fingerprint differs")
    selected_epochs = int(np.median(parent_metrics["selected_finetune_epochs"][method]))
    resolved = {**config, "input_dir": str(input_dir), "results_root": str(results_root),
                "resolved_device": str(device), "selected_finetune_epochs": selected_epochs,
                "selection_evidence": state_vs_dynamics}
    run_dir, manifest = create_run(REPO, results_root, config_path, resolved,
                                   version=config["run_prefix"], entrypoint="finalize_v12.py")
    try:
        manifest.update({"model": "torch_v12_selected_state_mae", "split_version": split["version"],
                         "data_files": data_files, "data_fingerprint": fingerprint,
                         "evaluation_version": "v12_selected_strict_inductive_oof_survival_0.5",
                         "device": str(device), "parent_run_id": parent.name,
                         "locked_labels_used_in_fit_or_metrics": False,
                         "dependencies": {name: importlib.metadata.version(name)
                                          for name in ("torch", "scikit-learn", "duckdb", "pandas", "numpy")}})
        write_json(run_dir / "manifest.json", manifest)
        shutil.copy2(split_path, run_dir / "splits.json")
        audit.update({"parent_run_id": parent.name, "selected_method": method,
                      "selection_reason": config["selection_reason"],
                      "locked_holdout_scored": False, "future_days_visible_to_anchor": False})
        write_json(run_dir / "leakage_audit.json", audit)

        oof = read_parquet(parent / "predictions" / "oof_landmarks.parquet")
        oof["daily_hazard"] = oof[f"daily_hazard_{method}"]
        oof["probability"] = oof[f"probability_{method}"]
        write_parquet(run_dir / "predictions" / "oof_landmarks.parquet", oof)
        metrics = {"target": parent_metrics["target"], "holdout_evaluated": False,
                   "primary_method": method, "parent_run_id": parent.name,
                   "selection_reason": config["selection_reason"], "by_anchor": {},
                   "all_parent_methods": parent_metrics["by_method_and_anchor"],
                   "parent_stage_gates_reported": parent_metrics["stage_gates"],
                   "stage_gates_after_effect_rule": {
                       "semantic_state_grouping_supported": True,
                       "future_dynamics_supported": False,
                       "data_driven_graph_supported": False,
                       "directed_chain_supported": False,
                       "human_chain_prior_supported": False,
                   }}
        for anchor, part in oof.groupby("anchor_day"):
            key = f"{anchor}_to_{int(part.horizon_days.iloc[0])}"
            score = score_binary(part.label, part.probability)
            score["auc_95pct_bootstrap"] = auc_interval(part.label, part.probability,
                                                          config["seed"] + int(anchor))
            metrics["by_anchor"][key] = score
        metrics["development_oof"] = metrics["by_anchor"]["20_to_40"]
        current = oof[oof.anchor_day.eq(20)][["gpsno", "label", "probability"]]
        metrics["paired_vs_parent_dynamics"] = paired_auc(
            current, oof[oof.anchor_day.eq(20)][["gpsno", "label", "probability_dynamics"]].rename(
                columns={"probability_dynamics": "probability"}), config["seed"] + 1)
        for name, run_id in (("v11_masked_ensemble", config["v11_reference_run_id"]),
                             ("v3", config["v3_reference_run_id"])):
            metrics[f"paired_vs_{name}"] = paired_auc(
                current, read_reference(results_root, run_id), config["seed"] + len(name))

        scaler = NodeDayScaler().fit(day_table, development_ids)
        pack = vehicle_pack(day_table, outcomes, development_ids, scaler)
        backbone_state, pretrain_history, diagnostics = pretrain_v12(
            pack, settings, method, int(settings["ssl_steps"]), config["seed"] + 90000, device)
        net, finetune_history, _, _ = fit_risk_v12(
            pack, settings, method, backbone_state, config["seed"] + 91000,
            device, epochs=selected_epochs)
        torch.save({"state_dict": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                    "scaler": scaler.describe(), "model_config": settings, "method": method,
                    "selected_ssl_steps": int(settings["ssl_steps"]),
                    "selected_finetune_epochs": selected_epochs},
                   run_dir / "models" / f"development_{method}.pt")
        pd.DataFrame(pretrain_history).to_csv(run_dir / "pretrain_history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(finetune_history).to_csv(run_dir / "finetune_history.csv", index=False, encoding="utf-8-sig")
        write_json(run_dir / "representation_diagnostics.json", diagnostics)

        ordered = day_table.sort_values(["gpsno", "day_index"])
        events, context = scaler.transform(ordered)
        events = events.reshape(500, 60, events.shape[-2], events.shape[-1])
        context = context.reshape(500, 60, context.shape[-1])
        hazard = predict_day60_v12(net, events, context, device, int(settings["finetune_batch_size"]))
        full_probability = horizon_probability(hazard, 40)
        matched = np.asarray([gpsno in set(reference.gpsno) for gpsno in profile_ids])
        behavior = day_table.groupby("gpsno")[["trajectory_recorded_today", "imu_recorded_today"]].any().loc[
            profile_ids].any(axis=1).to_numpy()
        prior = float(outcomes[outcomes.anchor_day.eq(20)].label.mean())
        probability = np.where(matched | behavior, full_probability, prior)
        candidate = pd.DataFrame({"gpsno": profile_ids, "anchor_day": 60, "horizon_days": 40,
                                  "route": np.where(matched, "full", np.where(
                                      behavior, "context_only_no_matched_event", "prior_no_observed_behavior")),
                                  "probability": probability,
                                  "prediction_at_0_5": (probability >= .5).astype(int),
                                  "full_probability": full_probability,
                                  "label_status": "future_unknown", "method": method})
        write_parquet(run_dir / "predictions" / "candidate_500.parquet", candidate)
        candidate[["gpsno", "probability", "prediction_at_0_5"]].to_csv(
            run_dir / "predictions" / "submission_candidate.csv", index=False, encoding="utf-8-sig")
        metrics["final_refit"] = {"method": method, "ssl_steps": int(settings["ssl_steps"]),
                                  "finetune_epochs": selected_epochs, "representation": diagnostics,
                                  "candidate_routing": candidate.route.value_counts().to_dict()}
        metrics["warning"] = "Day60 future 40-day labels are unavailable; 96 locked labels were not evaluated."
        write_json(run_dir / "metrics.json", metrics)
        score = metrics["development_oof"]
        lines = [f"# {manifest['run_id']}", "", "V12 stopping-rule finalization.", "",
                 f"- Selected method: **{method}**", f"- 20→40 OOF AUC: {score['roc_auc']:.6f}",
                 f"- PR-AUC: {score['pr_auc']:.6f}", f"- Brier: {score['brier']:.6f}",
                 f"- Reason: {config['selection_reason']}", "", "96 locked vehicles were not evaluated.", ""]
        (run_dir / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
        verify_source(REPO, manifest)
        manifest.update({"status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
                         "primary_method": method, "development_vehicles": 382, "holdout_vehicles": 96})
        write_json(run_dir / "manifest.json", manifest)
        update_leaderboard(results_root)
    except Exception:
        manifest.update({"status": "failed", "failed_utc": datetime.now(timezone.utc).isoformat(),
                         "error": traceback.format_exc()})
        write_json(run_dir / "manifest.json", manifest)
        raise
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-root", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path,
                        default=REPO / "configs/experiments/v12_finalize_state_mae.json")
    arguments = parser.parse_args()
    print(run(arguments.config, arguments.database_root))

