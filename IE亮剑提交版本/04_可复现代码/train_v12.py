"""Run V12 state-MAE, future dynamics and human-guided risk-chain SSL experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from ie_alpaca.data.splits import audit_split, load_or_create_split
from ie_alpaca.data.task_one import labeled_reference, load_tables
from ie_alpaca.evaluation.metrics import auc_interval, score_binary
from ie_alpaca.features.daily_v10 import CONTEXT_COLUMNS, NodeDayScaler, build_day_table
from ie_alpaca.features.landmark_v3 import ANCHORS
from ie_alpaca.models.state_encoder_v12 import RiskNetV12
from ie_alpaca.models.state_mae_v12 import SSLModelV12
from ie_alpaca.tracking.run_store import create_run, file_manifest, update_leaderboard, verify_source, write_json, write_parquet
from ie_alpaca.training.event_v4 import resolve_device
from ie_alpaca.training.finetune_v12 import fit_risk_v12, predict_anchors_v12, predict_day60_v12
from ie_alpaca.training.landmark_v3 import horizon_probability, make_outcomes
from ie_alpaca.training.pretrain_v12 import pretrain_v12
from ie_alpaca.training.risk_chain_v10 import vehicle_pack


METHODS = ("state_mae", "state_shuffle", "dynamics", "hrc_no_prior", "hrc", "hrc_shuffle")


def paired_auc(current: pd.DataFrame, prior: pd.DataFrame, seed: int) -> dict:
    pair = current[["gpsno", "label", "probability"]].merge(
        prior[["gpsno", "label", "probability"]], on="gpsno", suffixes=("_new", "_prior"), validate="one_to_one")
    if not pair.label_new.eq(pair.label_prior).all():
        raise ValueError("paired labels differ")
    y = pair.label_new.to_numpy(int)
    new, old = pair.probability_new.to_numpy(), pair.probability_prior.to_numpy()
    rng = np.random.default_rng(seed); values = []
    for _ in range(2000):
        positions = rng.integers(0, len(y), len(y))
        if len(np.unique(y[positions])) == 2:
            values.append(roc_auc_score(y[positions], new[positions]) - roc_auc_score(y[positions], old[positions]))
    new_auc, old_auc = roc_auc_score(y, new), roc_auc_score(y, old)
    return {"vehicles": len(y), "new_auc": float(new_auc), "prior_auc": float(old_auc),
            "delta_auc": float(new_auc - old_auc),
            "delta_auc_95pct_paired_bootstrap": np.quantile(values, [.025, .975]).tolist()}


def read_reference(root: Path, run_id: str) -> pd.DataFrame:
    path = root / "runs" / run_id / "predictions" / "oof_landmarks.parquet"
    with duckdb.connect() as connection:
        return connection.execute(
            "SELECT gpsno,label,probability FROM read_parquet(?) WHERE anchor_day=20", [str(path)]).df()


def save_checkpoint(path: Path, net: RiskNetV12, scaler: NodeDayScaler, config: dict,
                    method: str, ssl_steps: int, epochs: int) -> None:
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                "scaler": scaler.describe(), "model_config": config, "method": method,
                "selected_ssl_steps": ssl_steps, "selected_finetune_epochs": epochs}, path)


def run(config_path: Path, database_root: Path) -> Path:
    config_path, database_root = config_path.resolve(), database_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    methods = tuple(config["methods"])
    if tuple(config["anchors"]) != ANCHORS or not methods or not set(methods).issubset(METHODS):
        raise ValueError("V12 frozen protocol mismatch")
    input_dir = database_root / "任务一预处理结果"
    results_root = database_root / "任务一实验结果"
    tables = load_tables(input_dir)
    reference = labeled_reference(tables.bags)
    split_path = results_root / "splits" / f"split_v1_seed{config['split_seed']}.json"
    split = load_or_create_split(reference, split_path, seed=config["split_seed"], folds=5, holdout_fraction=.2)
    audit, development_ids = audit_split(reference, split), set(split["development"])
    profile_ids = sorted(tables.profile.gpsno.astype(str))
    labels20 = reference.set_index("gpsno").label.astype(int)
    if (len(reference), len(development_ids), len(profile_ids)) != (478, 382, 500):
        raise ValueError("frozen vehicle counts changed")
    day_table = build_day_table(tables.daily, profile_ids)
    landmarks = pd.MultiIndex.from_product([sorted(development_ids), ANCHORS],
                                           names=["gpsno", "anchor_day"]).to_frame(index=False)
    outcomes = make_outcomes(tables.daily, landmarks)
    if not outcomes[outcomes.anchor_day.eq(20)].set_index("gpsno").label.eq(labels20.loc[sorted(development_ids)]).all():
        raise ValueError("day20 label contract changed")

    settings = config["model"]
    device = resolve_device(settings["device"])
    if device.type == "cpu":
        torch.set_num_threads(int(settings["cpu_threads"]))
    counts = {}
    for method in methods:
        ssl = SSLModelV12(len(CONTEXT_COLUMNS), method, int(settings["width"]), float(settings["dropout"]))
        downstream = RiskNetV12(len(CONTEXT_COLUMNS), method, int(settings["width"]), float(settings["dropout"]))
        counts[method] = {"ssl_trainable": sum(p.numel() for p in ssl.parameters() if p.requires_grad),
                          "backbone": sum(p.numel() for p in downstream.backbone.parameters()),
                          "downstream_total": sum(p.numel() for p in downstream.parameters())}
    data_files = file_manifest(input_dir)
    fingerprint = hashlib.sha256(json.dumps(data_files, sort_keys=True).encode()).hexdigest()
    resolved = {**config, "input_dir": str(input_dir), "results_root": str(results_root),
                "resolved_device": str(device), "split_version": split["version"], "parameter_count": counts}
    run_dir, manifest = create_run(REPO, results_root, config_path, resolved,
                                   version=str(config["run_prefix"]), entrypoint="train_v12.py")
    try:
        manifest.update({"model": "torch_v12_human_guided_risk_chain_ssl", "split_version": split["version"],
                         "data_files": data_files, "data_fingerprint": fingerprint,
                         "evaluation_version": "v12_strict_inductive_oof_survival_0.5",
                         "parameter_count": counts, "device": str(device),
                         "locked_labels_used_in_fit_or_metrics": False,
                         "dependencies": {name: importlib.metadata.version(name)
                                          for name in ("torch", "scikit-learn", "duckdb", "pandas", "numpy")}})
        write_json(run_dir / "manifest.json", manifest)
        shutil.copy2(split_path, run_dir / "splits.json")
        audit.update({"ssl_protocol": "outer-train vehicles only; future days are targets only",
                      "ssl_severe_events_as_future_target": False, "locked_holdout_scored": False,
                      "future_days_visible_to_anchor": False, "methods": list(methods)})
        write_json(run_dir / "leakage_audit.json", audit)

        oof_parts, pretrain_rows, finetune_rows, diagnostics, inner_splits = [], [], [], [], {}
        selected_epochs = {method: [] for method in methods}
        steps = int(settings["ssl_steps"])
        for fold_text, validation_list in sorted(split["validation_folds"].items(), key=lambda pair: int(pair[0])):
            fold = int(fold_text)
            validation_ids = set(validation_list)
            training_ids = development_ids - validation_ids
            ordered = np.asarray(sorted(training_ids))
            fit_array, inner_array = train_test_split(
                ordered, test_size=float(settings["inner_validation_fraction"]),
                random_state=config["seed"] + fold, stratify=labels20.loc[ordered])
            inner_fit, inner_validation = set(fit_array), set(inner_array)
            inner_splits[str(fold)] = {"fit": sorted(inner_fit), "validation": sorted(inner_validation)}
            if training_ids & validation_ids or inner_fit & inner_validation or inner_fit | inner_validation != training_ids:
                raise ValueError("vehicle isolation failed")

            outer_scaler = NodeDayScaler().fit(day_table, training_ids)
            outer_train = vehicle_pack(day_table, outcomes, training_ids, outer_scaler)
            outer_validation = vehicle_pack(day_table, outcomes, validation_ids, outer_scaler)
            inner_scaler = NodeDayScaler().fit(day_table, inner_fit)
            inner_train = vehicle_pack(day_table, outcomes, inner_fit, inner_scaler)
            inner_val = vehicle_pack(day_table, outcomes, inner_validation, inner_scaler)
            prediction_frame = outcomes[outcomes.gpsno.isin(validation_ids)].sort_values(
                ["gpsno", "anchor_day"]).copy()
            prediction_frame["fold"] = fold

            for method_index, method in enumerate(methods):
                started = time.perf_counter()
                inner_state, inner_history, inner_diagnostic = pretrain_v12(
                    inner_train, settings, method, steps,
                    config["seed"] + fold * 100 + method_index, device)
                for row in inner_history:
                    pretrain_rows.append({"fold": fold, "method": method,
                                          "phase": "inner_selection", **row})
                _, selection_history, epochs, _ = fit_risk_v12(
                    inner_train, settings, method, inner_state,
                    config["seed"] + fold * 1000 + method_index, device,
                    validation_pack=inner_val)
                selected_epochs[method].append(epochs)
                for row in selection_history:
                    finetune_rows.append({"fold": fold, "method": method,
                                          "phase_scope": "inner_selection", **row})

                outer_state, outer_history, outer_diagnostic = pretrain_v12(
                    outer_train, settings, method, steps,
                    config["seed"] + fold * 100 + method_index + 50000, device)
                for row in outer_history:
                    pretrain_rows.append({"fold": fold, "method": method,
                                          "phase": "outer_refit", **row})
                net, training_history, _, probe_state = fit_risk_v12(
                    outer_train, settings, method, outer_state,
                    config["seed"] + fold * 1000 + method_index + 50000,
                    device, epochs=epochs)
                for row in training_history:
                    finetune_rows.append({"fold": fold, "method": method,
                                          "phase_scope": "outer_refit", **row})

                daily_hazard = predict_anchors_v12(net, outer_validation, device,
                                                   int(settings["finetune_batch_size"]))
                probe = RiskNetV12(outer_train["context"].shape[-1], method,
                                   int(settings["width"]), float(settings["dropout"])).to(device)
                probe.load_state_dict(probe_state)
                probe_hazard = predict_anchors_v12(probe, outer_validation, device,
                                                   int(settings["finetune_batch_size"]))
                probe_probability = horizon_probability(probe_hazard[:, ANCHORS.index(20)], 40)
                probe_auc = float(roc_auc_score(outer_validation["label"][:, ANCHORS.index(20)],
                                                probe_probability))
                prediction_frame[f"daily_hazard_{method}"] = daily_hazard.reshape(-1)
                prediction_frame[f"probability_{method}"] = horizon_probability(
                    daily_hazard.reshape(-1), prediction_frame.horizon_days.to_numpy())
                save_checkpoint(run_dir / "models" / f"fold_{fold}_{method}.pt", net,
                                outer_scaler, settings, method, steps, epochs)
                torch.save(outer_state, run_dir / "models" / f"fold_{fold}_{method}_pretrained_backbone.pt")
                diagnostics.append({"fold": fold, "method": method,
                                    "inner": inner_diagnostic, "outer": outer_diagnostic,
                                    "ssl_steps": steps, "selected_epochs": epochs,
                                    "frozen_probe_20_to_40_auc": probe_auc,
                                    "wall_seconds": time.perf_counter() - started})
                print(f"V12 fold={fold} method={method} steps={steps} epochs={epochs} complete", flush=True)
                del inner_state, outer_state, net, probe
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            oof_parts.append(prediction_frame)

        write_json(run_dir / "pretrain_inner_splits.json", inner_splits)
        pd.DataFrame(pretrain_rows).to_csv(run_dir / "pretrain_history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(finetune_rows).to_csv(run_dir / "finetune_history.csv", index=False, encoding="utf-8-sig")
        write_json(run_dir / "representation_diagnostics.json", {"folds": diagnostics})
        oof = pd.concat(oof_parts).sort_values(["anchor_day", "gpsno"])
        if len(oof) != len(development_ids) * len(ANCHORS) or oof.duplicated(["gpsno", "anchor_day"]).any():
            raise ValueError("V12 OOF is incomplete")

        metrics = {"target": "right-censored stationary daily first-event hazard",
                   "holdout_evaluated": False, "by_method_and_anchor": {},
                   "selected_finetune_epochs": selected_epochs, "fixed_ssl_steps": steps,
                   "parameter_count": counts}
        for method in methods:
            metrics["by_method_and_anchor"][method] = {}
            for anchor, part in oof.groupby("anchor_day"):
                key = f"{anchor}_to_{int(part.horizon_days.iloc[0])}"
                score = score_binary(part.label, part[f"probability_{method}"])
                score["auc_95pct_bootstrap"] = auc_interval(part.label,
                                                              part[f"probability_{method}"],
                                                              config["seed"] + int(anchor) + methods.index(method))
                metrics["by_method_and_anchor"][method][key] = score
        at20 = oof[oof.anchor_day.eq(20)]
        baseline = at20[["gpsno", "label", "probability_state_mae"]].rename(
            columns={"probability_state_mae": "probability"})
        metrics["paired_vs_state_mae"] = {}
        metrics["fold_delta_auc_vs_state_mae"] = {}
        for method in methods:
            current = at20[["gpsno", "label", f"probability_{method}"]].rename(
                columns={f"probability_{method}": "probability"})
            metrics["paired_vs_state_mae"][method] = paired_auc(
                current, baseline, config["seed"] + methods.index(method))
            metrics["fold_delta_auc_vs_state_mae"][method] = [
                {"fold": int(fold),
                 "delta_auc": float(roc_auc_score(group.label, group[f"probability_{method}"]) -
                                    roc_auc_score(group.label, group.probability_state_mae))}
                for fold, group in at20.groupby("fold")
            ]
        auc = {method: metrics["by_method_and_anchor"][method]["20_to_40"]["roc_auc"]
               for method in methods}
        def comparison(new_method: str, old_method: str) -> dict:
            new = at20[["gpsno", "label", f"probability_{new_method}"]].rename(
                columns={f"probability_{new_method}": "probability"})
            old = at20[["gpsno", "label", f"probability_{old_method}"]].rename(
                columns={f"probability_{old_method}": "probability"})
            result = paired_auc(new, old, config["seed"] + methods.index(new_method) * 31 + methods.index(old_method))
            result["nonnegative_folds"] = sum(
                roc_auc_score(group.label, group[f"probability_{new_method}"]) >=
                roc_auc_score(group.label, group[f"probability_{old_method}"])
                for _, group in at20.groupby("fold"))
            result["brier_change"] = (metrics["by_method_and_anchor"][new_method]["20_to_40"]["brier"] -
                                      metrics["by_method_and_anchor"][old_method]["20_to_40"]["brier"])
            low, high = result["delta_auc_95pct_paired_bootstrap"]
            result["passes_effect_rule"] = bool(
                result["delta_auc"] > 0 and result["nonnegative_folds"] >= 3 and result["brier_change"] <= .005 and
                not (result["delta_auc"] < .01 and low <= 0 <= high))
            return result

        metrics["stage_comparisons"] = {
            "state_mae_vs_state_shuffle": comparison("state_mae", "state_shuffle"),
            "dynamics_vs_state_mae": comparison("dynamics", "state_mae"),
            "hrc_no_prior_vs_dynamics": comparison("hrc_no_prior", "dynamics"),
            "hrc_vs_dynamics": comparison("hrc", "dynamics"),
            "hrc_vs_hrc_no_prior": comparison("hrc", "hrc_no_prior"),
            "hrc_vs_hrc_shuffle": comparison("hrc", "hrc_shuffle"),
        }
        stage = metrics["stage_comparisons"]
        metrics["stage_gates"] = {
            "semantic_state_grouping_supported": stage["state_mae_vs_state_shuffle"]["passes_effect_rule"],
            "future_dynamics_supported": stage["dynamics_vs_state_mae"]["passes_effect_rule"],
            "data_driven_graph_supported": stage["hrc_no_prior_vs_dynamics"]["passes_effect_rule"],
            "directed_chain_supported": (stage["hrc_vs_dynamics"]["passes_effect_rule"] and
                                          stage["hrc_vs_hrc_shuffle"]["delta_auc"] > 0),
            "human_chain_prior_supported": stage["hrc_vs_hrc_no_prior"]["passes_effect_rule"],
        }
        primary = "state_mae"
        if metrics["stage_gates"]["future_dynamics_supported"]:
            primary = "dynamics"
        if metrics["stage_gates"]["data_driven_graph_supported"]:
            primary = "hrc_no_prior"
        if metrics["stage_gates"]["directed_chain_supported"]:
            primary = "hrc"
        oof["daily_hazard"] = oof[f"daily_hazard_{primary}"]
        oof["probability"] = oof[f"probability_{primary}"]
        write_parquet(run_dir / "predictions" / "oof_landmarks.parquet", oof)
        metrics["primary_method"] = primary
        metrics["development_oof"] = metrics["by_method_and_anchor"][primary]["20_to_40"]
        current = at20[["gpsno", "label", f"probability_{primary}"]].rename(
            columns={f"probability_{primary}": "probability"})
        for name, run_id in (("v11_masked_ensemble", config["v11_reference_run_id"]),
                             ("v3", config["v3_reference_run_id"])):
            metrics[f"paired_primary_vs_{name}"] = paired_auc(
                current, read_reference(results_root, run_id), config["seed"] + len(name))

        final_epochs = int(np.median(selected_epochs[primary]))
        final_scaler = NodeDayScaler().fit(day_table, development_ids)
        final_pack = vehicle_pack(day_table, outcomes, development_ids, final_scaler)
        final_state, final_pretrain_history, final_diagnostic = pretrain_v12(
            final_pack, settings, primary, steps, config["seed"] + 90000, device)
        final_net, final_history, _, _ = fit_risk_v12(
            final_pack, settings, primary, final_state, config["seed"] + 91000,
            device, epochs=final_epochs)
        save_checkpoint(run_dir / "models" / f"development_{primary}.pt", final_net,
                        final_scaler, settings, primary, steps, final_epochs)
        ordered_days = day_table.sort_values(["gpsno", "day_index"])
        events, context = final_scaler.transform(ordered_days)
        events = events.reshape(500, 60, events.shape[-2], events.shape[-1])
        context = context.reshape(500, 60, context.shape[-1])
        daily_hazard = predict_day60_v12(final_net, events, context, device,
                                         int(settings["finetune_batch_size"]))
        full_probability = horizon_probability(daily_hazard, 40)
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
                                  "label_status": "future_unknown", "method": primary})
        write_parquet(run_dir / "predictions" / "candidate_500.parquet", candidate)
        candidate[["gpsno", "probability", "prediction_at_0_5"]].to_csv(
            run_dir / "predictions" / "submission_candidate.csv", index=False, encoding="utf-8-sig")
        metrics["final_refit"] = {"method": primary, "ssl_steps": steps,
                                  "finetune_epochs": final_epochs,
                                  "representation": final_diagnostic,
                                  "candidate_routing": candidate.route.value_counts().to_dict()}
        metrics["warning"] = "Day60 future 40-day labels are unavailable; OOF uses observable earlier landmarks."
        write_json(run_dir / "metrics.json", metrics)
        lines = [f"# {manifest['run_id']}", "", "## 20→40 OOF", "",
                 "| Method | AUC | Brier | PR-AUC | ΔAUC vs State-MAE |",
                 "|---|---:|---:|---:|---:|"]
        for method in methods:
            score = metrics["by_method_and_anchor"][method]["20_to_40"]
            delta = metrics["paired_vs_state_mae"][method]["delta_auc"]
            lines.append(f"| {method} | {score['roc_auc']:.6f} | {score['brier']:.6f} | "
                         f"{score['pr_auc']:.6f} | {delta:+.6f} |")
        lines += ["", f"Primary method: **{primary}**", "",
                  f"Stage gates: `{json.dumps(metrics['stage_gates'], ensure_ascii=False)}`", "",
                  "96 locked vehicles were not evaluated.", ""]
        (run_dir / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
        verify_source(REPO, manifest)
        manifest.update({"status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
                         "primary_method": primary, "development_vehicles": 382, "holdout_vehicles": 96})
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
                        default=REPO / "configs/experiments/v12_risk_chain_ssl.json")
    arguments = parser.parse_args()
    print(run(arguments.config, arguments.database_root))

