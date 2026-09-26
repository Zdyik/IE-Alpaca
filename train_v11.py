"""Run V11 Scratch, masked reconstruction, contrastive and causal JEPA experiments."""

from __future__ import annotations

import argparse, hashlib, importlib.metadata, json, shutil, sys, time, traceback
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
from ie_alpaca.models.representation_v11 import RepresentationEncoderV11, RiskNetV11
from ie_alpaca.tracking.run_store import create_run, file_manifest, update_leaderboard, verify_source, write_json, write_parquet
from ie_alpaca.training.event_v4 import resolve_device
from ie_alpaca.training.finetune_v11 import fit_risk, predict_anchors, predict_day60
from ie_alpaca.training.landmark_v3 import horizon_probability, make_outcomes
from ie_alpaca.training.pretrain_v11 import pretrain
from ie_alpaca.training.risk_chain_v10 import vehicle_pack


METHODS = ("scratch", "masked", "contrastive", "jepa")


def paired_auc(current: pd.DataFrame, prior: pd.DataFrame, seed: int) -> dict:
    pair = current[["gpsno", "label", "probability"]].merge(prior[["gpsno", "label", "probability"]], on="gpsno",
                                                               suffixes=("_new", "_prior"), validate="one_to_one")
    if not pair.label_new.eq(pair.label_prior).all(): raise ValueError("paired labels differ")
    y = pair.label_new.to_numpy(int); a = pair.probability_new.to_numpy(); b = pair.probability_prior.to_numpy()
    rng = np.random.default_rng(seed); deltas = []
    for _ in range(2000):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) == 2: deltas.append(roc_auc_score(y[index], a[index]) - roc_auc_score(y[index], b[index]))
    return {"vehicles": len(y), "new_auc": float(roc_auc_score(y, a)), "prior_auc": float(roc_auc_score(y, b)),
            "delta_auc": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "delta_auc_95pct_paired_bootstrap": np.quantile(deltas, [.025, .975]).tolist()}


def read_reference(root: Path, run_id: str) -> pd.DataFrame:
    path = root / "runs" / run_id / "predictions" / "oof_landmarks.parquet"
    with duckdb.connect() as con:
        return con.execute("SELECT gpsno,label,probability FROM read_parquet(?) WHERE anchor_day=20", [str(path)]).df()


def encoder_state(encoder):
    return {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}


def save_checkpoint(path, net, scaler, config, method, selected_steps, selected_epochs):
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "scaler": scaler.describe(),
                "model_config": config, "method": method, "selected_ssl_steps": selected_steps,
                "selected_finetune_epochs": selected_epochs}, path)


def run(config_path: Path, database_root: Path) -> Path:
    config_path, database_root = config_path.resolve(), database_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8")); methods = tuple(config["methods"])
    if tuple(config["anchors"]) != ANCHORS or not set(methods).issubset(METHODS): raise ValueError("V11 frozen protocol mismatch")
    input_dir, results_root = database_root / "任务一预处理结果", database_root / "任务一实验结果"
    tables = load_tables(input_dir); reference = labeled_reference(tables.bags)
    split_path = results_root / "splits" / f"split_v1_seed{config['split_seed']}.json"
    split = load_or_create_split(reference, split_path, seed=config["split_seed"], folds=5, holdout_fraction=.2)
    audit, dev_ids = audit_split(reference, split), set(split["development"])
    profile_ids = sorted(tables.profile.gpsno.astype(str)); labels20 = reference.set_index("gpsno").label.astype(int)
    if (len(reference), len(dev_ids), len(profile_ids)) != (478, 382, 500): raise ValueError("frozen vehicle counts changed")
    day_table = build_day_table(tables.daily, profile_ids)
    landmarks = pd.MultiIndex.from_product([sorted(dev_ids), ANCHORS], names=["gpsno", "anchor_day"]).to_frame(index=False)
    outcomes = make_outcomes(tables.daily, landmarks)
    if not outcomes[outcomes.anchor_day.eq(20)].set_index("gpsno").label.eq(labels20.loc[sorted(dev_ids)]).all():
        raise ValueError("day20 label contract changed")
    settings = config["model"]; device = resolve_device(settings["device"])
    if device.type == "cpu": torch.set_num_threads(int(settings["cpu_threads"]))
    sample_encoder = RepresentationEncoderV11(len(CONTEXT_COLUMNS), int(settings["width"]), float(settings["dropout"]))
    sample_net = RiskNetV11(len(CONTEXT_COLUMNS), int(settings["width"]), float(settings["dropout"]))
    counts = {"encoder": sum(p.numel() for p in sample_encoder.parameters()), "downstream_total": sum(p.numel() for p in sample_net.parameters())}
    data_files = file_manifest(input_dir); fingerprint = hashlib.sha256(json.dumps(data_files, sort_keys=True).encode()).hexdigest()
    resolved = {**config, "input_dir": str(input_dir), "results_root": str(results_root), "resolved_device": str(device),
                "split_version": split["version"], "parameter_count": counts}
    run_dir, manifest = create_run(REPO, results_root, config_path, resolved, version=str(config["run_prefix"]), entrypoint="train_v11.py")
    try:
        manifest.update({"model": "torch_v11_self_supervised_risk_chain", "split_version": split["version"], "data_files": data_files,
                         "data_fingerprint": fingerprint, "evaluation_version": "v11_strict_inductive_oof_survival_0.5",
                         "parameter_count": counts, "device": str(device), "locked_labels_used_in_fit_or_metrics": False,
                         "dependencies": {x: importlib.metadata.version(x) for x in ("torch", "scikit-learn", "duckdb", "pandas", "numpy")}})
        write_json(run_dir / "manifest.json", manifest); shutil.copy2(split_path, run_dir / "splits.json")
        audit.update({"ssl_protocol": "outer-train vehicles only", "locked_holdout_scored": False,
                      "future_days_visible_to_anchor": False, "methods": list(methods)})
        write_json(run_dir / "leakage_audit.json", audit)
        oof_parts, pretrain_rows, finetune_rows, diagnostics, inner_splits = [], [], [], [], {}
        selected_by_method = {m: {"steps": [], "epochs": []} for m in methods}
        for fold_text, val_list in sorted(split["validation_folds"].items(), key=lambda x: int(x[0])):
            fold = int(fold_text); val_ids = set(val_list); train_ids = dev_ids - val_ids
            ordered = np.asarray(sorted(train_ids)); fit_array, inner_array = train_test_split(
                ordered, test_size=float(settings["inner_validation_fraction"]), random_state=config["seed"] + fold,
                stratify=labels20.loc[ordered])
            inner_fit, inner_val = set(fit_array), set(inner_array)
            inner_splits[str(fold)] = {"fit": sorted(inner_fit), "validation": sorted(inner_val)}
            if (train_ids & val_ids) or (inner_fit & inner_val) or inner_fit | inner_val != train_ids: raise ValueError("vehicle isolation failed")
            outer_scaler = NodeDayScaler().fit(day_table, train_ids)
            outer_train = vehicle_pack(day_table, outcomes, train_ids, outer_scaler)
            outer_val = vehicle_pack(day_table, outcomes, val_ids, outer_scaler)
            inner_scaler = NodeDayScaler().fit(day_table, inner_fit)
            inner_train = vehicle_pack(day_table, outcomes, inner_fit, inner_scaler)
            inner_validation = vehicle_pack(day_table, outcomes, inner_val, inner_scaler)
            prediction_frame = outcomes[outcomes.gpsno.isin(val_ids)].sort_values(["gpsno", "anchor_day"]).copy(); prediction_frame["fold"] = fold
            for method in methods:
                started = time.perf_counter()
                max_steps = 0 if method == "scratch" else int(settings["ssl_max_steps"])
                inner_encoder, inner_history, inner_diag = pretrain(inner_train, settings, method, max_steps,
                                                                     config["seed"] + fold * 100 + METHODS.index(method), device,
                                                                     inner_validation if method != "scratch" else None)
                steps = int(inner_diag.get("selected_steps", 0)); selected_by_method[method]["steps"].append(steps)
                for row in inner_history: pretrain_rows.append({"fold": fold, "method": method, "phase": "inner_selection", **row})
                _, selection_history, epochs, _ = fit_risk(inner_train, settings, encoder_state(inner_encoder),
                                                            config["seed"] + fold * 1000 + METHODS.index(method), device,
                                                            validation_pack=inner_validation)
                selected_by_method[method]["epochs"].append(epochs)
                for row in selection_history: finetune_rows.append({"fold": fold, "method": method, "phase_scope": "inner_selection", **row})
                outer_encoder, outer_history, outer_diag = pretrain(outer_train, settings, method, steps,
                                                                     config["seed"] + fold * 100 + METHODS.index(method) + 50000, device)
                for row in outer_history: pretrain_rows.append({"fold": fold, "method": method, "phase": "outer_refit", **row})
                net, training_history, _, probe_state = fit_risk(outer_train, settings, encoder_state(outer_encoder),
                                                                  config["seed"] + fold * 1000 + METHODS.index(method) + 50000,
                                                                  device, epochs=epochs)
                for row in training_history: finetune_rows.append({"fold": fold, "method": method, "phase_scope": "outer_refit", **row})
                q = predict_anchors(net, outer_val, device, int(settings["finetune_batch_size"]))
                probe_net = RiskNetV11(outer_train["context"].shape[-1], int(settings["width"]),
                                       float(settings["dropout"])).to(device)
                probe_net.load_state_dict(probe_state)
                probe_q = predict_anchors(probe_net, outer_val, device, int(settings["finetune_batch_size"]))
                probe_p20 = horizon_probability(probe_q[:, ANCHORS.index(20)], 40)
                probe_labels = outer_val["label"][:, ANCHORS.index(20)]
                probe_auc = float(roc_auc_score(probe_labels, probe_p20))
                prediction_frame[f"daily_hazard_{method}"] = q.reshape(-1)
                prediction_frame[f"probability_{method}"] = horizon_probability(q.reshape(-1), prediction_frame.horizon_days.to_numpy())
                save_checkpoint(run_dir / "models" / f"fold_{fold}_{method}.pt", net, outer_scaler, settings, method, steps, epochs)
                torch.save(encoder_state(outer_encoder), run_dir / "models" / f"fold_{fold}_{method}_pretrained_encoder.pt")
                diagnostics.append({"fold": fold, "method": method, "selection": inner_diag, "outer": outer_diag,
                                    "selected_steps": steps, "selected_epochs": epochs,
                                    "frozen_probe_20_to_40_auc": probe_auc,
                                    "wall_seconds": time.perf_counter() - started})
                print(f"V11 fold={fold} method={method} steps={steps} epochs={epochs} complete", flush=True)
            oof_parts.append(prediction_frame)
        write_json(run_dir / "pretrain_inner_splits.json", inner_splits)
        pd.DataFrame(pretrain_rows).to_csv(run_dir / "pretrain_history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(finetune_rows).to_csv(run_dir / "finetune_history.csv", index=False, encoding="utf-8-sig")
        write_json(run_dir / "representation_diagnostics.json", {"folds": diagnostics})
        oof = pd.concat(oof_parts).sort_values(["anchor_day", "gpsno"])
        if len(oof) != len(dev_ids) * len(ANCHORS) or oof.duplicated(["gpsno", "anchor_day"]).any(): raise ValueError("OOF incomplete")
        metrics = {"target": "right-censored stationary daily first-event hazard", "holdout_evaluated": False,
                   "by_method_and_anchor": {}, "selected_hyperparameters": selected_by_method, "parameter_count": counts}
        for method in methods:
            metrics["by_method_and_anchor"][method] = {}
            for anchor, part in oof.groupby("anchor_day"):
                key = f"{anchor}_to_{int(part.horizon_days.iloc[0])}"; score = score_binary(part.label, part[f"probability_{method}"])
                score["auc_95pct_bootstrap"] = auc_interval(part.label, part[f"probability_{method}"], config["seed"] + int(anchor))
                metrics["by_method_and_anchor"][method][key] = score
        part20 = oof[oof.anchor_day.eq(20)]; scratch = part20[["gpsno", "label", "probability_scratch"]].rename(columns={"probability_scratch": "probability"})
        metrics["paired_vs_scratch"] = {}; fold_deltas = {}
        for method in methods:
            current = part20[["gpsno", "label", f"probability_{method}"]].rename(columns={f"probability_{method}": "probability"})
            metrics["paired_vs_scratch"][method] = paired_auc(current, scratch, config["seed"] + METHODS.index(method))
            fold_deltas[method] = []
            for fold, group in part20.groupby("fold"):
                fold_deltas[method].append({"fold": int(fold), "delta_auc": float(roc_auc_score(group.label, group[f"probability_{method}"]) - roc_auc_score(group.label, group.probability_scratch))})
        metrics["fold_delta_auc_vs_scratch"] = fold_deltas
        best_method = max(methods, key=lambda m: metrics["by_method_and_anchor"][m]["20_to_40"]["roc_auc"])
        oof["daily_hazard"] = oof[f"daily_hazard_{best_method}"]; oof["probability"] = oof[f"probability_{best_method}"]
        write_parquet(run_dir / "predictions" / "oof_landmarks.parquet", oof)
        metrics["primary_method"] = best_method; metrics["development_oof"] = metrics["by_method_and_anchor"][best_method]["20_to_40"]
        for name, run_id in (("v10_chain_aux", config["v10_reference_run_id"]), ("v3", config["v3_reference_run_id"])):
            current = part20[["gpsno", "label", f"probability_{best_method}"]].rename(columns={f"probability_{best_method}": "probability"})
            metrics[f"paired_primary_vs_{name}"] = paired_auc(current, read_reference(results_root, run_id), config["seed"] + len(name))
        final_steps = int(np.median(selected_by_method[best_method]["steps"])); final_epochs = int(np.median(selected_by_method[best_method]["epochs"]))
        final_scaler = NodeDayScaler().fit(day_table, dev_ids); final_pack = vehicle_pack(day_table, outcomes, dev_ids, final_scaler)
        final_encoder, _, final_diag = pretrain(final_pack, settings, best_method, final_steps, config["seed"] + 90000, device)
        final_net, final_history, _, _ = fit_risk(final_pack, settings, encoder_state(final_encoder), config["seed"] + 91000,
                                                  device, epochs=final_epochs)
        save_checkpoint(run_dir / "models" / f"development_{best_method}.pt", final_net, final_scaler, settings, best_method, final_steps, final_epochs)
        ordered_days = day_table.sort_values(["gpsno", "day_index"]); events, context = final_scaler.transform(ordered_days)
        events = events.reshape(500, 60, events.shape[-2], events.shape[-1]); context = context.reshape(500, 60, context.shape[-1])
        q = predict_day60(final_net, events, context, device, int(settings["finetune_batch_size"])); full = horizon_probability(q, 40)
        matched = np.asarray([x in set(reference.gpsno) for x in profile_ids]); behavior = day_table.groupby("gpsno")[["trajectory_recorded_today", "imu_recorded_today"]].any().loc[profile_ids].any(axis=1).to_numpy()
        prior = float(outcomes[outcomes.anchor_day.eq(20)].label.mean()); probability = np.where(matched | behavior, full, prior)
        candidate = pd.DataFrame({"gpsno": profile_ids, "anchor_day": 60, "horizon_days": 40,
                                  "route": np.where(matched, "full", np.where(behavior, "context_only_no_matched_event", "prior_no_observed_behavior")),
                                  "probability": probability, "prediction_at_0_5": (probability >= .5).astype(int), "full_probability": full,
                                  "label_status": "future_unknown", "method": best_method})
        write_parquet(run_dir / "predictions" / "candidate_500.parquet", candidate)
        candidate[["gpsno", "probability", "prediction_at_0_5"]].to_csv(run_dir / "predictions" / "submission_candidate.csv", index=False, encoding="utf-8-sig")
        metrics["final_refit"] = {"method": best_method, "ssl_steps": final_steps, "finetune_epochs": final_epochs,
                                  "representation": final_diag, "candidate_routing": candidate.route.value_counts().to_dict()}
        metrics["warning"] = "Day60 future 40-day labels are unavailable; OOF uses observable earlier landmarks."
        write_json(run_dir / "metrics.json", metrics)
        lines = [f"# {manifest['run_id']}", "", "## 20→40 OOF", "", "| Method | AUC | Brier | Accuracy@0.5 | ΔAUC vs Scratch |", "|---|---:|---:|---:|---:|"]
        for method in methods:
            score = metrics["by_method_and_anchor"][method]["20_to_40"]; delta = metrics["paired_vs_scratch"][method]["delta_auc"]
            lines.append(f"| {method} | {score['roc_auc']:.6f} | {score['brier']:.6f} | {score['accuracy_at_0_5']:.6f} | {delta:+.6f} |")
        lines += ["", f"Primary method: **{best_method}**", "", "96 locked vehicles were not evaluated.", ""]
        (run_dir / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
        verify_source(REPO, manifest)
        manifest.update({"status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(), "primary_method": best_method,
                         "development_vehicles": 382, "holdout_vehicles": 96})
        write_json(run_dir / "manifest.json", manifest); update_leaderboard(results_root)
    except Exception:
        manifest.update({"status": "failed", "failed_utc": datetime.now(timezone.utc).isoformat(), "error": traceback.format_exc()})
        write_json(run_dir / "manifest.json", manifest); raise
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--database-root", type=Path, default=Path(r"E:\Databases\2026 清华IE亮剑-算法赛道赛题"))
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/v11_stage1.json")
    args = parser.parse_args(); print(run(args.config, args.database_root))
