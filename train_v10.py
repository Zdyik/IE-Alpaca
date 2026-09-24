"""Train V10 RiskChainNet variants with frozen vehicle folds and right censoring."""

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
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from ie_alpaca.data.splits import audit_split, load_or_create_split  # noqa: E402
from ie_alpaca.data.task_one import labeled_reference, load_tables  # noqa: E402
from ie_alpaca.evaluation.metrics import auc_interval, score_binary  # noqa: E402
from ie_alpaca.features.daily_v10 import (  # noqa: E402
    CONTEXT_COLUMNS, EVENT_FEATURE_NAMES, NodeDayScaler, build_day_table,
)
from ie_alpaca.features.landmark_v3 import ANCHORS  # noqa: E402
from ie_alpaca.features.landmark_v4 import EVENT_CODES, EVENT_NAMES  # noqa: E402
from ie_alpaca.models.risk_chain_v10 import RELATIONS, RiskChainNetV10  # noqa: E402
from ie_alpaca.tracking.run_store import (  # noqa: E402
    create_run, file_manifest, update_leaderboard, verify_source, write_json, write_parquet,
)
from ie_alpaca.training.event_v4 import resolve_device  # noqa: E402
from ie_alpaca.training.landmark_v3 import horizon_probability, make_outcomes  # noqa: E402
from ie_alpaca.training.risk_chain_v10 import (  # noqa: E402
    fit_network, predict_anchors, predict_day60, select_epochs, vehicle_pack,
)


def paired_auc(current: pd.DataFrame, prior: pd.DataFrame, *, seed: int) -> dict:
    pair = current[["gpsno", "label", "probability"]].merge(
        prior[["gpsno", "label", "probability"]], on="gpsno",
        suffixes=("_new", "_prior"), validate="one_to_one")
    if len(pair) != len(current) or not pair.label_new.eq(pair.label_prior).all():
        raise ValueError("paired predictions have different vehicles or labels")
    y = pair.label_new.to_numpy(int); new = pair.probability_new.to_numpy(float)
    old = pair.probability_prior.to_numpy(float); rng = np.random.default_rng(seed); values = []
    for _ in range(2000):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) == 2:
            values.append(roc_auc_score(y[index], new[index]) - roc_auc_score(y[index], old[index]))
    return {"vehicles": len(pair), "new_auc": float(roc_auc_score(y, new)),
            "prior_auc": float(roc_auc_score(y, old)),
            "delta_auc": float(roc_auc_score(y, new) - roc_auc_score(y, old)),
            "delta_auc_95pct_paired_bootstrap": [float(x) for x in np.quantile(values, [.025, .975])],
            "valid_bootstraps": len(values)}


def read_reference(results_root: Path, run_id: str, probability_column: str = "probability") -> pd.DataFrame:
    path = results_root / "runs" / run_id / "predictions" / "oof_landmarks.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    with duckdb.connect() as con:
        return con.execute(
            f'SELECT gpsno, label, "{probability_column}" AS probability FROM read_parquet(?) WHERE anchor_day=20',
            [str(path)]).df()


def save_model(path: Path, net: RiskChainNetV10, scaler: NodeDayScaler,
               model_config: dict, mode: str) -> None:
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                "mode": mode, "model_config": model_config, "scaler": scaler.describe(),
                "relations": net.relations}, path)


def edge_rows(net: RiskChainNetV10, fold: int | str, mode: str) -> list[dict]:
    strength = torch.sigmoid(net.edge_strength).detach().cpu().numpy()
    lag = torch.softmax(net.lag_logits, dim=-1).detach().cpu().numpy()
    rows = []
    for layer in range(2):
        for relation, (source, target) in enumerate(RELATIONS):
            for day in range(1, 8):
                rows.append({"fold": fold, "variant": mode, "layer": layer + 1,
                             "relation": relation, "source_codes": "/".join(map(str, source)),
                             "target_codes": "/".join(map(str, target)),
                             "edge_strength": float(strength[layer, relation]),
                             "lag_day": day, "lag_weight": float(lag[layer, relation, day - 1])})
    return rows


def run(config_path: Path, database_root: Path) -> Path:
    config_path, database_root = config_path.resolve(), database_root.resolve()
    if not config_path.is_relative_to(REPO):
        raise ValueError("V10 config must be inside the repository")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    variants, primary = config["variants"], config["primary_variant"]
    if tuple(config["anchors"]) != ANCHORS or primary not in variants:
        raise ValueError("V10 anchors or primary variant differ from the frozen protocol")
    if not set(variants).issubset(RiskChainNetV10.MODES):
        raise ValueError("V10 config contains an unsupported variant")
    input_dir, results_root = database_root / "任务一预处理结果", database_root / "任务一实验结果"
    tables = load_tables(input_dir); reference = labeled_reference(tables.bags)
    split_path = results_root / "splits" / f"split_v1_seed{config['seed']}.json"
    split = load_or_create_split(reference, split_path, seed=config["seed"], folds=config["folds"],
                                 holdout_fraction=config["holdout_fraction"])
    audit, dev_ids = audit_split(reference, split), set(split["development"])
    profile_ids = sorted(tables.profile.gpsno.astype(str).tolist())
    if len(reference) != 478 or len(dev_ids) != 382 or len(profile_ids) != 500:
        raise ValueError("vehicle counts differ from the frozen protocol")
    day_table = build_day_table(tables.daily, profile_ids)
    landmarks = pd.MultiIndex.from_product([sorted(dev_ids), ANCHORS],
                                           names=["gpsno", "anchor_day"]).to_frame(index=False)
    outcomes = make_outcomes(tables.daily, landmarks)
    labels20 = reference.set_index("gpsno").label.astype(int)
    at20 = outcomes[outcomes.anchor_day.eq(20)].set_index("gpsno").label
    if not at20.eq(labels20.loc[at20.index]).all():
        raise ValueError("V10 day-20 labels differ from the frozen reference")
    device = resolve_device(config["model"]["device"])
    if device.type == "cpu":
        torch.set_num_threads(int(config["model"]["cpu_threads"]))
    model_args = config["model"]
    parameter_count = {mode: sum(parameter.numel() for parameter in RiskChainNetV10(
        context_width=len(CONTEXT_COLUMNS), mode=mode, width=model_args["width"],
        event_embedding=model_args["event_embedding"], role_embedding=model_args["role_embedding"],
        stage_embedding=model_args["stage_embedding"], dropout=model_args["dropout"],
        modality_dropout=model_args["modality_dropout"]).parameters()) for mode in variants}
    data_files = file_manifest(input_dir)
    fingerprint = hashlib.sha256(json.dumps(data_files, sort_keys=True).encode()).hexdigest()
    resolved = {**config, "input_dir": str(input_dir), "results_root": str(results_root),
                "split_version": split["version"], "resolved_device": str(device),
                "event_codes": list(EVENT_CODES), "event_feature_names": list(EVENT_FEATURE_NAMES),
                "context_columns": list(CONTEXT_COLUMNS), "parameter_count": parameter_count}
    default_prefix = "V10S1" if set(variants) == {"flat", "role", "chain", "chain_shuffle"} else "V10S2"
    prefix = str(config.get("run_prefix", default_prefix))
    run_dir, manifest = create_run(REPO, results_root, config_path, resolved,
                                   version=prefix, entrypoint="train_v10.py")
    try:
        manifest.update({"model": "torch_v10_risk_chain_tcn", "split_version": split["version"],
                         "data_files": data_files, "data_fingerprint": fingerprint,
                         "evaluation_version": "v10_vehicle_oof_daily_node_chain_survival_0.5",
                         "parameter_count": parameter_count, "device": str(device),
                         "locked_labels_used_in_fit_or_metrics": False,
                         "dependencies": {name: importlib.metadata.version(name) for name in
                                          ("torch", "scikit-learn", "duckdb", "pandas", "numpy")}})
        write_json(run_dir / "manifest.json", manifest); shutil.copy2(split_path, run_dir / "splits.json")
        audit.update({"landmark_days": list(ANCHORS), "label_start": "t+1",
                      "scaler_fit": "inner/outer training vehicles days 1..53",
                      "future_days_visible_to_anchor": False, "event_node_count": len(EVENT_CODES),
                      "context_node_count": 3, "locked_holdout_scored": False,
                      "variants": variants})
        write_json(run_dir / "leakage_audit.json", audit)
        settings = config["model"]
        epoch_policy = str(settings.get("epoch_policy", "fixed_equal_budget"))
        if epoch_policy not in {"fixed_equal_budget", "inner_validation"}:
            raise ValueError(f"unsupported V10 epoch policy: {epoch_policy}")
        fixed_epochs = int(settings["epochs"])
        if fixed_epochs < 1:
            raise ValueError("V10 fixed epochs must be positive")
        oof_parts, training_rows, gate_rows, graph_rows = [], [], [], []
        for fold_text, val_list in sorted(split["validation_folds"].items(), key=lambda item: int(item[0])):
            fold = int(fold_text); val_ids = set(val_list); train_ids = dev_ids - val_ids
            if val_ids & train_ids or val_ids | train_ids != dev_ids:
                raise ValueError("outer fold vehicle isolation failed")
            scaler = NodeDayScaler().fit(day_table, train_ids)
            train_pack = vehicle_pack(day_table, outcomes, train_ids, scaler)
            val_pack = vehicle_pack(day_table, outcomes, val_ids, scaler)
            val = outcomes[outcomes.gpsno.isin(val_ids)].sort_values(["gpsno", "anchor_day"]).copy()
            val["fold"] = fold
            for mode in variants:
                selected_epochs = fixed_epochs
                if epoch_policy == "inner_validation":
                    ordered_train = np.asarray(sorted(train_ids))
                    inner_fit, inner_val = train_test_split(
                        ordered_train, test_size=float(settings["inner_validation_fraction"]),
                        random_state=config["seed"] + fold, stratify=labels20.loc[ordered_train])
                    inner_fit_ids, inner_val_ids = set(inner_fit), set(inner_val)
                    inner_scaler = NodeDayScaler().fit(day_table, inner_fit_ids)
                    inner_fit_pack = vehicle_pack(day_table, outcomes, inner_fit_ids, inner_scaler)
                    inner_val_pack = vehicle_pack(day_table, outcomes, inner_val_ids, inner_scaler)
                    selected_epochs, selection_history = select_epochs(
                        inner_fit_pack, inner_val_pack, settings, mode=mode,
                        seed=config["seed"] + fold, device=device)
                    training_rows.extend({"fold": fold, "variant": mode,
                                          "phase": "inner_epoch_selection",
                                          "selected_epochs": selected_epochs, **row}
                                         for row in selection_history)
                net, history = fit_network(train_pack, settings, mode=mode, epochs=selected_epochs,
                                           seed=config["seed"] + fold, device=device)
                q, gates = predict_anchors(net, val_pack, device, int(settings["batch_size"]))
                val[f"daily_hazard_{mode}"] = q.reshape(-1)
                val[f"probability_{mode}"] = horizon_probability(q.reshape(-1), val.horizon_days.to_numpy())
                training_rows.extend({"fold": fold, "variant": mode, "phase": "outer_fit",
                                      "selected_epochs": selected_epochs, **row} for row in history)
                for index, code in enumerate(EVENT_CODES):
                    gate_rows.append({"fold": fold, "variant": mode, "event_code": code,
                                      "event_name": EVENT_NAMES[code], "mean_gate": float(gates[:, index].mean())})
                if mode in {"chain", "chain_shuffle", "chain_aux", "quality_chain"}:
                    graph_rows.extend(edge_rows(net, fold, mode))
                save_model(run_dir / "models" / f"fold_{fold}_{mode}.pt", net, scaler,
                           settings, mode)
            oof_parts.append(val); print(f"V10 fold {fold} complete", flush=True)
        pd.DataFrame(training_rows).to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(gate_rows).to_csv(run_dir / "event_gates_by_fold.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(graph_rows).to_csv(run_dir / "graph_edges_by_fold.csv", index=False, encoding="utf-8-sig")
        oof = pd.concat(oof_parts, ignore_index=True).sort_values(["anchor_day", "gpsno"])
        if len(oof) != len(dev_ids) * len(ANCHORS) or oof.duplicated(["gpsno", "anchor_day"]).any():
            raise ValueError("V10 OOF coverage is incomplete")
        oof["daily_hazard"] = oof[f"daily_hazard_{primary}"]
        oof["probability"] = oof[f"probability_{primary}"]
        write_parquet(run_dir / "predictions" / "oof_landmarks.parquet", oof)
        metrics = {"target": "day-end next-event stationary hazard with daily structured nodes",
                   "primary_variant": primary, "parameter_count": parameter_count,
                   "holdout_evaluated": False, "by_variant_and_anchor": {},
                   "training_protocol": {"epoch_policy": epoch_policy, "epochs": fixed_epochs,
                                         "batch_size": int(settings["batch_size"])} }
        if epoch_policy == "inner_validation":
            selected = pd.DataFrame(training_rows)
            selected = selected[selected.phase.eq("outer_fit")].groupby(
                ["fold", "variant"], as_index=False).selected_epochs.first()
            metrics["training_protocol"]["selected_epochs_by_fold"] = selected.to_dict("records")
        for mode in variants:
            metrics["by_variant_and_anchor"][mode] = {}
            for anchor, part in oof.groupby("anchor_day"):
                key = f"{anchor}_to_{int(part.horizon_days.iloc[0])}"
                score = score_binary(part.label, part[f"probability_{mode}"])
                score["auc_95pct_bootstrap"] = auc_interval(
                    part.label, part[f"probability_{mode}"], config["seed"] + int(anchor))
                metrics["by_variant_and_anchor"][mode][key] = score
        metrics["development_oof"] = metrics["by_variant_and_anchor"][primary]["20_to_40"]
        part20 = oof[oof.anchor_day.eq(20)]
        comparisons = {}
        for index, (left, right) in enumerate(zip(variants[1:], variants[:-1])):
            current = part20[["gpsno", "label", f"probability_{left}"]].rename(
                columns={f"probability_{left}": "probability"})
            prior = part20[["gpsno", "label", f"probability_{right}"]].rename(
                columns={f"probability_{right}": "probability"})
            comparisons[f"{left}_minus_{right}"] = paired_auc(current, prior, seed=config["seed"] + index)
        if "chain" in variants and "chain_shuffle" in variants:
            current = part20[["gpsno", "label", "probability_chain"]].rename(columns={"probability_chain": "probability"})
            prior = part20[["gpsno", "label", "probability_chain_shuffle"]].rename(
                columns={"probability_chain_shuffle": "probability"})
            comparisons["chain_minus_chain_shuffle"] = paired_auc(current, prior, seed=config["seed"] + 30)
        metrics["paired_comparisons_at_20_to_40"] = comparisons
        current_primary = part20[["gpsno", "label", f"probability_{primary}"]].rename(
            columns={f"probability_{primary}": "probability"})
        metrics["paired_primary_vs_v7_no_attention"] = paired_auc(
            current_primary, read_reference(results_root, config["parent_run_id"], "probability_no_attention"),
            seed=config["seed"] + 40)
        metrics["paired_primary_vs_v3"] = paired_auc(
            current_primary, read_reference(results_root, config["v3_reference_run_id"]), seed=config["seed"] + 41)
        if {"role", "chain", "chain_shuffle"}.issubset(variants):
            delta_role = comparisons["chain_minus_role"]["delta_auc"]
            delta_shuffle = comparisons["chain_minus_chain_shuffle"]["delta_auc"]
            metrics["stage2_decision"] = {
                "eligible": bool(delta_role > 0 and delta_shuffle > 0),
                "rule": "run stage2 only when chain is directionally better than both role and shuffled graph",
                "chain_minus_role": delta_role, "chain_minus_shuffle": delta_shuffle,
            }

        final_scaler = NodeDayScaler().fit(day_table, dev_ids)
        final_pack = vehicle_pack(day_table, outcomes, dev_ids, final_scaler)
        sorted_days = day_table.sort_values(["gpsno", "day_index"])
        final_events, final_context = final_scaler.transform(sorted_days)
        final_events = final_events.reshape(len(profile_ids), 60, len(EVENT_CODES), len(EVENT_FEATURE_NAMES))
        final_context = final_context.reshape(len(profile_ids), 60, len(CONTEXT_COLUMNS))
        matched = np.asarray([gpsno in set(reference.gpsno) for gpsno in profile_ids])
        behavior = day_table.groupby("gpsno")[["trajectory_recorded_today", "imu_recorded_today"]].any()
        has_behavior = behavior.loc[profile_ids].any(axis=1).to_numpy(bool); prior_probability = float(at20.loc[sorted(dev_ids)].mean())
        final_epochs = fixed_epochs
        if epoch_policy == "inner_validation":
            chosen = [row["selected_epochs"] for row in metrics["training_protocol"]["selected_epochs_by_fold"]
                      if row["variant"] == primary]
            final_epochs = int(np.median(chosen))
            metrics["training_protocol"]["final_epochs"] = final_epochs
        net, _ = fit_network(final_pack, settings, mode=primary, epochs=final_epochs,
                             seed=config["seed"] + 999, device=device)
        save_model(run_dir / "models" / f"development_{primary}.pt", net, final_scaler, settings, primary)
        if primary in {"chain", "chain_shuffle", "chain_aux", "quality_chain"}:
            graph_rows.extend(edge_rows(net, "development", primary))
        q, gates = predict_day60(net, final_events, final_context, device, int(settings["batch_size"]))
        full = horizon_probability(q, 40); probability = np.where(matched | has_behavior, full, prior_probability)
        candidate = pd.DataFrame({"gpsno": profile_ids, "anchor_day": 60, "horizon_days": 40,
            "route": np.where(matched, "full", np.where(has_behavior, "context_only_no_matched_event",
                                                          "prior_no_observed_behavior")),
            "probability": probability, "prediction_at_0_5": (probability >= .5).astype(int),
            "full_probability": full, "label_status": "future_unknown"})
        write_parquet(run_dir / "predictions" / f"candidate_500_{primary}.parquet", candidate)
        pd.DataFrame({"event_code": EVENT_CODES, "event_name": [EVENT_NAMES[c] for c in EVENT_CODES],
                      "mean_day60_gate": gates.mean(axis=0)}).to_csv(
            run_dir / "models" / f"development_{primary}_event_gates.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(graph_rows).to_csv(run_dir / "graph_edges_by_fold.csv", index=False, encoding="utf-8-sig")
        write_parquet(run_dir / "predictions" / "candidate_500.parquet", candidate)
        candidate[["gpsno", "probability", "prediction_at_0_5"]].to_csv(
            run_dir / "predictions" / "submission_candidate.csv", index=False, encoding="utf-8-sig")
        metrics["candidate_routing"] = {"full": int(matched.sum()),
            "context_only_no_matched_event": int((~matched & has_behavior).sum()),
            "prior_no_observed_behavior": int((~matched & ~has_behavior).sum()),
            "development_20_to_40_prior": prior_probability}
        metrics["warning"] = "Day60 future 40 days are unobserved; V10 OOF is an earlier-anchor proxy."
        write_json(run_dir / "metrics.json", metrics)
        lines = [f"# {manifest['run_id']}", "", "## 20→40 OOF", "",
                 "| Variant | Parameters | AUC | Brier | Accuracy@0.5 |", "|---|---:|---:|---:|---:|"]
        for mode in variants:
            score = metrics["by_variant_and_anchor"][mode]["20_to_40"]
            lines.append(f"| {mode} | {parameter_count[mode]} | {score['roc_auc']:.6f} | {score['brier']:.6f} | {score['accuracy_at_0_5']:.6f} |")
        if "stage2_decision" in metrics:
            lines += ["", f"Stage2 eligible: {metrics['stage2_decision']['eligible']}"]
        lines += ["", "96 locked vehicles were not evaluated.", ""]
        (run_dir / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
        verify_source(REPO, manifest)
        manifest.update({"status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
                         "development_vehicles": len(dev_ids), "holdout_vehicles": len(split["holdout"])})
        write_json(run_dir / "manifest.json", manifest); update_leaderboard(results_root)
    except Exception:
        manifest.update({"status": "failed", "failed_utc": datetime.now(timezone.utc).isoformat(),
                         "error": traceback.format_exc()})
        write_json(run_dir / "manifest.json", manifest); raise
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-root", type=Path,
                        default=Path(r"E:\Databases\2026 清华IE亮剑-算法赛道赛题"))
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/v10_stage1.json")
    args = parser.parse_args(); print(run(args.config, args.database_root))


if __name__ == "__main__":
    main()
