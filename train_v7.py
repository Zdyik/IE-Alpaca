"""V7: one-layer daily Transformer versus a no-attention day model."""

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

from ie_alpaca.data.splits import audit_split, load_or_create_split  # noqa: E402
from ie_alpaca.data.task_one import labeled_reference, load_tables  # noqa: E402
from ie_alpaca.evaluation.metrics import auc_interval, score_binary  # noqa: E402
from ie_alpaca.features.daily_v7 import DayScaler, TOKEN_COLUMNS, build_day_table  # noqa: E402
from ie_alpaca.features.landmark_v3 import ANCHORS  # noqa: E402
from ie_alpaca.models.temporal_transformer_v7 import TemporalHazardV7  # noqa: E402
from ie_alpaca.tracking.run_store import (  # noqa: E402
    create_run, file_manifest, update_leaderboard, verify_source, write_json, write_parquet,
)
from ie_alpaca.training.event_v4 import resolve_device  # noqa: E402
from ie_alpaca.training.event_v5 import inner_vehicle_split  # noqa: E402
from ie_alpaca.training.landmark_v3 import horizon_probability, make_outcomes  # noqa: E402
from ie_alpaca.training.temporal_v7 import (  # noqa: E402
    fit_network, predict_anchors, predict_day60, select_epochs, vehicle_pack,
)


def paired_auc(current: pd.DataFrame, prior: pd.DataFrame, *, seed: int) -> dict:
    pair = current[["gpsno", "label", "probability"]].merge(
        prior[["gpsno", "label", "probability"]], on="gpsno",
        suffixes=("_new", "_prior"), validate="one_to_one",
    )
    if len(pair) != len(current) or not pair.label_new.eq(pair.label_prior).all():
        raise ValueError("Paired day-20 predictions have different vehicles or labels")
    y = pair.label_new.to_numpy(dtype=int)
    new = pair.probability_new.to_numpy(dtype=float)
    old = pair.probability_prior.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(1000):
        ix = rng.integers(0, len(pair), len(pair))
        if len(np.unique(y[ix])) == 2:
            deltas.append(roc_auc_score(y[ix], new[ix]) - roc_auc_score(y[ix], old[ix]))
    return {"vehicles": len(pair), "new_auc": float(roc_auc_score(y, new)),
            "prior_auc": float(roc_auc_score(y, old)),
            "delta_auc": float(roc_auc_score(y, new) - roc_auc_score(y, old)),
            "delta_auc_95pct_paired_bootstrap": [float(v) for v in np.quantile(deltas, [.025, .975])]}


def read_reference(results_root: Path, run_id: str) -> pd.DataFrame:
    path = results_root / "runs" / run_id / "predictions" / "oof_landmarks.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    with duckdb.connect() as con:
        return con.execute(
            "SELECT gpsno, label, probability FROM read_parquet(?) WHERE anchor_day=20", [str(path)],
        ).df()


def save_model(path: Path, net: TemporalHazardV7, scaler: DayScaler,
               model_config: dict, mode: str) -> None:
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "mode": mode, "model_config": model_config,
                "scaler": scaler.describe(), "token_columns": list(TOKEN_COLUMNS)}, path)


def run(config_path: Path, database_root: Path) -> Path:
    config_path, database_root = config_path.resolve(), database_root.resolve()
    if not config_path.is_relative_to(REPO):
        raise ValueError("Configuration must be inside the repository for source snapshotting")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if tuple(config["anchors"]) != ANCHORS or config["threshold"] != .5:
        raise ValueError("V7 anchors and threshold must match the frozen protocol")
    input_dir = database_root / "任务一预处理结果"
    results_root = database_root / "任务一实验结果"
    tables = load_tables(input_dir)
    reference = labeled_reference(tables.bags)
    results_root.mkdir(parents=True, exist_ok=True)
    split_path = results_root / "splits" / f"split_v1_seed{config['seed']}.json"
    split = load_or_create_split(reference, split_path, seed=config["seed"], folds=config["folds"],
                                 holdout_fraction=config["holdout_fraction"])
    audit = audit_split(reference, split)
    dev_ids = set(split["development"])
    profile_ids = sorted(tables.profile.gpsno.tolist())
    if len(reference) != 478 or len(dev_ids) != 382 or len(profile_ids) != 500:
        raise ValueError("Vehicle counts differ from the frozen V3-V6 protocol")
    day_table = build_day_table(tables.daily, profile_ids)
    landmarks = pd.MultiIndex.from_product(
        [sorted(dev_ids), ANCHORS], names=["gpsno", "anchor_day"],
    ).to_frame(index=False)
    outcomes = make_outcomes(tables.daily, landmarks)
    at20 = outcomes[outcomes.anchor_day.eq(20)].set_index("gpsno").label
    if not at20.eq(reference.set_index("gpsno").loc[at20.index, "label"]).all():
        raise ValueError("Day-20 outcome differs from the frozen reference")
    data_files = file_manifest(input_dir)
    fingerprint = hashlib.sha256(json.dumps(data_files, sort_keys=True).encode()).hexdigest()
    device = resolve_device(config["model"]["device"])
    if device.type == "cpu":
        torch.set_num_threads(int(config["model"]["cpu_threads"]))
    model_args = config["model"]
    parameter_count = {mode: sum(p.numel() for p in TemporalHazardV7(
        len(TOKEN_COLUMNS), mode=mode, d_model=model_args["d_model"],
        heads=model_args["heads"], ff_dim=model_args["ff_dim"],
        dropout=model_args["dropout"],
    ).parameters()) for mode in ("no_attention", "transformer")}
    resolved = {**config, "input_dir": str(input_dir), "results_root": str(results_root),
                "split_version": split["version"], "resolved_device": str(device),
                "token_columns": list(TOKEN_COLUMNS), "parameter_count": parameter_count}
    run_dir, manifest = create_run(REPO, results_root, config_path, resolved,
                                   version="V7", entrypoint="train_v7.py")
    try:
        manifest.update({"model": "torch_daily_transformer_v7", "split_version": split["version"],
                         "data_files": data_files, "data_fingerprint": fingerprint,
                         "evaluation_version": "v7_vehicle_oof_daily_sequence_stationary_hazard_0.5",
                         "parameter_count": parameter_count, "device": str(device),
                         "locked_labels_used_in_fit_or_metrics": False,
                         "dependencies": {name: importlib.metadata.version(name) for name in
                                          ("torch", "scikit-learn", "duckdb", "pandas", "numpy")}})
        write_json(run_dir / "manifest.json", manifest)
        shutil.copy2(split_path, run_dir / "splits.json")
        audit.update({"landmark_days": list(ANCHORS), "day_end_boundary": "23:59:59",
                      "label_start": "t+1", "scaler_fit": "inner/outer training vehicles, days 1..53",
                      "token_width": len(TOKEN_COLUMNS), "attention_padding_mask": True})
        write_json(run_dir / "leakage_audit.json", audit)
        labels20 = reference.set_index("gpsno").label.astype(int)
        oof_parts, selection_rows, curve_rows, history_rows = [], [], [], []
        inner_splits = {}
        for fold_text, val_list in sorted(split["validation_folds"].items(), key=lambda x: int(x[0])):
            fold = int(fold_text)
            val_ids = set(val_list)
            train_ids = dev_ids - val_ids
            if val_ids & train_ids or val_ids | train_ids != dev_ids:
                raise ValueError("Outer training/validation vehicle separation failed")
            inner_fit, inner_val = inner_vehicle_split(
                train_ids, labels20, seed=config["seed"] + 100 + fold,
                validation_fraction=config["selection"]["validation_fraction"],
            )
            inner_splits[fold_text] = {"fit": sorted(inner_fit), "validation": sorted(inner_val)}
            inner_scaler = DayScaler().fit(day_table, inner_fit)
            inner_fit_pack = vehicle_pack(day_table, outcomes, inner_fit, inner_scaler)
            inner_val_pack = vehicle_pack(day_table, outcomes, inner_val, inner_scaler)
            outer_scaler = DayScaler().fit(day_table, train_ids)
            train_pack = vehicle_pack(day_table, outcomes, train_ids, outer_scaler)
            val_pack = vehicle_pack(day_table, outcomes, val_ids, outer_scaler)
            val = outcomes[outcomes.gpsno.isin(val_ids)].sort_values(["gpsno", "anchor_day"]).copy()
            val["fold"] = fold
            for mode in ("no_attention", "transformer"):
                selected, curve = select_epochs(
                    inner_fit_pack, inner_val_pack, {**config["model"], **config["selection"]},
                    mode=mode, seed=config["seed"] + fold, device=device,
                )
                selection_rows.append({"fold": fold, "mode": mode, "selected_epoch": selected,
                                       "observed_epochs": len(curve), "inner_fit_vehicles": len(inner_fit),
                                       "inner_validation_vehicles": len(inner_val),
                                       "selected_inner_val_nll": curve[selected - 1]["inner_val_nll"]})
                curve_rows.extend({"fold": fold, "mode": mode, **entry} for entry in curve)
                net, history = fit_network(train_pack, config["model"], mode=mode, epochs=selected,
                                           seed=config["seed"] + fold, device=device)
                q = predict_anchors(net, val_pack, device, int(config["model"]["batch_size"]))
                val[f"daily_hazard_{mode}"] = q.reshape(-1)
                val[f"probability_{mode}"] = horizon_probability(q.reshape(-1),
                                                                 val.horizon_days.to_numpy())
                save_model(run_dir / "models" / f"fold_{fold}_{mode}.pt", net, outer_scaler,
                           {**config["model"], "epochs": selected}, mode)
                history_rows.extend({"fold": fold, "mode": mode, **entry} for entry in history)
            oof_parts.append(val)
            print(f"V7 fold {fold}: transformer epoch {selection_rows[-1]['selected_epoch']}, "
                  f"no-attention epoch {selection_rows[-2]['selected_epoch']}", flush=True)
        write_json(run_dir / "inner_vehicle_splits.json", inner_splits)
        pd.DataFrame(selection_rows).to_csv(run_dir / "selected_epochs.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(curve_rows).to_csv(run_dir / "inner_validation_curves.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(history_rows).to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8-sig")
        oof = pd.concat(oof_parts, ignore_index=True).sort_values(["anchor_day", "gpsno"])
        if len(oof) != len(dev_ids) * len(ANCHORS) or oof.duplicated(["gpsno", "anchor_day"]).any():
            raise ValueError("OOF lacks one prediction per development vehicle and anchor")
        oof["probability"] = oof.probability_transformer
        write_parquet(run_dir / "predictions" / "oof_landmarks.parquet", oof)
        metrics = {"target": "day-end landmark next-event stationary hazard; right censored at day 60",
                   "holdout_evaluated": False, "input_width": len(TOKEN_COLUMNS),
                   "parameter_count": parameter_count, "by_anchor": {}, "selected_epochs": selection_rows}
        for t, part in oof.groupby("anchor_day"):
            key = f"{t}_to_{int(part.horizon_days.iloc[0])}"
            metrics["by_anchor"][key] = {
                mode: score_binary(part.label, part[f"probability_{mode}"])
                for mode in ("no_attention", "transformer")
            }
            metrics["by_anchor"][key]["transformer_auc_95pct_bootstrap"] = auc_interval(
                part.label, part.probability_transformer, config["seed"] + int(t),
            )
        metrics["development_oof"] = metrics["by_anchor"]["20_to_40"]["transformer"]
        current = oof[oof.anchor_day.eq(20)]
        control = current[["gpsno", "label", "probability_no_attention"]].rename(
            columns={"probability_no_attention": "probability"})
        metrics["paired_transformer_vs_no_attention_20_to_40"] = paired_auc(
            current, control, seed=config["seed"] + 1,
        )
        metrics["paired_vs_v5_20_to_40"] = paired_auc(
            current, read_reference(results_root, config["parent_run_id"]), seed=config["seed"],
        )
        metrics["paired_vs_v3_20_to_40"] = paired_auc(
            current, read_reference(results_root, config["v3_reference_run_id"]), seed=config["seed"],
        )
        final_epoch = int(np.median([row["selected_epoch"] for row in selection_rows
                                     if row["mode"] == "transformer"]))
        final_scaler = DayScaler().fit(day_table, dev_ids)
        final_pack = vehicle_pack(day_table, outcomes, dev_ids, final_scaler)
        final_net, final_history = fit_network(final_pack, config["model"], mode="transformer",
                                               epochs=final_epoch, seed=config["seed"] + 999, device=device)
        save_model(run_dir / "models" / "development_transformer.pt", final_net, final_scaler,
                   {**config["model"], "epochs": final_epoch}, "transformer")
        sorted_days = day_table.sort_values(["gpsno", "day_index"])
        x = final_scaler.transform(sorted_days).reshape(len(profile_ids), 60, len(TOKEN_COLUMNS))
        q = predict_day60(final_net, x, device, int(config["model"]["batch_size"]))
        full_probability = horizon_probability(q, 40)
        matched = np.asarray([gpsno in set(reference.gpsno) for gpsno in profile_ids])
        behavior = day_table.groupby("gpsno")[["trajectory_recorded_today", "imu_recorded_today"]].any()
        has_behavior = behavior.loc[profile_ids].any(axis=1).to_numpy(dtype=bool)
        prior = float(at20.loc[sorted(dev_ids)].mean())
        probability = np.where(matched | has_behavior, full_probability, prior)
        route = np.where(matched, "full", np.where(has_behavior, "context_only_no_matched_event",
                                                   "prior_no_observed_behavior"))
        candidate = pd.DataFrame({"gpsno": profile_ids, "anchor_day": 60, "horizon_days": 40,
                                  "route": route, "probability": probability,
                                  "prediction_at_0_5": (probability >= .5).astype(int),
                                  "full_probability": full_probability,
                                  "label_status": "future_unknown"})
        if (len(candidate) != 500 or candidate.gpsno.duplicated().any() or
                not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))):
            raise ValueError("500-car candidate probabilities are invalid")
        write_parquet(run_dir / "predictions" / "candidate_500.parquet", candidate)
        candidate[["gpsno", "probability", "prediction_at_0_5"]].to_csv(
            run_dir / "predictions" / "submission_candidate.csv", index=False, encoding="utf-8-sig",
        )
        metrics["candidate_routing"] = {"full": int(matched.sum()),
                                        "context_only_no_matched_event": int((~matched & has_behavior).sum()),
                                        "prior_no_observed_behavior": int((~matched & ~has_behavior).sum()),
                                        "development_20_to_40_prior": prior}
        metrics["final_training_objective"] = final_history[-1]["train_objective"]
        metrics["final_selected_epoch"] = final_epoch
        metrics["warning"] = "Day60 future 40 days are not observed locally; OOF is an earlier-anchor proxy."
        write_json(run_dir / "metrics.json", metrics)
        score = metrics["development_oof"]
        (run_dir / "result_summary.md").write_text(
            f"# {manifest['run_id']}\n\n"
            f"- 20→40 vehicle OOF AUC: {score['roc_auc']:.6f}; "
            f"Brier: {score['brier']:.6f}; Accuracy@0.5: {score['accuracy_at_0_5']:.6f}\n"
            f"- No-attention control AUC: {metrics['by_anchor']['20_to_40']['no_attention']['roc_auc']:.6f}\n"
            f"- Transformer parameters: {parameter_count['transformer']}; "
            f"control parameters: {parameter_count['no_attention']}\n"
            f"- Selected transformer epochs: {[row['selected_epoch'] for row in selection_rows if row['mode'] == 'transformer']}; "
            f"final refit: {final_epoch}\n"
            "- 96 locked vehicles were not evaluated; Day60→future40 has no local labels.\n",
            encoding="utf-8",
        )
        verify_source(REPO, manifest)
        manifest.update({"status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
                         "development_vehicles": len(dev_ids), "holdout_vehicles": len(split["holdout"])})
        write_json(run_dir / "manifest.json", manifest)
        update_leaderboard(results_root)
    except Exception:
        manifest.update({"status": "failed", "failed_utc": datetime.now(timezone.utc).isoformat(),
                         "error": traceback.format_exc()})
        write_json(run_dir / "manifest.json", manifest)
        raise
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-root", type=Path,
                        default=Path(r"E:\Databases\2026 清华IE亮剑-算法赛道赛题"))
    parser.add_argument("--config", type=Path,
                        default=REPO / "configs/experiments/v7_temporal_transformer.json")
    args = parser.parse_args()
    print(run(args.config, args.database_root))


if __name__ == "__main__":
    main()
