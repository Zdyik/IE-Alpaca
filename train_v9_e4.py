"""Run V9-E4 direct, auxiliary-exposure and factorized-exposure hazard tests."""

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
from ie_alpaca.features.landmark_v3 import ANCHORS, build_landmarks, feature_columns  # noqa: E402
from ie_alpaca.models.exposure_v9 import ExposureRiskNetV9  # noqa: E402
from ie_alpaca.tracking.run_store import (  # noqa: E402
    create_run, file_manifest, update_leaderboard, verify_source, write_json, write_parquet,
)
from ie_alpaca.training.event_v4 import resolve_device  # noqa: E402
from ie_alpaca.training.event_v5 import inner_vehicle_split  # noqa: E402
from ie_alpaca.training.exposure_v9 import (  # noqa: E402
    LandmarkScalerV9, add_future_exposure, fit_exposure, predict_exposure,
    select_exposure_epochs, vehicle_pack,
)
from ie_alpaca.training.landmark_v3 import horizon_probability, make_outcomes  # noqa: E402


def _paired(y: np.ndarray, new: np.ndarray, old: np.ndarray, seed: int) -> dict:
    rng, delta = np.random.default_rng(seed), []
    for _ in range(2000):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) == 2:
            delta.append(roc_auc_score(y[index], new[index]) - roc_auc_score(y[index], old[index]))
    return {"delta_auc": float(roc_auc_score(y, new) - roc_auc_score(y, old)),
            "delta_auc_95pct_paired_bootstrap": [float(x) for x in np.quantile(delta, [.025, .975])],
            "valid_bootstraps": len(delta)}


def _save(path: Path, net: ExposureRiskNetV9, scaler: LandmarkScalerV9,
          config: dict, variant: str) -> None:
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                "variant": variant, "model_config": config, "scaler": scaler.describe()}, path)


def run(config_path: Path, database_root: Path) -> Path:
    config_path, database_root = config_path.resolve(), database_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    variants, primary = config["variants"], config["primary_variant"]
    if tuple(config["anchors"]) != ANCHORS or primary not in variants:
        raise ValueError("V9-E4 配置与冻结协议不一致")
    input_dir, results_root = database_root / "任务一预处理结果", database_root / "任务一实验结果"
    tables = load_tables(input_dir)
    reference = labeled_reference(tables.bags)
    split_path = results_root / "splits" / f"split_v1_seed{config['seed']}.json"
    split = load_or_create_split(reference, split_path, seed=config["seed"], folds=config["folds"],
                                 holdout_fraction=config["holdout_fraction"])
    audit, dev_ids = audit_split(reference, split), set(split["development"])
    profile_ids = sorted(tables.profile.gpsno.astype(str).tolist())
    features = build_landmarks(tables.daily, profile_ids, anchors=(*ANCHORS, 60))
    columns, _ = feature_columns(features)
    development = features[features.gpsno.isin(dev_ids) & features.anchor_day.ne(60)].copy()
    outcomes = add_future_exposure(tables.daily, make_outcomes(tables.daily, development))
    labels20 = reference.set_index("gpsno").label.astype(int)
    check = outcomes[outcomes.anchor_day.eq(20)].set_index("gpsno").label
    if len(reference) != 478 or len(dev_ids) != 382 or len(profile_ids) != 500 or not check.eq(labels20.loc[check.index]).all():
        raise ValueError("车辆数量或冻结标签不一致")
    device = resolve_device(config["model"]["device"])
    example = ExposureRiskNetV9(2 * len(columns), variant=primary, dropout=config["model"]["dropout"],
                                base_daily_hazard=.01, base_log_exposure=2.0)
    parameter_count = sum(value.numel() for value in example.parameters())
    data_files = file_manifest(input_dir)
    fingerprint = hashlib.sha256(json.dumps(data_files, sort_keys=True).encode()).hexdigest()
    resolved = {**config, "input_dir": str(input_dir), "results_root": str(results_root),
                "split_version": split["version"], "resolved_device": str(device),
                "feature_columns": columns, "parameter_count": parameter_count,
                "future_exposure_target": "mean distance_km over observable post-anchor days"}
    run_dir, manifest = create_run(REPO, results_root, config_path, resolved,
                                   version="V9E4", entrypoint="train_v9_e4.py")
    try:
        manifest.update({"model": "torch_v9_exposure_factorization_ablation", "split_version": split["version"],
                         "data_files": data_files, "data_fingerprint": fingerprint,
                         "evaluation_version": "v9_vehicle_oof_stationary_hazard_exposure_0.5",
                         "parameter_count": parameter_count, "device": str(device),
                         "locked_labels_used_in_fit_or_metrics": False,
                         "dependencies": {name: importlib.metadata.version(name) for name in
                                          ("torch", "scikit-learn", "duckdb", "pandas", "numpy")}})
        write_json(run_dir / "manifest.json", manifest); shutil.copy2(split_path, run_dir / "splits.json")
        audit.update({"landmark_days": list(ANCHORS), "label_start": "t+1",
                      "future_exposure_used_as_input": False,
                      "future_exposure_used_as_training_target_only": True,
                      "locked_holdout_scored": False})
        write_json(run_dir / "leakage_audit.json", audit)
        settings = {**config["model"], **config["selection"]}
        oof_parts, selections, curves, histories, inner_splits = [], [], [], [], {}
        for fold_text, val_list in sorted(split["validation_folds"].items(), key=lambda item: int(item[0])):
            fold = int(fold_text); val_ids = set(val_list); train_ids = dev_ids - val_ids
            inner_fit, inner_val = inner_vehicle_split(
                train_ids, labels20, seed=config["seed"] + 100 + fold,
                validation_fraction=settings["validation_fraction"])
            inner_splits[fold_text] = {"fit": sorted(inner_fit), "validation": sorted(inner_val)}
            inner_scaler = LandmarkScalerV9(columns).fit(development[development.gpsno.isin(inner_fit)])
            inner_train = vehicle_pack(development, outcomes, inner_fit, inner_scaler)
            inner_validation = vehicle_pack(development, outcomes, inner_val, inner_scaler)
            scaler = LandmarkScalerV9(columns).fit(development[development.gpsno.isin(train_ids)])
            train = vehicle_pack(development, outcomes, train_ids, scaler)
            val_features = development[development.gpsno.isin(val_ids)].sort_values(["gpsno", "anchor_day"]).copy()
            val = val_features[["gpsno", "anchor_day"]].merge(
                outcomes, on=["gpsno", "anchor_day"], validate="one_to_one").sort_values(["gpsno", "anchor_day"])
            val["fold"] = fold
            for variant in variants:
                selected, curve = select_exposure_epochs(inner_train, inner_validation, settings,
                                                          variant=variant, seed=config["seed"] + fold,
                                                          device=device)
                selections.append({"fold": fold, "variant": variant, "selected_epoch": selected,
                                   "observed_epochs": len(curve),
                                   "selected_inner_val_survival_nll": curve[selected - 1]["inner_val_survival_nll"]})
                curves.extend({"fold": fold, "variant": variant, **row} for row in curve)
                net, history = fit_exposure(train, settings, variant=variant, epochs=selected,
                                            seed=config["seed"] + fold, device=device)
                q, predicted_exposure = predict_exposure(net, scaler, val_features, device)
                val[f"daily_hazard_{variant}"] = q
                val[f"predicted_future_km_per_day_{variant}"] = predicted_exposure
                val[f"probability_{variant}"] = horizon_probability(q, val.horizon_days.to_numpy())
                histories.extend({"fold": fold, "variant": variant, **row} for row in history)
                _save(run_dir / "models" / f"fold_{fold}_{variant}.pt", net, scaler, settings, variant)
            oof_parts.append(val); print(f"V9-E4 fold {fold} complete", flush=True)
        write_json(run_dir / "inner_vehicle_splits.json", inner_splits)
        pd.DataFrame(selections).to_csv(run_dir / "selected_epochs.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(curves).to_csv(run_dir / "inner_validation_curves.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(histories).to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8-sig")
        oof = pd.concat(oof_parts, ignore_index=True).sort_values(["anchor_day", "gpsno"])
        oof["daily_hazard"], oof["probability"] = oof[f"daily_hazard_{primary}"], oof[f"probability_{primary}"]
        write_parquet(run_dir / "predictions" / "oof_landmarks.parquet", oof)
        metrics = {"target": "day-end next-event hazard with future-km auxiliary target",
                   "primary_variant": primary, "parameter_count": parameter_count,
                   "holdout_evaluated": False, "by_variant_and_anchor": {}, "selected_epochs": selections}
        for variant in variants:
            metrics["by_variant_and_anchor"][variant] = {}
            for anchor, part in oof.groupby("anchor_day"):
                key = f"{anchor}_to_{int(part.horizon_days.iloc[0])}"
                score = score_binary(part.label, part[f"probability_{variant}"])
                score["auc_95pct_bootstrap"] = auc_interval(
                    part.label, part[f"probability_{variant}"], config["seed"] + int(anchor))
                error = part[f"predicted_future_km_per_day_{variant}"] - part.future_km_per_day
                score["future_km_mae"] = float(np.abs(error).mean())
                metrics["by_variant_and_anchor"][variant][key] = score
        metrics["development_oof"] = metrics["by_variant_and_anchor"][primary]["20_to_40"]
        part20 = oof[oof.anchor_day.eq(20)]; y = part20.label.to_numpy(dtype=int)
        metrics["paired_comparisons_at_20_to_40"] = {}
        for i, (left, right) in enumerate(zip(variants[1:], variants[:-1])):
            metrics["paired_comparisons_at_20_to_40"][f"{left}_minus_{right}"] = _paired(
                y, part20[f"probability_{left}"].to_numpy(),
                part20[f"probability_{right}"].to_numpy(), config["seed"] + i)
        v3_path = results_root / "runs" / config["v3_reference_run_id"] / "predictions" / "oof_landmarks.parquet"
        with duckdb.connect() as con:
            v3 = con.execute("SELECT gpsno, label, probability FROM read_parquet(?) WHERE anchor_day=20",
                             [str(v3_path)]).df()
        pair = part20[["gpsno", "label", f"probability_{primary}"]].merge(v3, on="gpsno", validate="one_to_one")
        metrics["paired_primary_vs_v3_20_to_40"] = _paired(
            pair.label_x.to_numpy(dtype=int), pair[f"probability_{primary}"].to_numpy(),
            pair.probability.to_numpy(), config["seed"] + 20)

        target = features[features.anchor_day.eq(60)].sort_values("gpsno").copy()
        matched = target.gpsno.isin(set(reference.gpsno)).to_numpy()
        has_behavior = ((target.exposure_trajectory_fraction.to_numpy() > 0)
                        | (target.exposure_imu_fraction.to_numpy() > 0)); prior = float(check.loc[sorted(dev_ids)].mean())
        candidates = {}
        for variant in variants:
            selected = int(np.median([row["selected_epoch"] for row in selections if row["variant"] == variant]))
            scaler = LandmarkScalerV9(columns).fit(development)
            pack = vehicle_pack(development, outcomes, dev_ids, scaler)
            net, _ = fit_exposure(pack, settings, variant=variant, epochs=selected,
                                  seed=config["seed"] + 999, device=device)
            _save(run_dir / "models" / f"development_{variant}.pt", net, scaler, settings, variant)
            q, predicted_exposure = predict_exposure(net, scaler, target, device)
            full = horizon_probability(q, 40); probability = np.where(matched | has_behavior, full, prior)
            candidate = pd.DataFrame({"gpsno": target.gpsno.to_numpy(), "anchor_day": 60,
                "horizon_days": 40, "route": np.where(matched, "full", np.where(has_behavior,
                    "context_only_no_matched_event", "prior_no_observed_behavior")),
                "predicted_future_km_per_day": predicted_exposure, "probability": probability,
                "prediction_at_0_5": (probability >= .5).astype(int), "full_probability": full,
                "label_status": "future_unknown"})
            write_parquet(run_dir / "predictions" / f"candidate_500_{variant}.parquet", candidate)
            candidates[variant] = candidate
        write_parquet(run_dir / "predictions" / "candidate_500.parquet", candidates[primary])
        candidates[primary][["gpsno", "probability", "prediction_at_0_5"]].to_csv(
            run_dir / "predictions" / "submission_candidate.csv", index=False, encoding="utf-8-sig")
        metrics["warning"] = "Day60 后 40 天不可见；未来暴露也由模型预测，未使用真实未来里程。"
        write_json(run_dir / "metrics.json", metrics)
        lines = [f"# {manifest['run_id']}", "", "## 20→40 OOF", "",
                 "| Variant | AUC | Brier | Accuracy@0.5 | Future km/day MAE |", "|---|---:|---:|---:|---:|"]
        for variant in variants:
            score = metrics["by_variant_and_anchor"][variant]["20_to_40"]
            lines.append(f"| {variant} | {score['roc_auc']:.6f} | {score['brier']:.6f} | {score['accuracy_at_0_5']:.6f} | {score['future_km_mae']:.4f} |")
        lines += ["", "96 台锁定测试车未评分。", ""]
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
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/v9e4_exposure.json")
    args = parser.parse_args(); print(run(args.config, args.database_root))


if __name__ == "__main__":
    main()

