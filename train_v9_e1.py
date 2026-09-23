"""Train the four V9-E1 hierarchical additive model variants."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from functools import partial
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
from ie_alpaca.features.landmark_v3 import ANCHORS  # noqa: E402
from ie_alpaca.features.landmark_v4 import EVENT_CODES, build_features, context_columns, event_columns  # noqa: E402
from ie_alpaca.models.causal_hnam_v9 import CausalHNAMV9  # noqa: E402
from ie_alpaca.tracking.run_store import (  # noqa: E402
    create_run, file_manifest, update_leaderboard, verify_source, write_json, write_parquet,
)
from ie_alpaca.training.event_v4 import FoldScaler, fit_network, predict_daily, resolve_device, vehicle_batch  # noqa: E402
from ie_alpaca.training.event_v5 import inner_vehicle_split, select_epochs  # noqa: E402
from ie_alpaca.training.landmark_v3 import horizon_probability, make_outcomes  # noqa: E402


def _paired(y: np.ndarray, new: np.ndarray, old: np.ndarray, seed: int) -> dict:
    rng, values = np.random.default_rng(seed), []
    for _ in range(2000):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) == 2:
            values.append(roc_auc_score(y[index], new[index]) - roc_auc_score(y[index], old[index]))
    return {"delta_auc": float(roc_auc_score(y, new) - roc_auc_score(y, old)),
            "delta_auc_95pct_paired_bootstrap": [float(x) for x in np.quantile(values, [.025, .975])],
            "valid_bootstraps": len(values)}


def _read_v3(results_root: Path, run_id: str) -> pd.DataFrame:
    path = results_root / "runs" / run_id / "predictions" / "oof_landmarks.parquet"
    with duckdb.connect() as con:
        return con.execute(
            "SELECT gpsno, label, probability FROM read_parquet(?) WHERE anchor_day=20", [str(path)]).df()


def _save(path: Path, net: CausalHNAMV9, scaler: FoldScaler, config: dict, variant: str) -> None:
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                "variant": variant, "model_config": config, "scaler": scaler.describe(),
                "event_codes": list(EVENT_CODES), "context_columns": context_columns()}, path)


def run(config_path: Path, database_root: Path) -> Path:
    config_path, database_root = config_path.resolve(), database_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    variants = config["variants"]
    primary = config["primary_variant"]
    if tuple(config["anchors"]) != ANCHORS or primary not in variants:
        raise ValueError("V9-E1 配置与冻结协议不一致")
    input_dir = database_root / "任务一预处理结果"
    results_root = database_root / "任务一实验结果"
    tables = load_tables(input_dir)
    reference = labeled_reference(tables.bags)
    split_path = results_root / "splits" / f"split_v1_seed{config['seed']}.json"
    split = load_or_create_split(reference, split_path, seed=config["seed"], folds=config["folds"],
                                 holdout_fraction=config["holdout_fraction"])
    audit = audit_split(reference, split)
    dev_ids = set(split["development"])
    profile_ids = sorted(tables.profile.gpsno.astype(str).tolist())
    features = build_features(tables.daily, profile_ids, anchors=(*ANCHORS, 60))
    development = features[features.gpsno.isin(dev_ids) & features.anchor_day.ne(60)].copy()
    outcomes = make_outcomes(tables.daily, development)
    labels20 = reference.set_index("gpsno").label.astype(int)
    check = outcomes[outcomes.anchor_day.eq(20)].set_index("gpsno").label
    if len(reference) != 478 or len(dev_ids) != 382 or len(profile_ids) != 500 or not check.eq(labels20.loc[check.index]).all():
        raise ValueError("车辆数量或冻结标签不一致")
    device = resolve_device(config["model"]["device"])
    parameter_counts = {}
    for variant in variants:
        example = CausalHNAMV9(2 * len(context_columns()), variant=variant, learned_weights=True)
        parameter_counts[variant] = sum(value.numel() for value in example.parameters())
    data_files = file_manifest(input_dir)
    fingerprint = hashlib.sha256(json.dumps(data_files, sort_keys=True).encode()).hexdigest()
    resolved = {**config, "input_dir": str(input_dir), "results_root": str(results_root),
                "split_version": split["version"], "resolved_device": str(device),
                "parameter_counts": parameter_counts, "event_columns": event_columns()}
    run_dir, manifest = create_run(REPO, results_root, config_path, resolved,
                                   version="V9E1", entrypoint="train_v9_e1.py")
    try:
        manifest.update({"model": "torch_v9_hierarchical_additive_ablation", "split_version": split["version"],
                         "data_files": data_files, "data_fingerprint": fingerprint,
                         "evaluation_version": "v9_vehicle_oof_stationary_hazard_hnam_0.5",
                         "parameter_counts": parameter_counts, "device": str(device),
                         "locked_labels_used_in_fit_or_metrics": False,
                         "dependencies": {name: importlib.metadata.version(name) for name in
                                          ("torch", "scikit-learn", "duckdb", "pandas", "numpy")}})
        write_json(run_dir / "manifest.json", manifest)
        shutil.copy2(split_path, run_dir / "splits.json")
        audit.update({"landmark_days": list(ANCHORS), "all_24_event_codes": True,
                      "label_start": "t+1", "future_features_used": False,
                      "locked_holdout_scored": False, "variants": variants})
        write_json(run_dir / "leakage_audit.json", audit)

        oof_parts, selection_rows, curve_rows, history_rows, weight_rows = [], [], [], [], []
        inner_splits = {}
        for fold_text, val_list in sorted(split["validation_folds"].items(), key=lambda item: int(item[0])):
            fold = int(fold_text)
            val_ids, train_ids = set(val_list), dev_ids - set(val_list)
            inner_fit, inner_val = inner_vehicle_split(
                train_ids, labels20, seed=config["seed"] + 100 + fold,
                validation_fraction=config["selection"]["validation_fraction"])
            inner_splits[fold_text] = {"fit": sorted(inner_fit), "validation": sorted(inner_val)}
            inner_scaler = FoldScaler().fit(development[development.gpsno.isin(inner_fit)])
            inner_fit_pack = vehicle_batch(development, outcomes, inner_fit, inner_scaler)
            inner_val_pack = vehicle_batch(development, outcomes, inner_val, inner_scaler)
            train_features = development[development.gpsno.isin(train_ids)]
            scaler = FoldScaler().fit(train_features)
            train_pack = vehicle_batch(development, outcomes, train_ids, scaler)
            val_features = development[development.gpsno.isin(val_ids)].sort_values(["gpsno", "anchor_day"]).copy()
            val = val_features[["gpsno", "anchor_day"]].merge(
                outcomes, on=["gpsno", "anchor_day"], validate="one_to_one").sort_values(["gpsno", "anchor_day"])
            val["fold"] = fold
            support = {code: int((train_features[train_features.anchor_day.eq(53)]
                                  [f"event_{code}_episodes_per_day"] > 0).sum()) for code in EVENT_CODES}
            for variant in variants:
                factory = partial(CausalHNAMV9, variant=variant)
                selected, curve = select_epochs(
                    inner_fit_pack, inner_val_pack, {**config["model"], **config["selection"]},
                    learned_weights=True, seed=config["seed"] + fold, device=device,
                    network_factory=factory)
                selection_rows.append({"fold": fold, "variant": variant, "selected_epoch": selected,
                                       "observed_epochs": len(curve),
                                       "selected_inner_val_nll": curve[selected - 1]["inner_val_nll"]})
                curve_rows.extend({"fold": fold, "variant": variant, **row} for row in curve)
                refit = {**config["model"], "epochs": selected}
                net, history = fit_network(train_pack, refit, learned_weights=True,
                                           seed=config["seed"] + fold, device=device,
                                           network_factory=factory)
                q = predict_daily(net, scaler, val_features, device)
                val[f"daily_hazard_{variant}"] = q
                val[f"probability_{variant}"] = horizon_probability(q, val.horizon_days.to_numpy())
                history_rows.extend({"fold": fold, "variant": variant, **row} for row in history)
                for code, weight in net.weight_report().items():
                    weight_rows.append({"fold": fold, "variant": variant, "event_code": code,
                                        "weight": weight, "training_vehicles_with_history": support[code]})
                _save(run_dir / "models" / f"fold_{fold}_{variant}.pt", net, scaler, refit, variant)
            oof_parts.append(val)
            print(f"V9-E1 fold {fold} complete", flush=True)

        write_json(run_dir / "inner_vehicle_splits.json", inner_splits)
        pd.DataFrame(selection_rows).to_csv(run_dir / "selected_epochs.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(curve_rows).to_csv(run_dir / "inner_validation_curves.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(history_rows).to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(weight_rows).to_csv(run_dir / "event_weights_by_fold.csv", index=False, encoding="utf-8-sig")
        oof = pd.concat(oof_parts, ignore_index=True).sort_values(["anchor_day", "gpsno"])
        if len(oof) != len(dev_ids) * len(ANCHORS) or oof.duplicated(["gpsno", "anchor_day"]).any():
            raise ValueError("OOF 覆盖不完整")
        oof["daily_hazard"] = oof[f"daily_hazard_{primary}"]
        oof["probability"] = oof[f"probability_{primary}"]
        write_parquet(run_dir / "predictions" / "oof_landmarks.parquet", oof)
        metrics = {"target": "day-end landmark next-event stationary hazard; right censored at day 60",
                   "primary_variant": primary, "parameter_counts": parameter_counts,
                   "holdout_evaluated": False, "by_variant_and_anchor": {}, "selected_epochs": selection_rows}
        for variant in variants:
            metrics["by_variant_and_anchor"][variant] = {}
            for anchor, part in oof.groupby("anchor_day"):
                key = f"{anchor}_to_{int(part.horizon_days.iloc[0])}"
                score = score_binary(part.label, part[f"probability_{variant}"])
                score["auc_95pct_bootstrap"] = auc_interval(
                    part.label, part[f"probability_{variant}"], config["seed"] + int(anchor))
                metrics["by_variant_and_anchor"][variant][key] = score
        metrics["development_oof"] = metrics["by_variant_and_anchor"][primary]["20_to_40"]
        part20 = oof[oof.anchor_day.eq(20)]
        y = part20.label.to_numpy(dtype=int)
        metrics["paired_comparisons_at_20_to_40"] = {}
        for index, (left, right) in enumerate(zip(variants[1:], variants[:-1])):
            metrics["paired_comparisons_at_20_to_40"][f"{left}_minus_{right}"] = _paired(
                y, part20[f"probability_{left}"].to_numpy(),
                part20[f"probability_{right}"].to_numpy(), config["seed"] + index)
        v3 = _read_v3(results_root, config["v3_reference_run_id"])
        pair = part20[["gpsno", "label", f"probability_{primary}"]].merge(
            v3, on="gpsno", suffixes=("_new", "_v3"), validate="one_to_one")
        metrics["paired_primary_vs_v3_20_to_40"] = _paired(
            pair.label_new.to_numpy(dtype=int), pair[f"probability_{primary}"].to_numpy(),
            pair.probability.to_numpy(), config["seed"] + 20)

        target = features[features.anchor_day.eq(60)].sort_values("gpsno").copy()
        matched = target.gpsno.isin(set(reference.gpsno)).to_numpy()
        has_behavior = ((target.exposure_trajectory_fraction.to_numpy() > 0)
                        | (target.exposure_imu_fraction.to_numpy() > 0))
        prior = float(check.loc[sorted(dev_ids)].mean())
        candidates = {}
        for variant in variants:
            selected = int(np.median([row["selected_epoch"] for row in selection_rows
                                      if row["variant"] == variant]))
            scaler = FoldScaler().fit(development)
            pack = vehicle_batch(development, outcomes, dev_ids, scaler)
            net, _ = fit_network(pack, {**config["model"], "epochs": selected}, learned_weights=True,
                                 seed=config["seed"] + 999, device=device,
                                 network_factory=partial(CausalHNAMV9, variant=variant))
            _save(run_dir / "models" / f"development_{variant}.pt", net, scaler,
                  {**config["model"], "epochs": selected}, variant)
            full = horizon_probability(predict_daily(net, scaler, target, device), 40)
            probability = np.where(matched | has_behavior, full, prior)
            candidate = pd.DataFrame({"gpsno": target.gpsno.to_numpy(), "anchor_day": 60,
                "horizon_days": 40, "route": np.where(matched, "full", np.where(has_behavior,
                    "context_only_no_matched_event", "prior_no_observed_behavior")),
                "probability": probability, "prediction_at_0_5": (probability >= .5).astype(int),
                "full_probability": full, "label_status": "future_unknown"})
            write_parquet(run_dir / "predictions" / f"candidate_500_{variant}.parquet", candidate)
            candidates[variant] = candidate
        write_parquet(run_dir / "predictions" / "candidate_500.parquet", candidates[primary])
        candidates[primary][["gpsno", "probability", "prediction_at_0_5"]].to_csv(
            run_dir / "predictions" / "submission_candidate.csv", index=False, encoding="utf-8-sig")
        metrics["candidate_routing"] = {"full": int(matched.sum()),
            "context_only_no_matched_event": int((~matched & has_behavior).sum()),
            "prior_no_observed_behavior": int((~matched & ~has_behavior).sum()),
            "development_20_to_40_prior": prior}
        metrics["warning"] = "Day60 后 40 天在本地不可见；OOF 是早期锚点代理验证。"
        write_json(run_dir / "metrics.json", metrics)
        lines = [f"# {manifest['run_id']}", "", "## 20→40 OOF", "",
                 "| Variant | Parameters | AUC | Brier | Accuracy@0.5 |", "|---|---:|---:|---:|---:|"]
        for variant in variants:
            score = metrics["by_variant_and_anchor"][variant]["20_to_40"]
            lines.append(f"| {variant} | {parameter_counts[variant]} | {score['roc_auc']:.6f} | {score['brier']:.6f} | {score['accuracy_at_0_5']:.6f} |")
        lines += ["", "96 台锁定测试车未评分。", ""]
        (run_dir / "result_summary.md").write_text("\n".join(lines), encoding="utf-8")
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
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/v9e1_hnam.json")
    args = parser.parse_args()
    print(run(args.config, args.database_root))


if __name__ == "__main__":
    main()

