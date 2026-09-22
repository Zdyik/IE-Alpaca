"""V6: compare wider versions of the V5 all-event hazard network."""

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
from ie_alpaca.features.landmark_v4 import (  # noqa: E402
    EVENT_CODES, EVENT_NAMES, GROUPS, QUALITY_CODES, RISK_CODES, build_features, context_columns, event_columns,
)
from ie_alpaca.models.event_hazard_v6 import EventHazardNetV6  # noqa: E402
from ie_alpaca.tracking.run_store import (  # noqa: E402
    create_run, file_manifest, update_leaderboard, verify_source, write_json, write_parquet,
)
from ie_alpaca.training.event_v4 import (  # noqa: E402
    FoldScaler, fit_network, predict_daily, resolve_device, vehicle_batch,
)
from ie_alpaca.training.event_v5 import inner_vehicle_split, select_epochs  # noqa: E402
from ie_alpaca.training.landmark_v3 import horizon_probability, make_outcomes  # noqa: E402


def paired_auc(current: pd.DataFrame, prior: pd.DataFrame, *, seed: int) -> dict:
    old = prior[["gpsno", "label", "probability"]]
    pair = current[["gpsno", "label", "probability"]].merge(
        old, on="gpsno", suffixes=("_new", "_prior"), validate="one_to_one",
    )
    if len(pair) != len(current) or not pair.label_new.eq(pair.label_prior).all():
        raise ValueError("配对车辆或 20→40 标签不一致")
    y = pair.label_new.to_numpy(dtype=int)
    new = pair.probability_new.to_numpy(dtype=float)
    old_p = pair.probability_prior.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    delta = []
    for _ in range(1000):
        ix = rng.integers(0, len(pair), len(pair))
        if len(np.unique(y[ix])) == 2:
            delta.append(roc_auc_score(y[ix], new[ix]) - roc_auc_score(y[ix], old_p[ix]))
    return {"vehicles": len(pair), "new_auc": float(roc_auc_score(y, new)),
            "prior_auc": float(roc_auc_score(y, old_p)),
            "delta_auc": float(roc_auc_score(y, new) - roc_auc_score(y, old_p)),
            "delta_auc_95pct_paired_bootstrap": [float(v) for v in np.quantile(delta, [.025, .975])]}


def read_v3_reference(results_root: Path, run_id: str) -> pd.DataFrame:
    path = results_root / "runs" / run_id / "predictions" / "oof_landmarks.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    with duckdb.connect() as con:
        return con.execute("SELECT gpsno, label, probability FROM read_parquet(?) WHERE anchor_day=20",
                           [str(path)]).df()


def _weights(net, fold: int, variant: str, support: dict[int, int]) -> list[dict]:
    values = net.weight_report()
    group_lookup = {code: group for group, codes in GROUPS.items() for code in codes}
    return [{"fold": fold, "variant": variant, "event_code": code,
             "event_name": EVENT_NAMES[code], "group": group_lookup[code],
             "weight_kind": "signed_quality" if code in QUALITY_CODES else "relative_nonnegative_risk",
             "weight": values[code], "training_vehicles_with_history": support[code]}
            for code in EVENT_CODES]


def _save_model(path: Path, net, scaler: FoldScaler, config: dict, variant: str) -> None:
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in net.state_dict().items()},
                "variant": variant, "model_config": config, "scaler": scaler.describe(),
                "event_codes": list(EVENT_CODES), "context_columns": context_columns()}, path)


def run(config_path: Path, database_root: Path) -> Path:
    config_path, database_root = config_path.resolve(), database_root.resolve()
    if not config_path.is_relative_to(REPO):
        raise ValueError("配置文件必须在项目内，以便归档")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if tuple(config["anchors"]) != ANCHORS or config["threshold"] != 0.5:
        raise ValueError("V6 锚点与分类阈值必须与 V5 一致")
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
    if len(reference) != 478 or len(profile_ids) != 500:
        raise ValueError("车辆数量与冻结协议不一致")
    features = build_features(tables.daily, profile_ids, anchors=(*ANCHORS, 60))
    development = features[features.gpsno.isin(dev_ids) & features.anchor_day.ne(60)].copy()
    outcomes = make_outcomes(tables.daily, development)
    at20 = outcomes[outcomes.anchor_day.eq(20)].set_index("gpsno").label
    if not at20.eq(reference.set_index("gpsno").loc[at20.index, "label"]).all():
        raise ValueError("V6 的 20→40 标签与 V5/V4/V3 不一致")
    data_files = file_manifest(input_dir)
    fingerprint = hashlib.sha256(json.dumps(data_files, sort_keys=True).encode()).hexdigest()
    device = resolve_device(config["model"]["device"])
    network_factory = partial(
        EventHazardNetV6,
        event_hidden=int(config["model"]["event_hidden"]),
        context_hidden=int(config["model"]["context_hidden"]),
    )
    example_net = network_factory(2 * len(context_columns()), learned_weights=True,
                                  dropout=float(config["model"]["dropout"]), base_daily_hazard=.01)
    parameter_count = sum(value.numel() for value in example_net.parameters())
    resolved = {**config, "input_dir": str(input_dir), "results_root": str(results_root),
                "split_version": split["version"], "event_codes": list(EVENT_CODES),
                "event_names": EVENT_NAMES, "context_columns": context_columns(),
                "event_columns": event_columns(), "resolved_device": str(device),
                "parameter_count": parameter_count}
    run_dir, manifest = create_run(REPO, results_root, config_path, resolved,
                                   version="V6", entrypoint="train_v6.py")
    try:
        manifest.update({"model": "torch_all_event_stationary_hazard_v6_width", "split_version": split["version"],
                         "data_files": data_files, "data_fingerprint": fingerprint,
                         "evaluation_version": "v6_vehicle_oof_stationary_hazard_width_sweep_0.5",
                         "parameter_count": parameter_count,
                         "device": str(device), "locked_labels_used_in_fit_or_metrics": False,
                         "dependencies": {name: importlib.metadata.version(name) for name in
                                          ("torch", "scikit-learn", "duckdb", "pandas", "numpy")}})
        write_json(run_dir / "manifest.json", manifest)
        shutil.copy2(split_path, run_dir / "splits.json")
        audit.update({"landmark_days": list(ANCHORS), "event_record_gate": False,
                      "all_24_event_codes": True, "day_end_boundary": "23:59:59", "label_start": "t+1"})
        write_json(run_dir / "leakage_audit.json", audit)

        oof_parts, weight_rows, history_rows = [], [], []
        selection_rows, curve_rows, inner_splits = [], [], {}
        labels20 = reference.set_index("gpsno").label.astype(int)
        for fold_text, val_list in sorted(split["validation_folds"].items(), key=lambda x: int(x[0])):
            fold = int(fold_text)
            val_ids = set(val_list)
            train_ids = dev_ids - val_ids
            inner_fit_ids, inner_val_ids = inner_vehicle_split(
                train_ids, labels20, seed=config["seed"] + 100 + fold,
                validation_fraction=config["selection"]["validation_fraction"],
            )
            inner_splits[fold_text] = {"fit": sorted(inner_fit_ids), "validation": sorted(inner_val_ids)}
            train_features = development[development.gpsno.isin(train_ids)]
            val_features = development[development.gpsno.isin(val_ids)].copy()
            inner_scaler = FoldScaler().fit(development[development.gpsno.isin(inner_fit_ids)])
            inner_fit_pack = vehicle_batch(development, outcomes, inner_fit_ids, inner_scaler)
            inner_val_pack = vehicle_batch(development, outcomes, inner_val_ids, inner_scaler)
            scaler = FoldScaler().fit(train_features)
            pack = vehicle_batch(development, outcomes, train_ids, scaler)
            val = val_features[["gpsno", "anchor_day"]].merge(
                outcomes, on=["gpsno", "anchor_day"], validate="one_to_one",
            )
            val["fold"] = fold
            support = {code: int((train_features[train_features.anchor_day.eq(53)]
                                  [f"event_{code}_episodes_per_day"] > 0).sum()) for code in EVENT_CODES}
            for variant, learned in (("uniform", False), ("learned", True)):
                selected_epoch, curve = select_epochs(
                    inner_fit_pack, inner_val_pack, {**config["model"], **config["selection"]},
                    learned_weights=learned, seed=config["seed"] + fold, device=device,
                    network_factory=network_factory,
                )
                selection_rows.append({"fold": fold, "variant": variant,
                                       "selected_epoch": selected_epoch,
                                       "observed_epochs": len(curve),
                                       "inner_fit_vehicles": len(inner_fit_ids),
                                       "inner_validation_vehicles": len(inner_val_ids),
                                       "selected_inner_val_nll": curve[selected_epoch - 1]["inner_val_nll"]})
                curve_rows.extend({"fold": fold, "variant": variant, **entry} for entry in curve)
                refit_config = {**config["model"], "epochs": selected_epoch}
                net, history = fit_network(pack, refit_config, learned_weights=learned,
                                           seed=config["seed"] + fold, device=device,
                                           network_factory=network_factory)
                q = predict_daily(net, scaler, val_features, device)
                val[f"daily_hazard_{variant}"] = q
                val[f"probability_{variant}"] = horizon_probability(q, val.horizon_days.to_numpy())
                _save_model(run_dir / "models" / f"fold_{fold}_{variant}.pt", net, scaler,
                            refit_config, variant)
                weight_rows.extend(_weights(net, fold, variant, support))
                history_rows.extend({"fold": fold, "variant": variant, **entry} for entry in history)
            oof_parts.append(val)

        write_json(run_dir / "inner_vehicle_splits.json", inner_splits)
        pd.DataFrame(selection_rows).to_csv(run_dir / "selected_epochs.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(curve_rows).to_csv(run_dir / "inner_validation_curves.csv", index=False, encoding="utf-8-sig")

        oof = pd.concat(oof_parts, ignore_index=True).sort_values(["anchor_day", "gpsno"])
        if len(oof) != len(dev_ids) * len(ANCHORS) or oof.duplicated(["gpsno", "anchor_day"]).any():
            raise ValueError("OOF 未覆盖每辆开发车的全部锚点")
        oof["probability"] = oof.probability_learned
        write_parquet(run_dir / "predictions" / "oof_landmarks.parquet", oof)
        pd.DataFrame(weight_rows).to_csv(run_dir / "event_weights_by_fold.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(history_rows).to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8-sig")
        metrics = {"target": "day-end landmark next-event hazard; right censored at day 60",
                   "holdout_evaluated": False,
                   "change_from_v5": "only event and context hidden widths change; V5 epoch selection retained",
                   "parameter_count": parameter_count,
                   "weight_interpretation": "predictive relative importance, not causal severity; camera events use signed quality weights",
                   "by_anchor": {}}
        for t, part in oof.groupby("anchor_day"):
            key = f"{t}_to_{int(part.horizon_days.iloc[0])}"
            metrics["by_anchor"][key] = {
                mode: score_binary(part.label, part[f"probability_{mode}"])
                for mode in ("uniform", "learned")
            }
            metrics["by_anchor"][key]["learned_auc_95pct_bootstrap"] = auc_interval(
                part.label, part.probability_learned, config["seed"] + int(t),
            )
        metrics["development_oof"] = metrics["by_anchor"]["20_to_40"]["learned"]
        metrics["selected_epochs"] = selection_rows
        current = oof[oof.anchor_day.eq(20)]
        v5 = read_v3_reference(results_root, config["parent_run_id"])
        metrics["paired_vs_v5_20_to_40"] = paired_auc(current, v5, seed=config["seed"])
        v3 = read_v3_reference(results_root, config["v3_reference_run_id"])
        metrics["paired_vs_v3_20_to_40"] = paired_auc(current, v3, seed=config["seed"])
        uniform = current[["gpsno", "label", "probability_uniform"]].rename(
            columns={"probability_uniform": "probability"})
        metrics["paired_learned_vs_uniform_20_to_40"] = paired_auc(current, uniform, seed=config["seed"] + 1)

        final_epoch = int(np.median([row["selected_epoch"] for row in selection_rows
                                     if row["variant"] == "learned"]))
        final_config = {**config["model"], "epochs": final_epoch}
        final_scaler = FoldScaler().fit(development)
        final_pack = vehicle_batch(development, outcomes, dev_ids, final_scaler)
        final_net, final_history = fit_network(final_pack, final_config, learned_weights=True,
                                               seed=config["seed"] + 999, device=device,
                                               network_factory=network_factory)
        _save_model(run_dir / "models" / "development_learned.pt", final_net, final_scaler,
                    final_config, "learned")
        write_json(run_dir / "models" / "development_weights.json",
                   {str(code): weight for code, weight in final_net.weight_report().items()})
        target = features[features.anchor_day.eq(60)].sort_values("gpsno").copy()
        q = predict_daily(final_net, final_scaler, target, device)
        full_probability = horizon_probability(q, 40)
        matched = target.gpsno.isin(set(reference.gpsno)).to_numpy()
        has_behavior = (target.exposure_trajectory_fraction.to_numpy() > 0) | (
            target.exposure_imu_fraction.to_numpy() > 0)
        prior = float(at20.loc[list(dev_ids)].mean())
        probability = np.where(matched | has_behavior, full_probability, prior)
        route = np.where(matched, "full", np.where(has_behavior, "context_only_no_matched_event", "prior_no_observed_behavior"))
        candidate = pd.DataFrame({"gpsno": target.gpsno.to_numpy(), "anchor_day": 60,
                                  "horizon_days": 40, "route": route, "probability": probability,
                                  "prediction_at_0_5": (probability >= .5).astype(int),
                                  "full_probability": full_probability, "label_status": "future_unknown"})
        if len(candidate) != 500 or candidate.gpsno.duplicated().any() or not np.isfinite(probability).all():
            raise ValueError("500 车候选概率无效")
        write_parquet(run_dir / "predictions" / "candidate_500.parquet", candidate)
        candidate[["gpsno", "probability", "prediction_at_0_5"]].to_csv(
            run_dir / "predictions" / "submission_candidate.csv", index=False, encoding="utf-8-sig")
        metrics["candidate_routing"] = {"full": int(matched.sum()),
                                        "context_only_no_matched_event": int((~matched & has_behavior).sum()),
                                        "prior_no_observed_behavior": int((~matched & ~has_behavior).sum()),
                                        "development_20_to_40_prior": prior}
        metrics["final_training_objective"] = final_history[-1]["train_objective"]
        metrics["final_selected_epoch"] = final_epoch
        metrics["warning"] = "本地只有早期锚点结局，Day60→未来40天没有真实标签；候选预测不等于官方成绩。"
        write_json(run_dir / "metrics.json", metrics)
        score = metrics["development_oof"]
        delta = metrics["paired_vs_v5_20_to_40"]["delta_auc"]
        (run_dir / "result_summary.md").write_text(
            f"# {manifest['run_id']}\n\n"
            f"- 参数量: {parameter_count}; 20→40 learned-weight OOF AUC: {score['roc_auc']:.6f}; 对 V5 ΔAUC: {delta:+.6f}\n"
            f"- Uniform-weight OOF AUC: {metrics['by_anchor']['20_to_40']['uniform']['roc_auc']:.6f}\n"
            f"- 外折内层选轮: {[row['selected_epoch'] for row in selection_rows if row['variant'] == 'learned']}; 最终重训 {final_epoch} 轮\n"
            f"- Brier: {score['brier']:.6f}; Accuracy@0.5: {score['accuracy_at_0_5']:.6f}\n"
            "- 96 台锁定车未评分；Day60 的 500 车输出仅为候选。\n", encoding="utf-8")
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
    parser.add_argument("--database-root", type=Path, default=Path(r"E:\Databases\2026 清华IE亮剑-算法赛道赛题"))
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/v6_width_medium.json")
    args = parser.parse_args()
    print(run(args.config, args.database_root))


if __name__ == "__main__":
    main()
