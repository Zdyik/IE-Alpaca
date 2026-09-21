"""Train V2 neural network on 60-day vehicle series with vehicle-disjoint folds.

The 60->future-40 target is unobserved. The comparable local test is June 20
20->40; full 60-day histories are used for train-only masked pretraining and
July 30 candidate inference. No locked-vehicle label is scored here.
"""

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

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from ie_alpaca.data.splits import audit_split, load_or_create_split  # noqa: E402
from ie_alpaca.data.task_one import labeled_reference, load_tables, prediction_index  # noqa: E402
from ie_alpaca.evaluation.metrics import auc_interval, score_binary  # noqa: E402
from ie_alpaca.features.sequence_v2 import (  # noqa: E402
    ANCHOR_DAYS, HORIZONS, assert_primary_boundary, build_sequences,
)
from ie_alpaca.tracking.run_store import (  # noqa: E402
    create_run, file_manifest, update_leaderboard, verify_source, write_json, write_parquet,
)
from ie_alpaca.training.temporal_v2 import fit_model, predict, resolve_device  # noqa: E402


def _validate_paths(input_dir: Path, results_root: Path, database_root: Path, config_path: Path) -> None:
    if not input_dir.is_relative_to(database_root) or not results_root.is_relative_to(database_root):
        raise ValueError("数据输入与实验结果必须位于数据库根目录内")
    if input_dir == results_root or input_dir in results_root.parents or results_root in input_dir.parents:
        raise ValueError("预处理目录与实验目录必须相互独立")
    if not config_path.is_relative_to(REPO):
        raise ValueError("配置必须放在项目内，以便与模型一起快照")


def _checkpoint(path: Path, model, data, scaler: dict, config: dict) -> None:
    torch.save({"state_dict": model.cpu().state_dict(), "features": data.features,
                "groups": data.groups, "scaler": scaler, "model_config": config["model"]}, path)


def run(config_path: Path, input_dir: Path, results_root: Path, database_root: Path) -> Path:
    config_path, input_dir, results_root, database_root = (
        path.resolve() for path in (config_path, input_dir, results_root, database_root)
    )
    _validate_paths(input_dir, results_root, database_root, config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert_primary_boundary()
    device = resolve_device(config["model"]["device"])
    tables = load_tables(input_dir)
    reference = labeled_reference(tables.bags)
    prediction_index(tables.bags, tables.profile)
    data = build_sequences(tables.daily, reference)
    reference_ids = set(reference.gpsno)
    results_root.mkdir(parents=True, exist_ok=True)
    split_path = results_root / "splits" / f"split_v1_seed{int(config['seed'])}.json"
    split = load_or_create_split(reference, split_path, seed=int(config["seed"]),
                                 folds=int(config["folds"]), holdout_fraction=float(config["holdout_fraction"]))
    audit = audit_split(reference, split)
    data_files = file_manifest(input_dir)
    fingerprint = hashlib.sha256(json.dumps(data_files, sort_keys=True).encode()).hexdigest()
    resolved = {**config, "input_dir": str(input_dir), "results_root": str(results_root),
                "split_version": split["version"], "feature_columns": data.features,
                "anchor_history_days": ANCHOR_DAYS, "horizons": HORIZONS,
                "training_device": str(device)}
    run_dir, manifest = create_run(REPO, results_root, config_path, resolved,
                                   version="V2", entrypoint="train_v2.py")
    try:
        manifest.update({"model": "temporal_v2", "split_version": split["version"],
                         "data_fingerprint": fingerprint, "data_files": data_files,
                         "evaluation_version": "v1_vehicle_oof_fixed_threshold_0.5",
                         "training_device": str(device), "seed": int(config["seed"]),
                         "dependencies": {name: importlib.metadata.version(name)
                                          for name in ("torch", "scikit-learn", "duckdb", "pandas", "numpy")},
                         "training_vehicles_only_for_self_supervision": True,
                         "lockbox_evaluated": False})
        write_json(run_dir / "manifest.json", manifest)
        shutil.copy2(split_path, run_dir / "splits.json")
        audit.update({"sequence_input_days": "2026-06-01..anchor inclusive",
                      "supervised_future": "strictly after anchor, observed by 2026-07-30",
                      "pretrain_scope": "full 60 days of fold-training vehicles only",
                      "normalizer_scope": "fold-training vehicles, June 1..20",
                      "target_60_to_future_40_observed": False})
        write_json(run_dir / "leakage_audit.json", audit)
        write_json(run_dir / "feature_columns.json", {"columns": data.features, "groups": data.groups})
        dev_ids = set(split["development"])
        if len(dev_ids) + len(split["holdout"]) != len(reference_ids):
            raise ValueError("车辆划分数目与标签表不一致")
        reference_labels = reference.set_index("gpsno").label
        oof_parts, auxiliary_parts, fold_metrics = [], [], []
        for fold_text, val_list in sorted(split["validation_folds"].items(), key=lambda kv: int(kv[0])):
            fold = int(fold_text)
            val_ids = set(val_list)
            train_ids = dev_ids - val_ids
            if train_ids & val_ids or train_ids & set(split["holdout"]) or val_ids & set(split["holdout"]):
                raise ValueError("训练、验证与锁定车辆发生交叉")
            model, scaler, training_log = fit_model(data, train_ids, config["model"],
                                                    int(config["seed"]) + fold, device)
            _checkpoint(run_dir / "models" / f"fold_{fold}.pt", model, data, scaler, config)
            model.to(device)
            write_json(run_dir / "models" / f"fold_{fold}_scaler.json", scaler)
            write_json(run_dir / "logs" / f"fold_{fold}.json", {"epochs": training_log})
            val_pos = data.positions(val_ids)
            ids = [data.vehicles[i] for i in val_pos]
            primary = predict(model, data, scaler, val_pos, 20, device)[:, 4]
            labels = reference_labels.loc[ids].to_numpy(dtype=int)
            fold_metrics.append({"fold": fold, **{k: v for k, v in score_binary(labels, primary).items()
                                                 if k != "confusion"}})
            oof_parts.append(pd.DataFrame({"gpsno": ids, "fold": fold,
                                           "anchor_date": "2026-06-20", "history_days": 20,
                                           "horizon_days": 40, "data_group": "development_oof",
                                           "label": labels, "probability": primary}))
            for days, horizon in zip(ANCHOR_DAYS, reversed(HORIZONS)):
                p = predict(model, data, scaler, val_pos, days, device)[:, HORIZONS.index(horizon)]
                y = data.events[val_pos, days:days+horizon].any(axis=1).astype(int)
                auxiliary_parts.append(pd.DataFrame({"gpsno": ids, "fold": fold,
                                                     "history_days": days, "horizon_days": horizon,
                                                     "label": y, "probability": p}))
            print(f"fold {fold}: {len(ids)} cars, 20→40 AUC={fold_metrics[-1]['roc_auc']:.4f}", flush=True)
        oof = pd.concat(oof_parts, ignore_index=True).sort_values("gpsno").reset_index(drop=True)
        if len(oof) != len(dev_ids) or oof.gpsno.duplicated().any() or set(oof.gpsno) != dev_ids:
            raise ValueError("主 OOF 必须为每辆开发车恰好一条预测")
        auxiliary = pd.concat(auxiliary_parts, ignore_index=True)
        write_parquet(run_dir / "predictions" / "oof_predictions.parquet", oof)
        write_parquet(run_dir / "predictions" / "auxiliary_oof.parquet", auxiliary)
        primary_score = score_binary(oof.label, oof.probability)
        auxiliary_scores = {f"{days}_to_{horizon}": score_binary(frame.label, frame.probability)
                            for (days, horizon), frame in auxiliary.groupby(["history_days", "horizon_days"])}
        metrics = {"target": "2026-06-20, first 20 days -> next observed 40 days",
                   "development_oof": primary_score,
                   "development_oof_auc_95pct_bootstrap": auc_interval(oof.label, oof.probability, int(config["seed"])),
                   "auxiliary_oof": auxiliary_scores, "holdout_evaluated": False,
                   "warning": "20→40 是代理回测；60→未来40天没有可观测标签，也没有本地实测准确率。"}
        write_json(run_dir / "metrics.json", metrics)
        pd.DataFrame(fold_metrics).to_csv(run_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
        # The development-only final model emits a 60-day candidate for all 500
        # vehicles. Locked labels never enter fitting, normalization or metrics.
        model, scaler, training_log = fit_model(data, dev_ids, config["model"],
                                                int(config["seed"]) + 999, device)
        _checkpoint(run_dir / "models" / "development.pt", model, data, scaler, config)
        model.to(device)
        write_json(run_dir / "models" / "development_scaler.json", scaler)
        write_json(run_dir / "logs" / "development.json", {"epochs": training_log})
        all_pos = np.arange(len(data.vehicles))
        candidate_probability = predict(model, data, scaler, all_pos, 60, device)[:, 4]
        candidate = pd.DataFrame({"gpsno": data.vehicles, "anchor_date": "2026-07-30",
                                  "history_days": 60, "horizon_days": 40,
                                  "data_group": "candidate_500", "label_status": "target_unknown",
                                  "event_feed_available": data.event_feed,
                                  "probability": candidate_probability,
                                  "prediction_at_0_5": (candidate_probability >= 0.5).astype(int)})
        if len(candidate) != 500 or candidate.gpsno.duplicated().any() or not candidate.probability.between(0, 1).all():
            raise ValueError("500 车候选预测无效")
        write_parquet(run_dir / "predictions" / "candidate_500.parquet", candidate)
        candidate[["gpsno", "probability", "prediction_at_0_5"]].to_csv(
            run_dir / "predictions" / "submission_candidate.csv", index=False, encoding="utf-8-sig")
        verify_source(REPO, manifest)
        (run_dir / "result_summary.md").write_text(
            f"# {manifest['run_id']}\n\n- 假设：{config['hypothesis']}\n"
            f"- 开发集 20→40 OOF ROC-AUC：{primary_score['roc_auc']:.6f}\n"
            f"- Accuracy@0.5：{primary_score['accuracy_at_0_5']:.6f}；"
            f"PR-AUC：{primary_score['pr_auc']:.6f}；Brier：{primary_score['brier']:.6f}\n"
            f"- 验证车：{len(oof)}；锁定车：{len(split['holdout'])}，未用于训练或评分。\n"
            "- 60 天输入候选仅为未标注推理；没有 60→40 真实测试分数。\n", encoding="utf-8")
        manifest.update({"status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
                         "development_vehicles": len(dev_ids), "holdout_vehicles": len(split["holdout"])})
        write_json(run_dir / "manifest.json", manifest)
        update_leaderboard(results_root)
        return run_dir
    except Exception:
        manifest.update({"status": "failed", "failed_utc": datetime.now(timezone.utc).isoformat(),
                         "error": traceback.format_exc()})
        write_json(run_dir / "manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-root", type=Path, default=Path(r"E:\Databases\2026 清华IE亮剑-算法赛道赛题"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "experiments" / "v2_temporal.json")
    args = parser.parse_args()
    root = args.database_root.resolve()
    print(run(args.config, args.input or root / "任务一预处理结果",
              args.results or root / "任务一实验结果", root))


if __name__ == "__main__":
    main()
