"""Train the first task-one 20→40 CatBoost baseline and archive the whole version.

This does not preprocess raw files. See README for the required Parquet inputs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from ie_alpaca.data.splits import audit_split, load_or_create_split  # noqa: E402
from ie_alpaca.data.task_one import (  # noqa: E402
    REFERENCE_ANCHOR,
    PREDICTION_ANCHOR,
    labeled_reference,
    load_tables,
    prediction_index,
)
from ie_alpaca.evaluation.metrics import auc_interval, score_binary  # noqa: E402
from ie_alpaca.features.tabular_v1 import build_features, feature_columns  # noqa: E402
from ie_alpaca.models.catboost_v1 import create_model  # noqa: E402
from ie_alpaca.tracking.run_store import (  # noqa: E402
    create_run, file_manifest, update_leaderboard, verify_source, write_json, write_parquet,
)


def _predict(model, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    probabilities = model.predict_proba(frame[columns])[:, 1]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("模型产生无效概率")
    return probabilities


def _fit_pair(train: pd.DataFrame, columns: list[str], fallback_columns: list[str], config: dict, seed: int):
    y = train.label.to_numpy(dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("训练折只有一个类别，不能训练二分类模型")
    full = create_model(config["model"], seed)
    fallback = create_model(config["model"], seed + 1000)
    full.fit(train[columns], y)
    fallback.fit(train[fallback_columns], y)
    return full, fallback


def _validate_inputs(input_dir: Path, results_root: Path, database_root: Path, config_path: Path) -> None:
    for name, path in (("input", input_dir), ("results", results_root)):
        if not path.is_relative_to(database_root):
            raise ValueError(f"{name} 路径必须位于数据库目录内：{database_root}")
    if input_dir == results_root or input_dir in results_root.parents or results_root in input_dir.parents:
        raise ValueError("预处理输入与实验结果目录必须相互独立")
    if not config_path.is_relative_to(REPO):
        raise ValueError("实验配置必须保存在项目目录中，才能进入代码快照")


def run(config_path: Path, input_dir: Path, results_root: Path, database_root: Path) -> Path:
    config_path, input_dir, results_root, database_root = (
        path.resolve() for path in (config_path, input_dir, results_root, database_root)
    )
    _validate_inputs(input_dir, results_root, database_root, config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config["lookback_days"]) != 20:
        raise ValueError("V1 必须使用 20 天历史；60→40 尚无直接监督验证")
    seed, folds = int(config["seed"]), int(config["folds"])
    tables = load_tables(input_dir)
    reference = labeled_reference(tables.bags)
    target = prediction_index(tables.bags, tables.profile)
    reference_features = build_features(tables.daily, reference, config["lookback_days"])
    target_features = build_features(tables.daily, target, config["lookback_days"])
    columns, fallback_columns = feature_columns(reference_features)
    if columns != feature_columns(target_features)[0]:
        raise ValueError("训练与预测特征列不一致")
    reference = reference[["gpsno", "anchor_date", "label", "label_status"]].merge(
        reference_features, on=["gpsno", "anchor_date"], validate="one_to_one"
    )
    target = target[["gpsno", "anchor_date", "label_status"]].merge(
        target_features, on=["gpsno", "anchor_date"], validate="one_to_one"
    )
    results_root.mkdir(parents=True, exist_ok=True)
    split_path = results_root / "splits" / f"split_v1_seed{seed}.json"
    split = load_or_create_split(
        reference, split_path, seed=seed, folds=folds,
        holdout_fraction=float(config["holdout_fraction"]),
    )
    leakage_audit = audit_split(reference, split)
    data_files = file_manifest(input_dir)
    data_fingerprint = __import__("hashlib").sha256(
        json.dumps(data_files, sort_keys=True).encode("utf-8")
    ).hexdigest()
    resolved_config = {
        **config,
        "input_dir": str(input_dir),
        "results_root": str(results_root),
        "split_version": split["version"],
        "feature_columns": columns,
        "fallback_feature_columns": fallback_columns,
        "label_policy": split["label_policy"],
    }
    run_dir, manifest = create_run(REPO, results_root, config_path, resolved_config)
    try:
        manifest.update({
            "split_version": split["version"], "data_fingerprint": data_fingerprint,
            "data_files": data_files, "seed": seed, "lookback_days": 20,
            "training_device": config["model"]["device"],
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("catboost", "scikit-learn", "duckdb", "pandas", "numpy")
            },
            "evaluation_version": "v1_vehicle_oof_fixed_threshold_0.5",
        })
        write_json(run_dir / "manifest.json", manifest)
        shutil.copy2(split_path, run_dir / "splits.json")
        write_json(run_dir / "leakage_audit.json", leakage_audit)
        (run_dir / "feature_columns.json").write_text(
            json.dumps({"full": columns, "fallback": fallback_columns}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        dev_ids = set(split["development"])
        development = reference[reference.gpsno.isin(dev_ids)].copy()
        if len(development) != len(dev_ids):
            raise ValueError("开发集不是每车一条样本")
        oof_parts, fold_rows = [], []
        for fold_text, val_list in sorted(split["validation_folds"].items(), key=lambda x: int(x[0])):
            fold = int(fold_text)
            val_ids = set(val_list)
            train = development[~development.gpsno.isin(val_ids)]
            val = development[development.gpsno.isin(val_ids)].copy()
            locked_ids = set(split["holdout"])
            if (
                train.empty or val.empty or set(train.gpsno) & set(val.gpsno)
                or set(train.gpsno) & locked_ids or set(val.gpsno) & locked_ids
            ):
                raise ValueError(f"折 {fold} 的训练/验证车辆有问题")
            full, fallback = _fit_pair(train, columns, fallback_columns, config, seed + fold)
            full_prob = _predict(full, val, columns)
            fallback_prob = _predict(fallback, val, fallback_columns)
            has_feed = val.event_feed_at_anchor.to_numpy(dtype=bool)
            probability = np.where(has_feed, full_prob, fallback_prob)
            fold_score = score_binary(val.label, probability)
            fold_rows.append({"fold": fold, **{k: v for k, v in fold_score.items() if k != "confusion"}})
            fold_result = pd.DataFrame({
                "gpsno": val.gpsno.to_numpy(),
                "anchor_date": str(REFERENCE_ANCHOR),
                "horizon_days": 40,
                "fold": fold,
                "data_group": "development_oof",
                "label": val.label.to_numpy(dtype=int),
                "label_status": val.label_status.to_numpy(),
                "route": np.where(has_feed, "full", "exposure_fallback"),
                "probability": probability,
                "full_probability": full_prob,
                "fallback_probability": fallback_prob,
            })
            oof_parts.append(fold_result)
            full.save_model(str(run_dir / "models" / f"fold_{fold}_full.cbm"))
            fallback.save_model(str(run_dir / "models" / f"fold_{fold}_fallback.cbm"))
        oof = pd.concat(oof_parts, ignore_index=True).sort_values("gpsno").reset_index(drop=True)
        if len(oof) != len(dev_ids) or oof.gpsno.duplicated().any() or set(oof.gpsno) != dev_ids:
            raise ValueError("OOF 预测未一车一条覆盖全部开发集")
        write_parquet(run_dir / "predictions" / "oof_predictions.parquet", oof)
        metrics = {
            "target": "2026-06-20 anchor, 20-day history to next 40 days",
            "development_oof": score_binary(oof.label, oof.probability),
            "development_oof_auc_95pct_bootstrap": auc_interval(oof.label, oof.probability, seed),
            "simulated_missing_event_feed": score_binary(oof.label, oof.fallback_probability),
            "holdout_evaluated": False,
            "warning": "本地代理任务；不是 60 天历史预测未来 40 天的实测指标。",
        }
        write_json(run_dir / "metrics.json", metrics)
        pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")

        # The locked group is neither fitted nor scored by V1. A dev-only model
        # still produces an inspectable 500-vehicle candidate prediction file.
        final_full, final_fallback = _fit_pair(development, columns, fallback_columns, config, seed + 999)
        final_full.save_model(str(run_dir / "models" / "development_full.cbm"))
        final_fallback.save_model(str(run_dir / "models" / "development_fallback.cbm"))
        full_prob = _predict(final_full, target, columns)
        fallback_prob = _predict(final_fallback, target, fallback_columns)
        has_feed = target.event_feed_at_anchor.to_numpy(dtype=bool)
        probability = np.where(has_feed, full_prob, fallback_prob)
        predictions = pd.DataFrame({
            "gpsno": target.gpsno.to_numpy(),
            "anchor_date": str(PREDICTION_ANCHOR),
            "horizon_days": 40,
            "data_group": "candidate_500",
            "label_status": target.label_status.to_numpy(),
            "route": np.where(has_feed, "full", "exposure_fallback"),
            "probability": probability,
            "prediction_at_0_5": (probability >= 0.5).astype(int),
            "full_probability": full_prob,
            "fallback_probability": fallback_prob,
        })
        if len(predictions) != 500 or predictions.gpsno.duplicated().any():
            raise ValueError("候选预测必须恰好覆盖 500 台唯一车辆")
        write_parquet(run_dir / "predictions" / "candidate_500.parquet", predictions)
        predictions[["gpsno", "probability", "prediction_at_0_5"]].to_csv(
            run_dir / "predictions" / "submission_candidate.csv", index=False, encoding="utf-8-sig"
        )
        verify_source(REPO, manifest)
        score = metrics["development_oof"]
        (run_dir / "result_summary.md").write_text(
            f"# {manifest['run_id']}\n\n"
            f"- 假设：{config['hypothesis']}\n"
            f"- 主指标：开发集 20→40 OOF ROC-AUC = {score['roc_auc']:.6f}\n"
            f"- Accuracy@0.5 = {score['accuracy_at_0_5']:.6f}；PR-AUC = {score['pr_auc']:.6f}；Brier = {score['brier']:.6f}\n"
            f"- 样本：{score['n']} 台开发车辆，正例 {score['positive']} 台。\n"
            "- 锁定组未评分；候选 500 车预测由开发集模型生成。\n"
            "- 这些分数属于 20→40 代理回测，不能当作 60→40 的验证结果。\n",
            encoding="utf-8",
        )
        manifest["status"] = "completed"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["development_vehicles"] = len(dev_ids)
        manifest["holdout_vehicles"] = len(split["holdout"])
        write_json(run_dir / "manifest.json", manifest)
        update_leaderboard(results_root)
    except Exception:
        manifest["status"] = "failed"
        manifest["failed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = traceback.format_exc()
        write_json(run_dir / "manifest.json", manifest)
        raise
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-root", type=Path, default=Path(r"E:\Databases\2026 清华IE亮剑-算法赛道赛题"))
    parser.add_argument("--input", type=Path, help="处理完成的 daily_features/bag_index/profile 所在目录")
    parser.add_argument("--results", type=Path, help="实验结果目录；必须在数据库根目录内")
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "experiments" / "v1_catboost.json")
    args = parser.parse_args()
    db_root = args.database_root.resolve()
    input_dir = args.input or db_root / "任务一预处理结果"
    results_root = args.results or db_root / "任务一实验结果"
    print(run(args.config, input_dir, results_root, db_root))


if __name__ == "__main__":
    main()
