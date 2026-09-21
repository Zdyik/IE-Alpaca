"""Train and archive V3 Model 0; the 96 locked vehicles are never scored."""

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

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from ie_alpaca.data.splits import audit_split, load_or_create_split  # noqa: E402
from ie_alpaca.data.task_one import labeled_reference, load_tables  # noqa: E402
from ie_alpaca.evaluation.metrics import auc_interval, score_binary  # noqa: E402
from ie_alpaca.features.landmark_v3 import ANCHORS, build_landmarks, feature_columns  # noqa: E402
from ie_alpaca.tracking.run_store import (  # noqa: E402
    create_run, file_manifest, update_leaderboard, verify_source, write_json, write_parquet,
)
from ie_alpaca.training.landmark_v3 import (  # noqa: E402
    HazardLogistic, compress_outcomes, horizon_probability, make_outcomes,
)


def _score_by_anchor(oof: pd.DataFrame) -> dict:
    result = {}
    for t, part in oof.groupby("anchor_day"):
        result[f"{t}_to_{int(part.horizon_days.iloc[0])}"] = {
            "full": score_binary(part.label, part.probability),
            "fallback": score_binary(part.label, part.fallback_probability),
            "auc_95pct_bootstrap": auc_interval(part.label, part.probability, 2026 + int(t)),
        }
    return result


def _paired_v1(oof: pd.DataFrame, results_root: Path, seed: int) -> dict | None:
    paths = sorted((results_root / "runs").glob("V1_*/predictions/oof_predictions.parquet"))
    if not paths:
        return None
    import duckdb

    reference_path = paths[-1]
    con = duckdb.connect()
    try:
        old = con.execute("SELECT gpsno, label, probability FROM read_parquet(?)", [str(reference_path)]).df()
    finally:
        con.close()
    current = oof[oof.anchor_day.eq(20)][["gpsno", "label", "probability"]]
    pair = current.merge(old, on="gpsno", suffixes=("_v3", "_v1"), validate="one_to_one")
    if len(pair) != len(current) or not pair.label_v3.eq(pair.label_v1).all():
        raise ValueError("V1 与 V3 的 20→40 OOF 车辆或标签不一致")
    y = pair.label_v3.to_numpy(dtype=int)
    p3, p1 = pair.probability_v3.to_numpy(), pair.probability_v1.to_numpy()
    rng, diffs = np.random.default_rng(seed), []
    for _ in range(1000):
        ix = rng.integers(0, len(pair), len(pair))
        if len(np.unique(y[ix])) == 2:
            diffs.append(roc_auc_score(y[ix], p3[ix]) - roc_auc_score(y[ix], p1[ix]))
    return {"v1_run_id": reference_path.parents[1].name, "vehicles": len(pair),
            "v1_auc": float(roc_auc_score(y, p1)), "v3_auc": float(roc_auc_score(y, p3)),
            "delta_auc": float(roc_auc_score(y, p3) - roc_auc_score(y, p1)),
            "delta_auc_95pct_paired_bootstrap": [float(x) for x in np.quantile(diffs, [0.025, 0.975])]}


def run(config_path: Path, database_root: Path) -> Path:
    config_path, database_root = config_path.resolve(), database_root.resolve()
    if not config_path.is_relative_to(REPO):
        raise ValueError("配置文件必须位于项目目录，才可一并保存代码快照")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if tuple(config["anchors"]) != ANCHORS or config["C"] != 0.1 or config["l1_ratio"] != 0.2:
        raise ValueError("Model 0 首轮参数已冻结；更改参数须创建新版本")
    input_dir = database_root / "任务一预处理结果"
    results_root = database_root / "任务一实验结果"
    tables = load_tables(input_dir)
    reference = labeled_reference(tables.bags)
    results_root.mkdir(parents=True, exist_ok=True)
    split_path = results_root / "splits" / f"split_v1_seed{config['seed']}.json"
    split = load_or_create_split(reference, split_path, seed=config["seed"], folds=config["folds"],
                                 holdout_fraction=config["holdout_fraction"])
    audit = audit_split(reference, split)
    profile_ids = sorted(tables.profile.gpsno.astype(str).tolist())
    matched_ids = set(reference.gpsno)
    if len(matched_ids) != 478 or len(profile_ids) != 500:
        raise ValueError("监督候选或提交车辆数量与数据审计不符")
    all_features = build_landmarks(tables.daily, profile_ids, anchors=(*ANCHORS, 60))
    columns, fallback_columns = feature_columns(all_features)
    dev_ids = set(split["development"])
    dev_landmarks = all_features[all_features.gpsno.isin(dev_ids) & all_features.anchor_day.ne(60)].copy()
    outcomes = make_outcomes(tables.daily, dev_landmarks)
    old = reference.set_index("gpsno").label
    check20 = outcomes[outcomes.anchor_day.eq(20)].set_index("gpsno").label
    if not check20.eq(old.loc[check20.index]).all():
        raise ValueError("V3 20→40 标签与冻结 V1 标签不一致")
    compressed = compress_outcomes(outcomes)
    files = file_manifest(input_dir)
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    resolved = {**config, "input_dir": str(input_dir), "results_root": str(results_root),
                "split_version": split["version"], "feature_columns": columns,
                "fallback_feature_columns": fallback_columns}
    run_dir, manifest = create_run(REPO, results_root, config_path, resolved,
                                   version="V3M0", entrypoint="train_v3_model0.py")
    try:
        manifest.update({"model": "landmark_elasticnet_hazard", "split_version": split["version"],
                         "data_fingerprint": fingerprint, "data_files": files,
                         "evaluation_version": "v3_vehicle_oof_stationary_hazard_fixed_threshold_0.5",
                         "dependencies": {name: importlib.metadata.version(name) for name in
                                          ("scikit-learn", "duckdb", "pandas", "numpy", "joblib")},
                         "locked_labels_used_in_fit_or_metrics": False})
        write_json(run_dir / "manifest.json", manifest)
        shutil.copy2(split_path, run_dir / "splits.json")
        audit.update({"landmark_days": list(ANCHORS), "event_record_gate": False,
                      "day_end_boundary": "23:59:59", "label_days_start": "t+1",
                      "no_matched_event_record_vehicles": 22})
        write_json(run_dir / "leakage_audit.json", audit)
        oof_parts = []
        for fold_text, validation in sorted(split["validation_folds"].items(), key=lambda x: int(x[0])):
            val_ids = set(validation)
            train_ids = dev_ids - val_ids
            train_landmarks = dev_landmarks[dev_landmarks.gpsno.isin(train_ids)]
            val_landmarks = dev_landmarks[dev_landmarks.gpsno.isin(val_ids)].copy()
            train_rows = compressed[compressed.gpsno.isin(train_ids)]
            full = HazardLogistic(columns, c=config["C"], seed=config["seed"] + int(fold_text)).fit(train_landmarks, train_rows)
            fallback = HazardLogistic(fallback_columns, c=config["C"], seed=config["seed"] + int(fold_text)).fit(train_landmarks, train_rows)
            val = val_landmarks[["gpsno", "anchor_day"]].merge(outcomes, on=["gpsno", "anchor_day"], validate="one_to_one")
            val["fold"] = int(fold_text)
            val["daily_hazard"] = full.daily_hazard(val_landmarks)
            val["fallback_daily_hazard"] = fallback.daily_hazard(val_landmarks)
            val["probability"] = horizon_probability(val.daily_hazard, val.horizon_days)
            val["fallback_probability"] = horizon_probability(val.fallback_daily_hazard, val.horizon_days)
            oof_parts.append(val)
            joblib.dump(full, run_dir / "models" / f"fold_{fold_text}_full.joblib")
            joblib.dump(fallback, run_dir / "models" / f"fold_{fold_text}_fallback.joblib")
        oof = pd.concat(oof_parts, ignore_index=True).sort_values(["anchor_day", "gpsno"])
        if len(oof) != len(dev_ids) * len(ANCHORS) or oof.duplicated(["gpsno", "anchor_day"]).any():
            raise ValueError("OOF 未覆盖每辆开发车的每个锚点")
        write_parquet(run_dir / "predictions" / "oof_landmarks.parquet", oof)
        anchor_scores = _score_by_anchor(oof)
        metrics = {"target": "day-end landmark next-event hazard; right censored at day 60",
                   "development_oof": anchor_scores["20_to_40"]["full"],
                   "by_anchor": anchor_scores, "paired_v1_20_to_40": _paired_v1(oof, results_root, config["seed"]),
                   "holdout_evaluated": False,
                   "warning": "只验证观测期内的早期锚点；不代表 60→40 的真实成绩。"}
        write_json(run_dir / "metrics.json", metrics)
        full = HazardLogistic(columns, c=config["C"], seed=config["seed"] + 999).fit(dev_landmarks, compressed)
        fallback = HazardLogistic(fallback_columns, c=config["C"], seed=config["seed"] + 999).fit(dev_landmarks, compressed)
        joblib.dump(full, run_dir / "models" / "development_full.joblib")
        joblib.dump(fallback, run_dir / "models" / "development_fallback.joblib")
        target = all_features[all_features.anchor_day.eq(60)].copy().sort_values("gpsno")
        pfull = horizon_probability(full.daily_hazard(target), 40)
        pback = horizon_probability(fallback.daily_hazard(target), 40)
        route = target.gpsno.isin(matched_ids).to_numpy()
        has_behavior = (target.exposure_trajectory_fraction.to_numpy() > 0) | (target.exposure_imu_fraction.to_numpy() > 0)
        prior = float(outcomes[outcomes.anchor_day.eq(20)].label.mean())
        probability = np.where(route, pfull, np.where(has_behavior, pback, prior))
        route_name = np.where(route, "full", np.where(has_behavior, "fallback_no_matched_event_record", "prior_no_observed_behavior"))
        metrics["candidate_routing"] = {
            "full": int(route.sum()),
            "fallback_no_matched_event_record": int((~route & has_behavior).sum()),
            "prior_no_observed_behavior": int((~route & ~has_behavior).sum()),
            "development_20_to_40_prior": prior,
        }
        write_json(run_dir / "metrics.json", metrics)
        candidate = pd.DataFrame({"gpsno": target.gpsno.to_numpy(), "anchor_day": 60,
                                  "horizon_days": 40, "route": route_name,
                                  "probability": probability, "prediction_at_0_5": (probability >= .5).astype(int),
                                  "full_probability": pfull, "fallback_probability": pback,
                                  "label_status": "future_unknown"})
        if len(candidate) != 500 or candidate.gpsno.duplicated().any():
            raise ValueError("候选预测必须覆盖恰好 500 台车")
        write_parquet(run_dir / "predictions" / "candidate_500.parquet", candidate)
        candidate[["gpsno", "probability", "prediction_at_0_5"]].to_csv(
            run_dir / "predictions" / "submission_candidate.csv", index=False, encoding="utf-8-sig")
        score = metrics["development_oof"]
        (run_dir / "result_summary.md").write_text(
            f"# {manifest['run_id']}\n\n- 20→40 OOF AUC: {score['roc_auc']:.6f}\n"
            f"- Brier: {score['brier']:.6f}; Accuracy@0.5: {score['accuracy_at_0_5']:.6f}\n"
            "- 96 台锁定测试车未评分；500 台 Day60 候选预测来自 382 台开发车训练的模型。\n"
            "- 本地无法验证 Day60→未来40天的真实成绩。\n", encoding="utf-8")
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
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/v3_model0_logistic.json")
    args = parser.parse_args()
    print(run(args.config, args.database_root))


if __name__ == "__main__":
    main()
