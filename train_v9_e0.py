"""Run the V9-E0 human-prior feature ablation without touching the locked holdout."""

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
from ie_alpaca.features.causal_v9 import build_causal_landmarks  # noqa: E402
from ie_alpaca.features.landmark_v3 import ANCHORS  # noqa: E402
from ie_alpaca.tracking.run_store import (  # noqa: E402
    create_run, file_manifest, update_leaderboard, verify_source, write_json, write_parquet,
)
from ie_alpaca.training.landmark_v3 import (  # noqa: E402
    HazardLogistic, compress_outcomes, horizon_probability, make_outcomes,
)


def _paired_auc(y: np.ndarray, candidate: np.ndarray, reference: np.ndarray, seed: int) -> dict:
    observed = float(roc_auc_score(y, candidate) - roc_auc_score(y, reference))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(2000):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) == 2:
            differences.append(roc_auc_score(y[index], candidate[index]) - roc_auc_score(y[index], reference[index]))
    low, high = np.quantile(differences, [0.025, 0.975])
    return {"delta_auc": observed, "delta_auc_95pct_paired_bootstrap": [float(low), float(high)],
            "valid_bootstraps": len(differences)}


def _score_variants(oof: pd.DataFrame, variants: list[str], seed: int) -> tuple[dict, dict]:
    by_variant, comparisons = {}, {}
    for variant in variants:
        probability = f"probability_{variant}"
        by_anchor = {}
        for anchor, part in oof.groupby("anchor_day"):
            key = f"{anchor}_to_{int(part.horizon_days.iloc[0])}"
            by_anchor[key] = score_binary(part.label, part[probability])
            by_anchor[key]["auc_95pct_bootstrap"] = auc_interval(
                part.label, part[probability], seed + int(anchor))
        by_variant[variant] = by_anchor
    baseline = "e0a_v3_base"
    anchor20 = oof[oof.anchor_day.eq(20)]
    y = anchor20.label.to_numpy(dtype=int)
    reference = anchor20[f"probability_{baseline}"].to_numpy()
    for i, variant in enumerate(variants[1:], start=1):
        comparisons[f"{variant}_minus_{baseline}"] = _paired_auc(
            y, anchor20[f"probability_{variant}"].to_numpy(), reference, seed + i)
    for i, (left, right) in enumerate(zip(variants[1:], variants[:-1]), start=10):
        comparisons[f"{left}_minus_{right}"] = _paired_auc(
            y, anchor20[f"probability_{left}"].to_numpy(),
            anchor20[f"probability_{right}"].to_numpy(), seed + i)
    return by_variant, comparisons


def run(config_path: Path, database_root: Path) -> Path:
    config_path, database_root = config_path.resolve(), database_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if tuple(config["anchors"]) != ANCHORS or config["C"] != 0.1 or config["l1_ratio"] != 0.2:
        raise ValueError("V9-E0 比较条件已冻结；参数变化必须作为新实验")
    input_dir = database_root / "任务一预处理结果"
    results_root = database_root / "任务一实验结果"
    tables = load_tables(input_dir)
    reference = labeled_reference(tables.bags)
    split_path = results_root / "splits" / f"split_v1_seed{config['seed']}.json"
    split = load_or_create_split(reference, split_path, seed=config["seed"], folds=config["folds"],
                                 holdout_fraction=config["holdout_fraction"])
    audit = audit_split(reference, split)
    profile_ids = sorted(tables.profile.gpsno.astype(str).tolist())
    if len(reference) != 478 or len(profile_ids) != 500:
        raise ValueError("监督候选或提交车辆数量与数据审计不符")
    features, feature_sets = build_causal_landmarks(tables.daily, profile_ids, anchors=(*ANCHORS, 60))
    variants = list(feature_sets)
    primary = config["primary_variant"]
    if primary not in variants:
        raise ValueError("primary_variant 不存在")

    dev_ids = set(split["development"])
    dev_features = features[features.gpsno.isin(dev_ids) & features.anchor_day.ne(60)].copy()
    outcomes = make_outcomes(tables.daily, dev_features)
    label20 = outcomes[outcomes.anchor_day.eq(20)].set_index("gpsno").label
    frozen = reference.set_index("gpsno").label
    if not label20.eq(frozen.loc[label20.index]).all():
        raise ValueError("20→40 标签与冻结标签不一致")
    compressed = compress_outcomes(outcomes)
    files = file_manifest(input_dir)
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    resolved = {**config, "input_dir": str(input_dir), "results_root": str(results_root),
                "split_version": split["version"], "feature_sets": feature_sets}
    run_dir, manifest = create_run(REPO, results_root, config_path, resolved,
                                   version="V9E0", entrypoint="train_v9_e0.py")
    try:
        manifest.update({"model": "v9e0_causal_prior_elasticnet_ablation", "split_version": split["version"],
                         "data_fingerprint": fingerprint, "data_files": files,
                         "evaluation_version": "v9_vehicle_oof_stationary_hazard_fixed_threshold_0.5",
                         "dependencies": {name: importlib.metadata.version(name) for name in
                                          ("scikit-learn", "duckdb", "pandas", "numpy", "joblib")},
                         "locked_labels_used_in_fit_or_metrics": False})
        write_json(run_dir / "manifest.json", manifest)
        shutil.copy2(split_path, run_dir / "splits.json")
        audit.update({"landmark_days": list(ANCHORS), "day_end_boundary": "23:59:59",
                      "label_days_start": "t+1", "future_features_used": False,
                      "locked_holdout_scored": False, "feature_variants": variants})
        write_json(run_dir / "leakage_audit.json", audit)

        oof_parts = []
        for fold_text, validation in sorted(split["validation_folds"].items(), key=lambda item: int(item[0])):
            fold = int(fold_text)
            val_ids = set(validation)
            train_ids = dev_ids - val_ids
            train_features = dev_features[dev_features.gpsno.isin(train_ids)]
            val_features = dev_features[dev_features.gpsno.isin(val_ids)].copy()
            train_rows = compressed[compressed.gpsno.isin(train_ids)]
            prediction = val_features[["gpsno", "anchor_day"]].merge(
                outcomes, on=["gpsno", "anchor_day"], validate="one_to_one")
            prediction["fold"] = fold
            for variant, columns in feature_sets.items():
                model = HazardLogistic(columns, c=config["C"], seed=config["seed"] + fold).fit(
                    train_features, train_rows)
                hazard = model.daily_hazard(val_features)
                prediction[f"daily_hazard_{variant}"] = hazard
                prediction[f"probability_{variant}"] = horizon_probability(hazard, prediction.horizon_days)
                joblib.dump(model, run_dir / "models" / f"fold_{fold}_{variant}.joblib")
            oof_parts.append(prediction)

        oof = pd.concat(oof_parts, ignore_index=True).sort_values(["anchor_day", "gpsno"])
        if len(oof) != len(dev_ids) * len(ANCHORS) or oof.duplicated(["gpsno", "anchor_day"]).any():
            raise ValueError("OOF 未覆盖每辆开发车的每个锚点")
        oof["daily_hazard"] = oof[f"daily_hazard_{primary}"]
        oof["probability"] = oof[f"probability_{primary}"]
        write_parquet(run_dir / "predictions" / "oof_landmarks.parquet", oof)
        scores, comparisons = _score_variants(oof, variants, config["seed"])
        metrics = {
            "target": "day-end landmark next-event hazard; right censored at day 60",
            "primary_variant": primary,
            "development_oof": scores[primary]["20_to_40"],
            "by_variant_and_anchor": scores,
            "paired_comparisons_at_20_to_40": comparisons,
            "holdout_evaluated": False,
            "warning": "只验证观测期内早期锚点；不代表 Day60→未来40天的真实成绩。",
        }

        target = features[features.anchor_day.eq(60)].copy().sort_values("gpsno")
        matched_ids = set(reference.gpsno)
        route = target.gpsno.isin(matched_ids).to_numpy()
        has_behavior = ((target.exposure_trajectory_fraction.to_numpy() > 0)
                        | (target.exposure_imu_fraction.to_numpy() > 0))
        prior = float(outcomes[outcomes.anchor_day.eq(20)].label.mean())
        fallback_columns = [c for c in feature_sets["e0a_v3_base"] if not c.startswith("event_")]
        fallback = HazardLogistic(fallback_columns, c=config["C"], seed=config["seed"] + 999).fit(
            dev_features, compressed)
        fallback_probability = horizon_probability(fallback.daily_hazard(target), 40)
        joblib.dump(fallback, run_dir / "models" / "development_fallback.joblib")
        candidates = {}
        for variant, columns in feature_sets.items():
            model = HazardLogistic(columns, c=config["C"], seed=config["seed"] + 999).fit(
                dev_features, compressed)
            full_probability = horizon_probability(model.daily_hazard(target), 40)
            probability = np.where(route, full_probability,
                                   np.where(has_behavior, fallback_probability, prior))
            candidate = pd.DataFrame({
                "gpsno": target.gpsno.to_numpy(), "anchor_day": 60, "horizon_days": 40,
                "route": np.where(route, "full", np.where(has_behavior,
                    "fallback_no_matched_event_record", "prior_no_observed_behavior")),
                "probability": probability, "prediction_at_0_5": (probability >= 0.5).astype(int),
                "full_probability": full_probability, "fallback_probability": fallback_probability,
                "label_status": "future_unknown",
            })
            write_parquet(run_dir / "predictions" / f"candidate_500_{variant}.parquet", candidate)
            joblib.dump(model, run_dir / "models" / f"development_{variant}.joblib")
            candidates[variant] = candidate
        primary_candidate = candidates[primary]
        write_parquet(run_dir / "predictions" / "candidate_500.parquet", primary_candidate)
        primary_candidate[["gpsno", "probability", "prediction_at_0_5"]].to_csv(
            run_dir / "predictions" / "submission_candidate.csv", index=False, encoding="utf-8-sig")
        metrics["candidate_routing"] = {"full": int(route.sum()),
            "fallback_no_matched_event_record": int((~route & has_behavior).sum()),
            "prior_no_observed_behavior": int((~route & ~has_behavior).sum()),
            "development_20_to_40_prior": prior}
        write_json(run_dir / "metrics.json", metrics)
        lines = [f"# {manifest['run_id']}", "", "## 20→40 OOF", "",
                 "| Variant | AUC | Brier | Accuracy@0.5 |", "|---|---:|---:|---:|"]
        for variant in variants:
            score = scores[variant]["20_to_40"]
            lines.append(f"| {variant} | {score['roc_auc']:.6f} | {score['brier']:.6f} | {score['accuracy_at_0_5']:.6f} |")
        lines += ["", "96 台锁定测试车未评分。Day60 候选预测由 382 台开发车重训模型生成。", ""]
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
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/v9e0_causal_features.json")
    args = parser.parse_args()
    print(run(args.config, args.database_root))


if __name__ == "__main__":
    main()

