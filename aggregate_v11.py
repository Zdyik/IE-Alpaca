"""Build an immutable three-seed V11 masked ensemble from completed strict-OOF runs."""

from __future__ import annotations

import argparse, json, shutil, sys, traceback
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from ie_alpaca.evaluation.metrics import auc_interval, score_binary
from ie_alpaca.tracking.run_store import create_run, update_leaderboard, verify_source, write_json, write_parquet


def read_parquet(path: Path) -> pd.DataFrame:
    with duckdb.connect() as con: return con.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()


def paired_auc(current: pd.DataFrame, prior: pd.DataFrame, seed: int) -> dict:
    pair = current[["gpsno", "label", "probability"]].merge(prior[["gpsno", "label", "probability"]], on="gpsno",
                                                               suffixes=("_new", "_prior"), validate="one_to_one")
    y = pair.label_new.to_numpy(int); a = pair.probability_new.to_numpy(); b = pair.probability_prior.to_numpy()
    rng = np.random.default_rng(seed); values = []
    for _ in range(2000):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) == 2: values.append(roc_auc_score(y[index], a[index]) - roc_auc_score(y[index], b[index]))
    return {"vehicles": len(y), "new_auc": float(roc_auc_score(y, a)), "prior_auc": float(roc_auc_score(y, b)),
            "delta_auc": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "delta_auc_95pct_paired_bootstrap": np.quantile(values, [.025, .975]).tolist()}


def run(config_path: Path, database_root: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf8")); results = database_root / "任务一实验结果"
    members = [results / "runs" / run_id for run_id in config["member_run_ids"]]
    for path in members:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf8"))
        if manifest.get("status") != "completed" or manifest.get("locked_labels_used_in_fit_or_metrics"):
            raise ValueError(f"invalid ensemble member: {path.name}")
    resolved = {**config, "results_root": str(results), "aggregation": "arithmetic mean of vehicle probabilities"}
    run_dir, manifest = create_run(REPO, results, config_path, resolved, version=config["run_prefix"], entrypoint="aggregate_v11.py")
    try:
        manifest.update({"model": "v11_masked_three_seed_probability_ensemble", "split_version": "split_v1_seed2026",
                         "data_fingerprint": json.loads((members[0] / "manifest.json").read_text(encoding="utf8"))["data_fingerprint"],
                         "evaluation_version": "v11_three_seed_strict_inductive_oof_ensemble_0.5",
                         "locked_labels_used_in_fit_or_metrics": False, "member_run_ids": config["member_run_ids"]})
        write_json(run_dir / "manifest.json", manifest)
        shutil.copy2(members[0] / "splits.json", run_dir / "splits.json")
        frames = []
        for index, member in enumerate(members):
            frame = read_parquet(member / "predictions" / "oof_landmarks.parquet")
            frames.append(frame[["gpsno", "anchor_day", "horizon_days", "label", "first_event_day", "fold",
                                 "daily_hazard_masked", "probability_masked"]].rename(columns={
                                     "daily_hazard_masked": f"daily_hazard_seed_{index}",
                                     "probability_masked": f"probability_seed_{index}"}))
        keys = ["gpsno", "anchor_day", "horizon_days", "label", "first_event_day", "fold"]
        oof = frames[0]
        for frame in frames[1:]: oof = oof.merge(frame, on=keys, validate="one_to_one")
        oof["daily_hazard"] = oof[[f"daily_hazard_seed_{i}" for i in range(len(frames))]].mean(axis=1)
        oof["probability"] = oof[[f"probability_seed_{i}" for i in range(len(frames))]].mean(axis=1)
        write_parquet(run_dir / "predictions" / "oof_landmarks.parquet", oof)
        candidates = []
        for index, member in enumerate(members):
            frame = read_parquet(member / "predictions" / "candidate_500.parquet")
            candidates.append(frame[["gpsno", "anchor_day", "horizon_days", "route", "label_status", "probability"]].rename(
                columns={"probability": f"probability_seed_{index}"}))
        candidate = candidates[0]
        candidate_keys = ["gpsno", "anchor_day", "horizon_days", "route", "label_status"]
        for frame in candidates[1:]: candidate = candidate.merge(frame, on=candidate_keys, validate="one_to_one")
        candidate["probability"] = candidate[[f"probability_seed_{i}" for i in range(len(candidates))]].mean(axis=1)
        candidate["prediction_at_0_5"] = (candidate.probability >= .5).astype(int)
        write_parquet(run_dir / "predictions" / "candidate_500.parquet", candidate)
        candidate[["gpsno", "probability", "prediction_at_0_5"]].to_csv(run_dir / "predictions" / "submission_candidate.csv",
                                                                           index=False, encoding="utf-8-sig")
        metrics = {"target": "right-censored stationary daily first-event hazard", "primary_method": "masked_three_seed_mean",
                   "holdout_evaluated": False, "by_anchor": {}, "member_results": []}
        for member in members:
            value = json.loads((member / "metrics.json").read_text(encoding="utf8"))
            masked = value["by_method_and_anchor"]["masked"]["20_to_40"]
            scratch = value["by_method_and_anchor"]["scratch"]["20_to_40"]
            metrics["member_results"].append({"run_id": member.name, "masked_auc": masked["roc_auc"],
                                               "scratch_auc": scratch["roc_auc"], "delta_auc": masked["roc_auc"] - scratch["roc_auc"],
                                               "masked_brier": masked["brier"]})
        for anchor, part in oof.groupby("anchor_day"):
            key = f"{anchor}_to_{int(part.horizon_days.iloc[0])}"; score = score_binary(part.label, part.probability)
            score["auc_95pct_bootstrap"] = auc_interval(part.label, part.probability, config["seed"] + int(anchor))
            metrics["by_anchor"][key] = score
        metrics["development_oof"] = metrics["by_anchor"]["20_to_40"]
        values = np.asarray([x["masked_auc"] for x in metrics["member_results"]]); deltas = np.asarray([x["delta_auc"] for x in metrics["member_results"]])
        metrics["three_seed_summary"] = {"masked_auc_mean": float(values.mean()), "masked_auc_sample_std": float(values.std(ddof=1)),
                                          "delta_vs_scratch_mean": float(deltas.mean()), "delta_vs_scratch_sample_std": float(deltas.std(ddof=1))}
        for name, run_id in (("v10", config["v10_reference_run_id"]), ("v3", config["v3_reference_run_id"])):
            ref = read_parquet(results / "runs" / run_id / "predictions" / "oof_landmarks.parquet")
            ref = ref[ref.anchor_day.eq(20)][["gpsno", "label", "probability"]]
            current = oof[oof.anchor_day.eq(20)][["gpsno", "label", "probability"]]
            metrics[f"paired_ensemble_vs_{name}"] = paired_auc(current, ref, config["seed"] + len(name))
        metrics["candidate_rows"] = len(candidate); metrics["warning"] = "Day60 future labels are unavailable; 96 locked labels were not evaluated."
        write_json(run_dir / "metrics.json", metrics)
        score = metrics["development_oof"]
        lines = [f"# {manifest['run_id']}", "", "Three-seed masked probability ensemble.", "",
                 f"- 20→40 OOF AUC: {score['roc_auc']:.6f}", f"- Brier: {score['brier']:.6f}",
                 f"- Seed AUC mean ± SD: {values.mean():.6f} ± {values.std(ddof=1):.6f}",
                 f"- Mean ΔAUC vs same-seed Scratch: {deltas.mean():+.6f}", "", "96 locked vehicles were not evaluated.", ""]
        (run_dir / "result_summary.md").write_text("\n".join(lines), encoding="utf8")
        verify_source(REPO, manifest)
        manifest.update({"status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
                         "development_vehicles": 382, "holdout_vehicles": 96})
        write_json(run_dir / "manifest.json", manifest); update_leaderboard(results)
    except Exception:
        manifest.update({"status": "failed", "failed_utc": datetime.now(timezone.utc).isoformat(), "error": traceback.format_exc()})
        write_json(run_dir / "manifest.json", manifest); raise
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--database-root", type=Path, default=Path(r"E:\Databases\2026 清华IE亮剑-算法赛道赛题"))
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/v11_masked_ensemble.json")
    args = parser.parse_args(); print(run(args.config.resolve(), args.database_root.resolve()))

