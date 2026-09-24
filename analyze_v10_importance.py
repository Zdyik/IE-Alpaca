"""Measure V10 event importance by masking each event signal in outer-fold OOF models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from ie_alpaca.data.task_one import load_tables  # noqa: E402
from ie_alpaca.features.daily_v10 import (  # noqa: E402
    CONTEXT_COLUMNS, EVENT_FEATURE_NAMES, NodeDayScaler, build_day_table,
)
from ie_alpaca.features.landmark_v3 import ANCHORS  # noqa: E402
from ie_alpaca.features.landmark_v4 import EVENT_CODES, EVENT_NAMES  # noqa: E402
from ie_alpaca.models.risk_chain_v10 import RiskChainNetV10  # noqa: E402
from ie_alpaca.training.event_v4 import resolve_device  # noqa: E402
from ie_alpaca.training.landmark_v3 import horizon_probability, make_outcomes  # noqa: E402
from ie_alpaca.training.risk_chain_v10 import predict_anchors, vehicle_pack  # noqa: E402


def _load_network(path: Path, device: torch.device) -> tuple[RiskChainNetV10, NodeDayScaler, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config, mode = checkpoint["model_config"], checkpoint["mode"]
    net = RiskChainNetV10(
        context_width=len(CONTEXT_COLUMNS), mode=mode, width=int(config["width"]),
        event_embedding=int(config["event_embedding"]), role_embedding=int(config["role_embedding"]),
        stage_embedding=int(config["stage_embedding"]), dropout=float(config["dropout"]),
        modality_dropout=float(config["modality_dropout"]), base_daily_hazard=.01,
    ).to(device)
    net.load_state_dict(checkpoint["state_dict"])
    return net.eval(), NodeDayScaler.from_description(checkpoint["scaler"]), checkpoint


def analyze(run_dir: Path, database_root: Path, device_name: str = "auto") -> tuple[Path, Path]:
    run_dir, database_root = run_dir.resolve(), database_root.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("model") != "torch_v10_risk_chain_tcn":
        raise ValueError("importance requires a completed V10 run")
    mode = resolved["primary_variant"]
    split = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    input_dir = database_root / "任务一预处理结果"
    tables = load_tables(input_dir)
    profile_ids = sorted(tables.profile.gpsno.astype(str))
    day_table = build_day_table(tables.daily, profile_ids)
    dev_ids = set(split["development"])
    landmarks = pd.MultiIndex.from_product([sorted(dev_ids), ANCHORS],
                                           names=["gpsno", "anchor_day"]).to_frame(index=False)
    outcomes = make_outcomes(tables.daily, landmarks)
    device = resolve_device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(int(resolved["model"].get("cpu_threads", 4)))
    rows = []
    for fold_text, val_list in sorted(split["validation_folds"].items(), key=lambda item: int(item[0])):
        fold, val_ids = int(fold_text), set(val_list)
        net, scaler, _ = _load_network(run_dir / "models" / f"fold_{fold}_{mode}.pt", device)
        pack = vehicle_pack(day_table, outcomes, val_ids, scaler)
        base_q, _ = predict_anchors(net, pack, device, int(resolved["model"]["batch_size"]))
        anchor_index = ANCHORS.index(20)
        labels = pack["label"][:, anchor_index].astype(int)
        base = horizon_probability(base_q[:, anchor_index], 40)
        for event_index, code in enumerate(EVENT_CODES):
            masked = {**pack, "events": pack["events"].copy()}
            # Remove counts, episodes, rates and occurrence while retaining the
            # day's exposure-observed flags and night context for fair masking.
            masked["events"][:, :, event_index, :5] = 0
            q, _ = predict_anchors(net, masked, device, int(resolved["model"]["batch_size"]))
            probability = horizon_probability(q[:, anchor_index], 40)
            rows.extend({"gpsno": gpsno, "fold": fold, "label": int(label),
                         "event_code": code, "event_name": EVENT_NAMES[code],
                         "baseline_probability": float(before),
                         "masked_probability": float(after)}
                        for gpsno, label, before, after in zip(pack["gpsno"], labels, base, probability))
        print(f"V10 importance fold {fold} complete", flush=True)
    detail = pd.DataFrame(rows)
    summary_rows = []
    for (code, name), part in detail.groupby(["event_code", "event_name"], sort=False):
        baseline_auc = roc_auc_score(part.label, part.baseline_probability)
        masked_auc = roc_auc_score(part.label, part.masked_probability)
        summary_rows.append({"event_code": int(code), "event_name": name,
                             "baseline_auc": float(baseline_auc), "masked_auc": float(masked_auc),
                             "auc_drop_when_masked": float(baseline_auc - masked_auc),
                             "mean_probability_change": float(
                                 (part.baseline_probability - part.masked_probability).mean()),
                             "mean_absolute_probability_change": float(
                                 (part.baseline_probability - part.masked_probability).abs().mean())})
    summary = pd.DataFrame(summary_rows).sort_values(
        ["auc_drop_when_masked", "mean_absolute_probability_change"], ascending=False)
    detail_path, summary_path = run_dir / "event_ablation_oof.csv", run_dir / "event_importance.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    (run_dir / "source" / Path(__file__).name).write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    return detail_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--database-root", type=Path,
                        default=Path(r"E:\Databases\2026 清华IE亮剑-算法赛道赛题"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    for path in analyze(args.run_dir, args.database_root, args.device):
        print(path)


if __name__ == "__main__":
    main()
