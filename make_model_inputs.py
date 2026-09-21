"""Build leakage-safe 20-day tensors and vehicle-level cross-validation folds.

This consumes preprocess.py's daily_features.parquet and bag_index.parquet.
Only the training vehicles' June 1-14 observations fit numeric transforms.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


REFERENCE_ANCHOR = date(2026, 6, 20)
EARLIEST_ANCHOR = date(2026, 6, 14)
SUBMISSION_ANCHOR = date(2026, 7, 30)
FIRST_DAY = date(2026, 6, 1)
LOOKBACK = 20


def resolve_device(requested: str) -> str:
    """Select optional CUDA only for the small tensor transform, not DuckDB scans."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        if requested == "auto":
            return "cpu"
        raise RuntimeError("CUDA was requested; install a CUDA-enabled PyTorch build first") from exc
    if torch.cuda.is_available():
        return "cuda"
    if requested == "auto":
        return "cpu"
    raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")


def is_count_feature(name: str) -> bool:
    return (
        name.endswith(("_count", "_episodes", "_rows", "_samples", "_windows"))
        or name in {"trip_starts", "trajectory_rows", "moving_rows", "long_gap_rows"}
    )


def load_tables(directory: Path):
    required = ("daily_features.parquet", "bag_index.parquet", "profile.parquet")
    for name in required:
        if not (directory / name).exists():
            raise FileNotFoundError(directory / name)
    con = duckdb.connect()
    daily = con.execute("SELECT * FROM read_parquet(?) ORDER BY gpsno,day", [str(directory / required[0])]).df()
    bags = con.execute("SELECT * FROM read_parquet(?) ORDER BY gpsno,anchor_date", [str(directory / required[1])]).df()
    profile = con.execute("SELECT gpsno,energy_type FROM read_parquet(?) ORDER BY gpsno", [str(directory / required[2])]).df()
    con.close()
    daily["day"] = pd.to_datetime(daily["day"]).dt.date
    bags["anchor_date"] = pd.to_datetime(bags["anchor_date"]).dt.date
    daily["gpsno"] = daily["gpsno"].astype(str)
    bags["gpsno"] = bags["gpsno"].astype(str)
    profile["gpsno"] = profile["gpsno"].astype(str)
    numeric = [
        name for name in daily.columns
        if name not in ("gpsno", "day", "energy_type", "gps_distance_check_km")
        and (pd.api.types.is_numeric_dtype(daily[name]) or pd.api.types.is_bool_dtype(daily[name]))
    ]
    if not numeric:
        raise ValueError("No numeric day features found")
    return daily, bags, profile, numeric


def assign_folds(bags: pd.DataFrame, folds: int, seed: int) -> list[list[str]]:
    reference = bags[bags.anchor_date == REFERENCE_ANCHOR]
    reference = reference[reference.label.notna()]
    if reference.gpsno.duplicated().any():
        raise ValueError("Multiple June 20 labels for a vehicle")
    classes = [reference[reference.label == label].gpsno.to_list() for label in (0, 1)]
    if min(map(len, classes)) < folds:
        raise ValueError(f"Need at least {folds} vehicles in each label class")
    rng = np.random.default_rng(seed)
    result: list[list[str]] = [[] for _ in range(folds)]
    for ids in classes:
        rng.shuffle(ids)
        for i, gpsno in enumerate(ids):
            result[i % folds].append(gpsno)
    return [sorted(x) for x in result]


def build_scaler(daily: pd.DataFrame, features: list[str], train_ids: set[str]) -> dict[str, object]:
    fit = daily[daily.gpsno.isin(train_ids) & (daily.day <= EARLIEST_ANCHOR)]
    if fit.empty:
        raise ValueError("No pre-anchor training days for scaler")
    parameters: dict[str, object] = {"fit_end": str(EARLIEST_ANCHOR), "features": features, "columns": {}}
    for name in features:
        values = pd.to_numeric(fit[name], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        values[~np.isfinite(values)] = np.nan
        valid = values[np.isfinite(values)]
        boolean = pd.api.types.is_bool_dtype(daily[name])
        log_transform = bool(is_count_feature(name) and not boolean and len(valid) and np.min(valid) >= 0)
        if log_transform:
            valid = np.log1p(valid)
        if boolean:
            lo, hi, median, scale = 0.0, 1.0, 0.0, 1.0
        elif len(valid):
            lo, hi = map(float, np.quantile(valid, [0.01, 0.99]))
            clipped = np.clip(valid, lo, hi)
            median = float(np.median(clipped))
            q25, q75 = np.quantile(clipped, [0.25, 0.75])
            scale = float(q75 - q25)
            if scale < 1e-6:
                scale = 1.0
        else:
            lo, hi, median, scale = 0.0, 0.0, 0.0, 1.0
        parameters["columns"][name] = {
            "clip_low": lo, "clip_high": hi, "median": median, "scale": scale,
            "log1p": log_transform, "boolean": boolean, "observed_fit_values": int(len(valid)),
        }
    return parameters


def lookup_daily(daily: pd.DataFrame, features: list[str]) -> dict[tuple[str, date], np.ndarray]:
    result = {}
    for row in daily.itertuples(index=False):
        raw = row._asdict()
        key = (str(raw["gpsno"]), raw["day"])
        result[key] = np.array([
            float(raw[name]) if pd.notna(raw[name]) else np.nan for name in features
        ], dtype=np.float64)
    if len(result) != len(daily):
        raise ValueError("Duplicate gpsno × day in daily features")
    return result


def transform_values(raw: np.ndarray, mask: np.ndarray, features: list[str], scaler: dict[str, object], device: str) -> np.ndarray:
    """Apply the same fold-fitted transform on CPU or CUDA; missing cells stay zero."""
    parameters = [scaler["columns"][name] for name in features]
    low = np.asarray([item["clip_low"] for item in parameters], dtype=np.float32)[None, None, :]
    high = np.asarray([item["clip_high"] for item in parameters], dtype=np.float32)[None, None, :]
    median = np.asarray([item["median"] for item in parameters], dtype=np.float32)[None, None, :]
    scale = np.asarray([item["scale"] for item in parameters], dtype=np.float32)[None, None, :]
    log_flags = np.asarray([item["log1p"] for item in parameters], dtype=np.bool_)[None, None, :]
    safe = np.where(mask.astype(bool), raw, 0).astype(np.float32, copy=False)
    if device == "cuda":
        import torch

        with torch.no_grad():
            values = torch.as_tensor(safe, device="cuda")
            flags = torch.as_tensor(log_flags, device="cuda")
            values = torch.where(flags, torch.log1p(torch.clamp_min(values, 0)), values)
            values = torch.maximum(torch.minimum(values, torch.as_tensor(high, device="cuda")), torch.as_tensor(low, device="cuda"))
            values = (values - torch.as_tensor(median, device="cuda")) / torch.as_tensor(scale, device="cuda")
            values = torch.where(torch.as_tensor(mask.astype(bool), device="cuda"), values, 0)
            return values.cpu().numpy()
    safe = np.where(log_flags, np.log1p(np.maximum(safe, 0)), safe)
    safe = np.clip(safe, low, high)
    return np.where(mask.astype(bool), (safe - median) / scale, 0).astype(np.float32, copy=False)


def make_tensor(
    selected: pd.DataFrame,
    lookup: dict[tuple[str, date], np.ndarray],
    features: list[str],
    scaler: dict[str, object],
    energy: dict[str, str],
    categories: list[str],
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    n, width = len(selected), len(features)
    raw_values = np.full((n, LOOKBACK, width), np.nan, dtype=np.float32)
    value_mask = np.zeros((n, LOOKBACK, width), dtype=np.uint8)
    day_mask = np.zeros((n, LOOKBACK), dtype=np.uint8)
    static = np.zeros((n, len(categories) + 1), dtype=np.uint8)
    ids = np.empty(n, dtype="U32")
    anchors = np.empty(n, dtype="U10")
    labels = np.full(n, -1, dtype=np.int8)
    statuses = np.empty(n, dtype="U32")
    for index, bag in enumerate(selected.itertuples(index=False)):
        gpsno = str(bag.gpsno)
        anchor = bag.anchor_date
        ids[index], anchors[index], statuses[index] = gpsno, anchor.isoformat(), str(bag.label_status)
        if pd.notna(bag.label):
            labels[index] = int(bag.label)
        category = energy.get(gpsno)
        static[index, categories.index(category) if category in categories else -1] = 1
        start = anchor - timedelta(days=LOOKBACK - 1)
        for offset in range(LOOKBACK):
            day = start + timedelta(days=offset)
            if day < FIRST_DAY or day > anchor:
                continue
            day_mask[index, offset] = 1
            raw = lookup.get((gpsno, day))
            if raw is None:
                continue
            finite = np.isfinite(raw)
            value_mask[index, offset] = finite.astype(np.uint8)
            raw_values[index, offset] = raw
    x = transform_values(raw_values, value_mask, features, scaler, device)
    return {
        "x": x, "value_mask": value_mask, "day_mask": day_mask, "static_energy": static,
        "gpsno": ids, "anchor_date": anchors, "label": labels, "label_status": statuses,
    }


def write_pack(path: Path, value: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **value)


def build(input_dir: Path, output_dir: Path | None, n_folds: int, seed: int, device: str = "cpu") -> dict[str, object]:
    input_dir = input_dir.resolve()
    output_dir = (output_dir if output_dir is not None else input_dir / "model_inputs").resolve()
    if input_dir not in output_dir.parents:
        raise ValueError("Model input output must be a subdirectory of the processed database directory")
    if device not in ("cpu", "auto", "cuda"):
        raise ValueError("Device must be cpu, auto or cuda")
    active_device = resolve_device(device)
    daily, bags, profile, features = load_tables(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_val_ids = assign_folds(bags, n_folds, seed)
    eligible = set().union(*(set(x) for x in fold_val_ids))
    lookup = lookup_daily(daily, features)
    energy = dict(zip(profile.gpsno, profile.energy_type))
    categories = sorted(profile[profile.gpsno.isin(eligible)].energy_type.dropna().unique().tolist())
    submission = bags[bags.anchor_date == SUBMISSION_ANCHOR].copy()
    if len(submission) != len(profile) or submission.gpsno.nunique() != len(profile):
        raise AssertionError("Submission index must contain exactly one bag per profile vehicle")
    summary: dict[str, object] = {
        "folds": n_folds, "seed": seed, "eligible_vehicles": len(eligible),
        "device_requested": device, "device_used": active_device,
        "features": features, "energy_categories": categories, "validation": [],
    }
    for fold_index, val_list in enumerate(fold_val_ids):
        val_ids = set(val_list)
        train_ids = eligible - val_ids
        scaler = build_scaler(daily, features, train_ids)
        train = bags[bags.gpsno.isin(train_ids) & (bags.anchor_date <= REFERENCE_ANCHOR)]
        val = bags[bags.gpsno.isin(val_ids) & (bags.anchor_date == REFERENCE_ANCHOR)]
        if train.label.isna().any() or val.label.isna().any():
            raise AssertionError("Training or validation bag has unknown label")
        write_pack(output_dir / f"fold_{fold_index}_train.npz", make_tensor(train, lookup, features, scaler, energy, categories, active_device))
        write_pack(output_dir / f"fold_{fold_index}_val.npz", make_tensor(val, lookup, features, scaler, energy, categories, active_device))
        (output_dir / f"fold_{fold_index}_scaler.json").write_text(
            json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["validation"].append({
            "fold": fold_index, "vehicles": val_list, "positive": int((val.label == 1).sum()),
            "negative": int((val.label == 0).sum()), "train_bags": len(train),
        })
    final_scaler = build_scaler(daily, features, eligible)
    final_train = bags[bags.gpsno.isin(eligible) & (bags.anchor_date <= REFERENCE_ANCHOR)]
    write_pack(output_dir / "final_train.npz", make_tensor(final_train, lookup, features, final_scaler, energy, categories, active_device))
    write_pack(output_dir / "final_submission.npz", make_tensor(submission, lookup, features, final_scaler, energy, categories, active_device))
    (output_dir / "final_scaler.json").write_text(
        json.dumps(final_scaler, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "splits.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"eligible_vehicles": len(eligible), "features": len(features), "folds": n_folds,
            "final_train_bags": len(final_train), "submission_bags": len(submission), "device_used": active_device,
            "output": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="preprocess.py output directory")
    parser.add_argument("--output", type=Path, help="NPZ output inside --input; defaults to --input/model_inputs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="auto",
                        help="Optional CUDA tensor transform; auto falls back to CPU when unavailable")
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    print(json.dumps(build(args.input, args.output, args.folds, args.seed, args.device), ensure_ascii=False))


if __name__ == "__main__":
    main()
