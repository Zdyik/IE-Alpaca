"""Validate task-one processed tables and 20-day tensor packs.

Writes validation_report.json beside the processed data. This does not train a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np


REQUIRED_TABLES = (
    "profile.parquet", "events_clean.parquet", "accidents.parquet", "event_daily.parquet",
    "trajectory_daily.parquet", "imu_daily.parquet", "coverage.parquet",
    "daily_features.parquet", "bag_index.parquet",
)


def quoted(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def validate(processed: Path) -> dict:
    processed = processed.resolve()
    missing = [name for name in REQUIRED_TABLES if not (processed / name).is_file()]
    if missing:
        raise FileNotFoundError(f"预处理产物不完整：{missing}")
    model_dir = processed / "model_inputs"
    split_path = model_dir / "splits.json"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    con = duckdb.connect()
    try:
        table = lambda name: f"read_parquet({quoted(processed / name)})"
        profile = con.execute(f"SELECT COUNT(*),COUNT(DISTINCT gpsno) FROM {table('profile.parquet')}").fetchone()
        daily = con.execute(f"""
            SELECT COUNT(*),COUNT(DISTINCT gpsno),MIN(day),MAX(day),
                   COUNT(*)-COUNT(DISTINCT (gpsno,day)),
                   COUNT(*) FILTER (WHERE distance_km<0 OR drive_hours<0)
            FROM {table('daily_features.parquet')}
        """).fetchone()
        days_per_vehicle = con.execute(f"""
            SELECT MIN(n),MAX(n) FROM (
                SELECT gpsno,COUNT(*) AS n FROM {table('daily_features.parquet')} GROUP BY gpsno
            )
        """).fetchone()
        bags = con.execute(f"""
            SELECT COUNT(*),COUNT(DISTINCT gpsno),COUNT(DISTINCT anchor_date),
                   COUNT(*)-COUNT(DISTINCT (gpsno,anchor_date))
            FROM {table('bag_index.parquet')}
        """).fetchone()
        statuses = con.execute(f"""
            SELECT label_status,COUNT(*) FROM {table('bag_index.parquet')}
            WHERE anchor_date=DATE '2026-06-20' GROUP BY label_status ORDER BY label_status
        """).fetchall()
        target = con.execute(f"""
            SELECT COUNT(*),COUNT(DISTINCT gpsno),COUNT(label)
            FROM {table('bag_index.parquet')} WHERE anchor_date=DATE '2026-07-30'
        """).fetchone()
        label_mismatches = con.execute(f"""
            WITH actual AS (
                SELECT b.gpsno,b.anchor_date,b.label,
                       COUNT(a.event_ts) AS future_accidents
                FROM {table('bag_index.parquet')} b
                LEFT JOIN {table('accidents.parquet')} a
                    ON a.gpsno=b.gpsno
                    AND a.event_ts>=CAST(b.anchor_date+INTERVAL 1 DAY AS TIMESTAMP)
                    AND a.event_ts<CAST(b.anchor_date+INTERVAL 41 DAY AS TIMESTAMP)
                WHERE b.anchor_date<=DATE '2026-06-20' AND b.label IS NOT NULL
                GROUP BY b.gpsno,b.anchor_date,b.label
            )
            SELECT COUNT(*) FROM actual WHERE label<>CAST(future_accidents>0 AS INTEGER)
        """).fetchone()[0]
        late_events = con.execute(f"""
            SELECT COUNT(*) FROM {table('events_clean.parquet')}
            WHERE event_ts>=TIMESTAMP '2026-07-31'
        """).fetchone()[0]
        late_trajectory = con.execute(f"""
            SELECT COUNT(*) FROM {table('trajectory_daily.parquet')}
            WHERE day>DATE '2026-07-30'
        """).fetchone()[0]
        late_imu = con.execute(f"""
            SELECT COUNT(*) FROM {table('imu_daily.parquet')}
            WHERE day>DATE '2026-07-30'
        """).fetchone()[0]
    finally:
        con.close()
    if profile != (500, 500):
        raise ValueError(f"画像车辆数不正确：{profile}")
    if daily != (30000, 500, date(2026, 6, 1), date(2026, 7, 30), 0, 0):
        raise ValueError(f"车辆日特征不满足 500×60 与时间/唯一性契约：{daily}")
    if days_per_vehicle != (60, 60) or bags != (4000, 500, 8, 0) or target != (500, 500, 0):
        raise ValueError(f"日历或标签索引异常：{days_per_vehicle}, {bags}, {target}")
    if label_mismatches or late_events or late_trajectory or late_imu:
        raise ValueError("事故标签与事故表不一致，或规则终点后的数据进入已清洗表")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    folds = int(split["folds"])
    eligible = int(split["eligible_vehicles"])
    features = split["features"]
    validation_ids: set[str] = set()
    fold_summary = []
    for fold in range(folds):
        with np.load(model_dir / f"fold_{fold}_train.npz", allow_pickle=False) as train, np.load(
            model_dir / f"fold_{fold}_val.npz", allow_pickle=False
        ) as val:
            train_ids = set(map(str, train["gpsno"].tolist()))
            val_ids = set(map(str, val["gpsno"].tolist()))
            if train_ids & val_ids or validation_ids & val_ids:
                raise ValueError(f"折 {fold} 发生车辆交叉")
            if train["x"].shape[1:] != (20, len(features)) or val["x"].shape[1:] != (20, len(features)):
                raise ValueError(f"折 {fold} 的张量形状异常")
            if not np.isin(train["label"], [0, 1]).all() or not np.isin(val["label"], [0, 1]).all():
                raise ValueError(f"折 {fold} 包含未知标签")
            validation_ids.update(val_ids)
            fold_summary.append({
                "fold": fold, "train_vehicles": len(train_ids), "validation_vehicles": len(val_ids),
                "train_bags": len(train["label"]), "validation_bags": len(val["label"]),
                "validation_positive": int(np.sum(val["label"] == 1)),
            })
    if len(validation_ids) != eligible:
        raise ValueError("验证折未覆盖全部合格车辆")
    with np.load(model_dir / "final_train.npz", allow_pickle=False) as final_train, np.load(
        model_dir / "final_submission.npz", allow_pickle=False
    ) as final_submission:
        if len(final_submission["gpsno"]) != 500 or len(set(final_submission["gpsno"].tolist())) != 500:
            raise ValueError("最终预测包未覆盖 500 台唯一车辆")
        if not np.all(final_submission["label"] == -1):
            raise ValueError("最终预测包意外含有已知标签")
        if set(final_train["gpsno"].tolist()) != validation_ids:
            raise ValueError("最终训练包的车辆与外折车辆不一致")
        final_train_bags = len(final_train["label"])
    report = {
        "status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "processed_path": str(processed), "profile_vehicles": 500,
        "daily_rows": 30000, "days_per_vehicle": 60, "bag_rows": 4000,
        "june20_label_status": dict(statuses), "target_vehicles": 500,
        "label_mismatches": 0, "after_rule_end_clean_rows": 0,
        "model_input_features": len(features), "eligible_vehicles": eligible,
        "final_train_bags": final_train_bags, "folds": fold_summary,
        "legacy_tensor_split_note": "model_inputs/splits.json 是既有 5 折张量划分；V1 CatBoost 另建开发/锁定组划分。",
        "manifest_sha256": hashlib.sha256((processed / "manifest.json").read_bytes()).hexdigest(),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    processed = args.input.resolve()
    report = validate(processed)
    destination = processed / "validation_report.json"
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
