"""Read the existing task-one preprocessing contract without touching raw files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd


REFERENCE_ANCHOR = date(2026, 6, 20)
PREDICTION_ANCHOR = date(2026, 7, 30)
FIRST_DAY = date(2026, 6, 1)


@dataclass(frozen=True)
class TaskOneTables:
    daily: pd.DataFrame
    bags: pd.DataFrame
    profile: pd.DataFrame


def load_tables(directory: Path) -> TaskOneTables:
    directory = directory.resolve()
    names = ("daily_features.parquet", "bag_index.parquet", "profile.parquet")
    missing = [str(directory / name) for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "预处理尚未完成，缺少训练所需文件：" + ", ".join(missing)
            + "。请先按 README 完成 preprocess.py 的 assemble 阶段。"
        )
    con = duckdb.connect()
    try:
        daily = con.execute("SELECT * FROM read_parquet(?)", [str(directory / names[0])]).df()
        bags = con.execute("SELECT * FROM read_parquet(?)", [str(directory / names[1])]).df()
        profile = con.execute("SELECT gpsno, energy_type FROM read_parquet(?)", [str(directory / names[2])]).df()
    finally:
        con.close()

    for frame, time_col in ((daily, "day"), (bags, "anchor_date")):
        frame["gpsno"] = frame["gpsno"].astype(str)
        frame[time_col] = pd.to_datetime(frame[time_col], errors="raise").dt.date
    profile["gpsno"] = profile["gpsno"].astype(str)
    if daily.duplicated(["gpsno", "day"]).any():
        raise ValueError("daily_features 存在重复 gpsno × day")
    if bags.duplicated(["gpsno", "anchor_date"]).any():
        raise ValueError("bag_index 存在重复 gpsno × anchor_date")
    if profile.gpsno.duplicated().any() or len(profile) != 500:
        raise ValueError("画像应当恰好包含 500 个唯一 gpsno")
    if daily.day.min() < FIRST_DAY or daily.day.max() > PREDICTION_ANCHOR:
        raise ValueError("日特征超出比赛 06-01 至 07-30 时间范围")
    needed = {
        "has_event_feed", "event_11804_count", "event_11803_count", "event_total_count",
        "distance_km", "drive_hours", "trajectory_recorded_today", "has_trajectory",
    }
    missing_columns = sorted(needed - set(daily.columns))
    if missing_columns:
        raise ValueError(f"日特征缺少第一版所需列：{missing_columns}")
    needed_bags = {
        "label", "label_status", "anchor_date", "gpsno",
        "input_start", "input_end", "future_end",
    }
    if needed_bags - set(bags.columns):
        raise ValueError(f"bag_index 缺少列：{sorted(needed_bags - set(bags.columns))}")
    return TaskOneTables(daily=daily, bags=bags, profile=profile)


def labeled_reference(bags: pd.DataFrame) -> pd.DataFrame:
    reference = bags[bags.anchor_date == REFERENCE_ANCHOR].copy()
    reference = reference[reference.label.notna()].copy()
    if reference.empty or reference.gpsno.duplicated().any():
        raise ValueError("06-20 代理标签为空或存在重复车辆")
    if not set(reference.label.astype(int)).issubset({0, 1}):
        raise ValueError("代理标签必须是 0/1")
    if set(reference.label.astype(int)) != {0, 1}:
        raise ValueError("代理标签需要同时包含正负两类")
    reference["label"] = reference.label.astype(int)
    if not set(reference.label_status).issubset({"observed_positive", "provisional_negative"}):
        raise ValueError("代理标签含有未知状态，不能用于监督")
    expected_status = reference.label.map({1: "observed_positive", 0: "provisional_negative"})
    if not reference.label_status.eq(expected_status).all():
        raise ValueError("代理标签与 label_status 不一致")
    for column, expected in (
        ("input_start", FIRST_DAY),
        ("input_end", REFERENCE_ANCHOR),
        ("future_end", PREDICTION_ANCHOR),
    ):
        if not pd.to_datetime(reference[column]).dt.date.eq(expected).all():
            raise ValueError(f"06-20 代理样本的 {column} 时间边界不正确")
    return reference.sort_values("gpsno").reset_index(drop=True)


def prediction_index(bags: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    target = bags[bags.anchor_date == PREDICTION_ANCHOR].copy()
    if len(target) != len(profile) or set(target.gpsno) != set(profile.gpsno):
        raise ValueError("07-30 预测索引必须与画像中的 500 台车完全一致")
    if target.gpsno.duplicated().any() or target.label.notna().any():
        raise ValueError("07-30 预测索引重复或错误地包含已知标签")
    if not target.label_status.eq("target_unknown").all():
        raise ValueError("07-30 预测目标的 label_status 必须为 target_unknown")
    if not pd.to_datetime(target.input_start).dt.date.eq(date(2026, 7, 11)).all():
        raise ValueError("07-30 V1 预测输入必须从 07-11 开始")
    if not pd.to_datetime(target.input_end).dt.date.eq(PREDICTION_ANCHOR).all():
        raise ValueError("07-30 预测输入结束时间不正确")
    return target.sort_values("gpsno").reset_index(drop=True)
