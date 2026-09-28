"""Prepare the competition's four local sources for task-one modeling.

Raw files are read only. Run ``python preprocess.py --help`` for the CLI.
The output is a set of Parquet feature/label tables plus JSON audit reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import duckdb
from openpyxl import load_workbook


START = date(2026, 6, 1)
END = date(2026, 7, 30)
PROXY_START = date(2026, 6, 14)
PROXY_END = date(2026, 6, 20)
ACCIDENT_CODES = (11803, 11804)
EVENT_CODES = (
    11401, 11402, 11403, 11405, 11406,
    30000, 30002, 30003, 30005, 30017,
    41001, 41002, 41003, 41004, 41005, 41006, 41009, 41021, 41023, 41029,
    60292, 60294, 11803, 11804,
)
EVENT_COLUMNS = ("gpsno", "lat", "lng", "event_type", "event_name", "speed", "start_time")
TRAJECTORY_COLUMNS = ("gpsno", "distance", "run_time", "trigger_time", "speed", "course", "lat", "lng")
IMU_COLUMNS = (
    "gpsno", "imei", "data_time", "data_date", "ems_speed", "gps_speed",
    "accel_x_raw", "accel_y_raw", "accel_z_raw", "gyro_x_raw", "gyro_y_raw", "gyro_z_raw",
)
OUTPUT_DIRNAME = "任务一预处理结果"


def quoted(value: str | Path) -> str:
    """A SQL string literal for trusted local paths and constants."""
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def csv_scan(paths: Iterable[Path], columns: tuple[str, ...], compression: str = "auto") -> str:
    files = list(paths)
    if not files:
        raise FileNotFoundError("No source files found")
    path_arg = "[" + ",".join(quoted(p) for p in files) + "]"
    names = "{" + ",".join(f"{quoted(c)}:'VARCHAR'" for c in columns) + "}"
    return (
        f"read_csv({path_arg}, delim='\\t', header=false, columns={names}, "
        f"nullstr={quoted(chr(92) + 'N')}, compression={quoted(compression)}, "
        "auto_detect=false, filename=true)"
    )


def parquet_scan(paths: Iterable[Path]) -> str:
    files = list(paths)
    if not files:
        raise FileNotFoundError("No Parquet parts found")
    return "read_parquet([" + ",".join(quoted(p) for p in files) + "])"


def copy_parquet(con: duckdb.DuckDBPyConnection, query: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        con.execute(f"COPY ({query}) TO {quoted(partial)} (FORMAT PARQUET, COMPRESSION ZSTD)")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def up_to_date(destination: Path, inputs: Iterable[Path]) -> bool:
    """Resume only completed parts newer than every input they depend on."""
    if not destination.exists() or destination.stat().st_size == 0:
        return False
    return all(destination.stat().st_mtime_ns >= path.stat().st_mtime_ns for path in inputs)


def scalar(con: duckdb.DuckDBPyConnection, query: str) -> int:
    return int(con.execute(query).fetchone()[0])


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def discover(input_dir: Path) -> dict[str, object]:
    profile = sorted(input_dir.glob("1.*.xlsx"))
    event_zip = sorted(input_dir.glob("2.*.zip"))
    imu_dirs = sorted(p for p in input_dir.glob("3.*") if p.is_dir())
    trajectory_dirs = sorted(p for p in input_dir.glob("4.*") if p.is_dir())
    if not all(len(group) == 1 for group in (profile, event_zip, imu_dirs, trajectory_dirs)):
        raise FileNotFoundError("Expected one profile XLSX, one event ZIP, one IMU directory and one trajectory directory")
    imu = sorted(imu_dirs[0].glob("*.gz"))
    trajectory = sorted(p for p in trajectory_dirs[0].glob("part-*") if p.is_file())
    if not imu or not trajectory:
        raise FileNotFoundError("IMU or trajectory parts are missing")
    return {"profile": profile[0], "events": event_zip[0], "imu": imu, "trajectory": trajectory}


def file_manifest(sources: dict[str, object], hash_files: bool) -> dict[str, object]:
    result: dict[str, object] = {"input_window": [str(START), str(END)], "sources": {}}
    for kind, value in sources.items():
        paths = value if isinstance(value, list) else [value]
        entries = []
        for path in paths:
            path = Path(path)
            item: dict[str, object] = {"path": str(path.resolve()), "bytes": path.stat().st_size}
            if hash_files:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                item["sha256"] = digest.hexdigest()
            entries.append(item)
        result["sources"][kind] = entries
    return result


def prepare_event_text(event_zip: Path, staging: Path) -> Path:
    destination = staging / "risk_events.tsv"
    staging.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    with zipfile.ZipFile(event_zip) as archive:
        candidates = [x for x in archive.infolist() if not x.is_dir() and not x.filename.startswith("__MACOSX/")]
        if len(candidates) != 1:
            raise ValueError(f"Expected one event data member, found {len(candidates)}")
        member = candidates[0]
        partial = destination.with_name(destination.name + ".partial")
        try:
            with archive.open(member) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            if partial.stat().st_size != member.file_size:
                raise IOError("Event ZIP extraction size mismatch")
            os.replace(partial, destination)
        finally:
            partial.unlink(missing_ok=True)
    return destination


def build_profile(con: duckdb.DuckDBPyConnection, source: Path, output: Path) -> dict[str, object]:
    book = load_workbook(source, read_only=True, data_only=True)
    sheet = book.active
    rows = sheet.values
    header = next(rows)
    expected = (
        "设备号", "能源类型", "月均行驶里程", "月均行驶时长", "月平均停留次数",
        "高速里程占比", "早晨行驶时长占比", "黄昏行驶时长占比", "夜间里程占比", "夜间行驶时长占比",
    )
    if tuple(header) != expected:
        raise ValueError(f"Unexpected profile columns: {header}")
    data = []
    seen = set()
    for row in rows:
        raw_id = row[0]
        if isinstance(raw_id, (int, float)) and float(raw_id).is_integer():
            gpsno = str(int(raw_id))
        else:
            gpsno = str(raw_id).strip() if raw_id is not None else ""
        if not gpsno or gpsno in seen:
            raise ValueError(f"Blank or duplicate profile gpsno: {gpsno!r}")
        seen.add(gpsno)
        data.append((gpsno, str(row[1]).strip() if row[1] is not None else None, *row[2:]))
    book.close()
    con.execute("""
        CREATE OR REPLACE TEMP TABLE profile_input (
            gpsno VARCHAR, energy_type VARCHAR, monthly_km DOUBLE, monthly_drive_hours DOUBLE,
            monthly_stops DOUBLE, highway_km_share DOUBLE, morning_hours_share DOUBLE,
            dusk_hours_share DOUBLE, night_km_share DOUBLE, night_hours_share DOUBLE
        )
    """)
    con.executemany("INSERT INTO profile_input VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data)
    copy_parquet(con, "SELECT * FROM profile_input ORDER BY gpsno", output / "profile.parquet")
    return {"rows": len(data), "unique_gpsno": len(seen), "energy_types": dict(con.execute(
        "SELECT energy_type, COUNT(*) FROM profile_input GROUP BY energy_type"
    ).fetchall())}


def build_events(con: duckdb.DuckDBPyConnection, event_zip: Path, output: Path) -> dict[str, object]:
    text_path = prepare_event_text(event_zip, output / "staging")
    scan = csv_scan([text_path], EVENT_COLUMNS, "uncompressed")
    canonical = output / "events_canonical.parquet"
    copy_parquet(con, f"""
        WITH raw AS (
            SELECT TRIM(gpsno) AS gpsno, TRY_CAST(start_time AS TIMESTAMP) AS event_ts,
                   TRY_CAST(event_type AS INTEGER) AS event_type, event_name,
                   TRY_CAST(lat AS DOUBLE) AS raw_lat, TRY_CAST(lng AS DOUBLE) AS raw_lng,
                   TRY_CAST(speed AS DOUBLE) AS raw_speed, filename AS source_file
            FROM {scan}
        )
        SELECT gpsno, event_ts, event_type, event_name,
               CASE WHEN isfinite(raw_lat) AND raw_lat BETWEEN -90 AND 90 THEN raw_lat END AS lat,
               CASE WHEN isfinite(raw_lng) AND raw_lng BETWEEN -180 AND 180 THEN raw_lng END AS lng,
               CASE WHEN isfinite(raw_speed) AND raw_speed BETWEEN 0 AND 180 THEN raw_speed END AS speed,
               raw_lat IS NULL OR NOT isfinite(raw_lat) OR raw_lat NOT BETWEEN -90 AND 90 AS bad_lat,
               raw_lng IS NULL OR NOT isfinite(raw_lng) OR raw_lng NOT BETWEEN -180 AND 180 AS bad_lng,
               raw_speed IS NULL OR NOT isfinite(raw_speed) OR raw_speed NOT BETWEEN 0 AND 180 AS bad_speed,
               source_file
        FROM raw
    """, canonical)
    source = f"read_parquet({quoted(canonical)})"
    clean = output / "events_clean.parquet"
    copy_parquet(con, f"""
        SELECT DISTINCT gpsno, event_ts, event_type, event_name, lat, lng, speed
        FROM {source}
        WHERE gpsno <> '' AND event_ts >= TIMESTAMP '2026-06-01'
          AND event_ts < TIMESTAMP '2026-07-31' AND event_type IS NOT NULL
    """, clean)
    clean_source = f"read_parquet({quoted(clean)})"
    daily = output / "event_daily.parquet"
    counts = ",\n".join(
        f"COUNT(*) FILTER (WHERE event_type={code}) AS event_{code}_count, "
        f"COUNT(*) FILTER (WHERE event_type={code} AND episode_start) AS event_{code}_episodes"
        for code in EVENT_CODES
    )
    known = ",".join(map(str, EVENT_CODES))
    copy_parquet(con, f"""
        WITH ordered AS (
            SELECT *, LAG(event_ts) OVER (
                PARTITION BY gpsno, event_type ORDER BY event_ts
            ) AS previous_ts
            FROM {clean_source}
        ), episodes AS (
            SELECT *, previous_ts IS NULL OR DATE_DIFF('second', previous_ts, event_ts) > 60 AS episode_start
            FROM ordered
        )
        SELECT gpsno, CAST(event_ts AS DATE) AS day, COUNT(*) AS event_total_count,
               COUNT(*) FILTER (WHERE episode_start) AS event_total_episodes,
               COUNT(*) FILTER (WHERE event_type NOT IN ({known})) AS event_other_count,
               QUANTILE_CONT(speed, 0.9) AS event_speed_p90,
               {counts}
        FROM episodes GROUP BY gpsno, day
    """, daily)
    accidents = output / "accidents.parquet"
    copy_parquet(con, f"""
        SELECT gpsno, event_ts, event_type FROM {clean_source}
        WHERE event_type IN (11803,11804)
        ORDER BY gpsno, event_ts
    """, accidents)
    row = con.execute(f"""
        SELECT COUNT(*), COUNT(DISTINCT gpsno),
               COUNT(*) FILTER (WHERE event_ts IS NULL OR event_type IS NULL OR gpsno=''),
               COUNT(*) FILTER (WHERE event_ts >= TIMESTAMP '2026-07-31'),
               COUNT(*) FILTER (WHERE bad_lat OR bad_lng),
               COUNT(*) FILTER (WHERE bad_speed)
        FROM {source}
    """).fetchone()
    report = {
        "raw_rows": row[0], "raw_vehicles": row[1], "invalid_key_or_time_rows": row[2],
        "after_rule_end_rows": row[3], "bad_coordinate_rows": row[4], "bad_speed_rows": row[5],
        "clean_rows": scalar(con, f"SELECT COUNT(*) FROM {clean_source}"),
        "accident_rows": scalar(con, f"SELECT COUNT(*) FROM read_parquet({quoted(accidents)})"),
    }
    return report


def build_trajectory(con: duckdb.DuckDBPyConnection, files: list[Path], output: Path, resume: bool = False) -> dict[str, object]:
    canonical_dir = output / "trajectory_canonical"
    canonical_dir.mkdir(exist_ok=True)
    canonical_paths = []
    for index, raw_file in enumerate(files):
        scan = csv_scan([raw_file], TRAJECTORY_COLUMNS)
        canonical = canonical_dir / f"part-{index:05d}.parquet"
        if resume and up_to_date(canonical, [raw_file]):
            print(f"[trajectory] reusing canonical part {index + 1}/{len(files)}", flush=True)
        else:
            print(f"[trajectory] parsing part {index + 1}/{len(files)}", flush=True)
            copy_parquet(con, f"""
        WITH raw AS (
            SELECT TRIM(gpsno) AS gpsno, TRY_CAST(trigger_time AS TIMESTAMP) AS ts,
                   TRY_CAST(distance AS BIGINT) AS distance_cm,
                   TRY_CAST(run_time AS BIGINT) AS run_time_s,
                   TRY_CAST(speed AS DOUBLE) AS raw_speed,
                   TRY_CAST(course AS DOUBLE) AS raw_course,
                   TRY_CAST(lat AS DOUBLE) AS raw_lat,
                   TRY_CAST(lng AS DOUBLE) AS raw_lng,
                   filename AS source_file
            FROM {scan}
        )
        SELECT gpsno, ts, distance_cm, run_time_s,
               CASE WHEN isfinite(raw_speed) AND raw_speed BETWEEN 0 AND 180 THEN raw_speed END AS speed,
               CASE WHEN isfinite(raw_course) AND raw_course BETWEEN 0 AND 359 THEN raw_course END AS course,
               CASE WHEN isfinite(raw_lat) AND raw_lat BETWEEN -90 AND 90 THEN raw_lat END AS lat,
               CASE WHEN isfinite(raw_lng) AND raw_lng BETWEEN -180 AND 180 THEN raw_lng END AS lng,
               source_file
        FROM raw
            """, canonical)
        canonical_paths.append(canonical)
    source = parquet_scan(canonical_paths)
    daily = output / "trajectory_daily.parquet"
    def daily_query(part_source: str) -> str:
        return f"""
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY gpsno, ts ORDER BY
                (CASE WHEN speed IS NOT NULL THEN 1 ELSE 0 END +
                 CASE WHEN lat IS NOT NULL AND lng IS NOT NULL THEN 1 ELSE 0 END) DESC,
                source_file
            ) AS duplicate_rank
            FROM {part_source}
            WHERE gpsno <> '' AND ts >= TIMESTAMP '2026-06-01' AND ts < TIMESTAMP '2026-07-31'
        ), ordered AS (
            SELECT *, LAG(ts) OVER w AS previous_ts,
                   LAG(speed) OVER w AS previous_speed,
                   LAG(course) OVER w AS previous_course,
                   LAG(lat) OVER w AS previous_lat,
                   LAG(lng) OVER w AS previous_lng
            FROM ranked WHERE duplicate_rank=1
            WINDOW w AS (PARTITION BY gpsno ORDER BY ts)
        ), intervals AS (
            SELECT *, DATE_DIFF('second', previous_ts, ts) AS gap_s,
                   CASE WHEN distance_cm >= 0 AND run_time_s > 0
                        THEN 3.6 * distance_cm / (100.0 * run_time_s) END AS implied_speed
            FROM ordered
        ), checked AS (
            SELECT *,
                   distance_cm = 0 AND run_time_s = 0 AS valid_zero,
                   distance_cm > 0 AND run_time_s > 0 AND gap_s > 0
                       AND run_time_s <= gap_s + 2
                       AND implied_speed <= 160
                       AND (speed IS NULL OR ABS(implied_speed-speed) <= GREATEST(20, 0.5*implied_speed))
                       AS valid_increment,
                   speed IS NOT NULL AND speed > 3 AND gap_s BETWEEN 1 AND 30 AS valid_fallback
            FROM intervals
        ), features AS (
            SELECT *,
                   CASE WHEN valid_zero THEN 0.0
                        WHEN valid_increment THEN distance_cm / 100000.0
                        WHEN valid_fallback THEN speed * gap_s / 3600.0 END AS distance_km,
                   CASE WHEN valid_increment THEN run_time_s / 3600.0
                        WHEN valid_fallback THEN gap_s / 3600.0 END AS drive_hours,
                   CASE WHEN speed > 3 AND (previous_speed IS NULL OR previous_speed <= 3 OR gap_s > 600)
                        THEN 1 ELSE 0 END AS trip_start,
                   CASE WHEN gap_s BETWEEN 1 AND 10 AND previous_speed IS NOT NULL AND speed IS NOT NULL
                        THEN ABS(speed - previous_speed) / gap_s END AS speed_change_per_s,
                   CASE WHEN gap_s BETWEEN 1 AND 10 AND previous_course IS NOT NULL AND course IS NOT NULL
                        THEN LEAST(ABS(course-previous_course),360-ABS(course-previous_course))/gap_s
                        END AS turn_deg_per_s,
                   CASE WHEN gap_s BETWEEN 1 AND 60 AND lat IS NOT NULL AND lng IS NOT NULL
                             AND previous_lat IS NOT NULL AND previous_lng IS NOT NULL
                        THEN 2*6371*ASIN(SQRT(LEAST(1.0,
                            POW(SIN(RADIANS(lat-previous_lat)/2),2) +
                            COS(RADIANS(previous_lat))*COS(RADIANS(lat))*
                            POW(SIN(RADIANS(lng-previous_lng)/2),2)))) END AS gps_step_km
            FROM checked
        )
        SELECT gpsno, CAST(ts AS DATE) AS day,
               COUNT(*) AS trajectory_rows,
               COUNT(*) FILTER (WHERE speed > 3) AS moving_rows,
               COUNT(*) FILTER (WHERE distance_km IS NOT NULL) AS distance_valid_rows,
               COUNT(*) FILTER (WHERE distance_km IS NULL) AS distance_invalid_rows,
               COUNT(*) FILTER (WHERE distance_cm > 0 AND run_time_s = 0) AS distance_without_runtime_rows,
               SUM(distance_km) AS distance_km,
               SUM(drive_hours) AS drive_hours,
               SUM(gps_step_km) AS gps_distance_check_km,
               SUM(trip_start) AS trip_starts,
               SUM(CASE WHEN EXTRACT(HOUR FROM ts)<6 OR EXTRACT(HOUR FROM ts)>=22
                        THEN distance_km ELSE 0 END) AS night_distance_km,
               QUANTILE_CONT(speed, 0.5) AS speed_p50,
               QUANTILE_CONT(speed, 0.9) AS speed_p90,
               QUANTILE_CONT(speed_change_per_s, 0.9) AS speed_change_p90,
               QUANTILE_CONT(turn_deg_per_s, 0.9) AS turn_rate_p90,
               COUNT(*) FILTER (WHERE gap_s > 60) AS long_gap_rows
        FROM features GROUP BY gpsno, day
        """
    overlap = scalar(con, f"""
        SELECT COUNT(*) FROM (
            SELECT gpsno FROM {source} GROUP BY gpsno HAVING COUNT(DISTINCT source_file)>1
        )
    """)
    daily_part_dir = output / "trajectory_daily_parts"
    daily_part_dir.mkdir(exist_ok=True)
    daily_sources = [source] if overlap else [f"read_parquet({quoted(p)})" for p in canonical_paths]
    print(f"[trajectory] sorting and aggregating {len(daily_sources)} vehicle partitions; cross-part vehicles={overlap}", flush=True)
    daily_part_paths = []
    for index, part_source in enumerate(daily_sources):
        part_output = daily_part_dir / f"part-{index:05d}.parquet"
        dependencies = canonical_paths if overlap else [canonical_paths[index]]
        if resume and up_to_date(part_output, dependencies):
            print(f"[trajectory] reusing daily part {index + 1}/{len(daily_sources)}", flush=True)
        else:
            copy_parquet(con, daily_query(part_source), part_output)
            print(f"[trajectory] aggregated part {index + 1}/{len(daily_sources)}", flush=True)
        daily_part_paths.append(part_output)
    if not resume or not up_to_date(daily, daily_part_paths):
        copy_parquet(con, f"SELECT * FROM {parquet_scan(daily_part_paths)}", daily)
    row = con.execute(f"""
        SELECT COUNT(*), COUNT(DISTINCT gpsno),
               COUNT(*) FILTER (WHERE ts IS NULL OR gpsno=''),
               COUNT(*) FILTER (WHERE ts >= TIMESTAMP '2026-07-31'),
               COUNT(*) FILTER (WHERE distance_cm > 0 AND run_time_s=0),
               COUNT(*) FILTER (WHERE speed IS NULL)
        FROM {source}
    """).fetchone()
    report = dict(zip(("raw_rows", "raw_vehicles", "invalid_key_or_time_rows", "after_rule_end_rows",
                       "distance_without_runtime_rows", "invalid_speed_rows"), row))
    report["cross_part_vehicles"] = overlap
    return report


def build_imu(con: duckdb.DuckDBPyConnection, files: list[Path], output: Path, resume: bool = False) -> dict[str, object]:
    canonical_dir = output / "imu_canonical"
    canonical_dir.mkdir(exist_ok=True)
    canonical_paths = []
    for index, raw_file in enumerate(files):
        scan = csv_scan([raw_file], IMU_COLUMNS, "gzip")
        canonical = canonical_dir / f"part-{index:05d}.parquet"
        if resume and up_to_date(canonical, [raw_file]):
            print(f"[imu] reusing canonical part {index + 1}/{len(files)}", flush=True)
        else:
            print(f"[imu] parsing compressed part {index + 1}/{len(files)}", flush=True)
            copy_parquet(con, f"""
        WITH raw AS (
            SELECT TRIM(gpsno) AS gpsno, TRIM(imei) AS imei,
                   TRY_CAST(data_time AS TIMESTAMP) AS ts,
                   TRY_CAST(data_date AS INTEGER) AS data_date,
                   TRY_CAST(ems_speed AS DOUBLE) AS ems_raw,
                   TRY_CAST(gps_speed AS DOUBLE) AS gps_raw,
                   TRY_CAST(accel_x_raw AS DOUBLE) AS ax_raw,
                   TRY_CAST(accel_y_raw AS DOUBLE) AS ay_raw,
                   TRY_CAST(accel_z_raw AS DOUBLE) AS az_raw,
                   TRY_CAST(gyro_x_raw AS DOUBLE) AS gx_raw,
                   TRY_CAST(gyro_y_raw AS DOUBLE) AS gy_raw,
                   TRY_CAST(gyro_z_raw AS DOUBLE) AS gz_raw,
                   filename AS source_file
            FROM {scan}
        )
        SELECT gpsno, imei, ts, data_date,
               CASE WHEN isfinite(ems_raw) AND ems_raw BETWEEN 0 AND 180 THEN ems_raw END AS ems_speed,
               CASE WHEN isfinite(gps_raw) AND gps_raw BETWEEN 0 AND 180 THEN gps_raw END AS gps_speed,
               CASE WHEN isfinite(ax_raw) AND ABS(ax_raw)<100 THEN ax_raw END AS ax,
               CASE WHEN isfinite(ay_raw) AND ABS(ay_raw)<100 THEN ay_raw END AS ay,
               CASE WHEN isfinite(az_raw) AND ABS(az_raw)<100 THEN az_raw END AS az,
               CASE WHEN isfinite(gx_raw) AND ABS(gx_raw)<10000 THEN gx_raw END AS gx,
               CASE WHEN isfinite(gy_raw) AND ABS(gy_raw)<10000 THEN gy_raw END AS gy,
               CASE WHEN isfinite(gz_raw) AND ABS(gz_raw)<10000 THEN gz_raw END AS gz,
               source_file
        FROM raw
            """, canonical)
        canonical_paths.append(canonical)
    source = parquet_scan(canonical_paths)
    def window_query(part_source: str) -> str:
        return f"""
        WITH base AS (
            SELECT gpsno, imei, ts, CAST(ts AS DATE) AS day,
                   CAST(FLOOR(EPOCH(ts)/10) AS BIGINT) AS window_id,
                   COALESCE(ems_speed,gps_speed) AS speed,
                   ems_speed, gps_speed,
                   CASE WHEN ax IS NOT NULL AND ay IS NOT NULL AND az IS NOT NULL
                        THEN SQRT(ax*ax+ay*ay+az*az) END AS accel_norm,
                   CASE WHEN gx IS NOT NULL AND gy IS NOT NULL AND gz IS NOT NULL
                        THEN SQRT(gx*gx+gy*gy+gz*gz) END AS gyro_norm
            FROM {part_source}
            WHERE gpsno <> '' AND ts >= TIMESTAMP '2026-06-01' AND ts < TIMESTAMP '2026-07-31'
        )
        SELECT gpsno, imei, day, window_id, COUNT(*) AS sample_count,
               COUNT(accel_norm) AS valid_accel_samples,
               COUNT(gyro_norm) AS valid_gyro_samples,
               COUNT(*) FILTER (WHERE ems_speed IS NULL) AS ems_missing_samples,
               COUNT(*) FILTER (WHERE gps_speed IS NULL) AS gps_missing_samples,
               MAX(ts) - MIN(ts) AS window_span,
               QUANTILE_CONT(accel_norm,0.95) AS accel_norm_p95,
               MAX(accel_norm) AS accel_norm_max,
               STDDEV_SAMP(accel_norm) AS accel_norm_std,
               QUANTILE_CONT(gyro_norm,0.95) AS gyro_norm_p95,
               MAX(gyro_norm) AS gyro_norm_max,
               QUANTILE_CONT(speed,0.5) AS speed_p50
        FROM base GROUP BY gpsno, imei, day, window_id
        HAVING COUNT(*) >= 3 AND COUNT(accel_norm) >= 3 AND COUNT(gyro_norm) >= 3
        """
    overlap = scalar(con, f"""
        SELECT COUNT(*) FROM (
            SELECT gpsno FROM {source} GROUP BY gpsno HAVING COUNT(DISTINCT source_file)>1
        )
    """)
    window_dir = output / "imu_windows"
    window_dir.mkdir(exist_ok=True)
    window_sources = [source] if overlap else [f"read_parquet({quoted(p)})" for p in canonical_paths]
    print(f"[imu] aggregating {len(window_sources)} ten-second-window partitions; cross-part vehicles={overlap}", flush=True)
    window_paths = []
    for index, part_source in enumerate(window_sources):
        window_path = window_dir / f"part-{index:05d}.parquet"
        dependencies = canonical_paths if overlap else [canonical_paths[index]]
        if resume and up_to_date(window_path, dependencies):
            print(f"[imu] reusing window part {index + 1}/{len(window_sources)}", flush=True)
        else:
            copy_parquet(con, window_query(part_source), window_path)
            print(f"[imu] aggregated window part {index + 1}/{len(window_sources)}", flush=True)
        window_paths.append(window_path)
    window_source = parquet_scan(window_paths)
    daily = output / "imu_daily.parquet"
    def daily_query(part_source: str) -> str:
        return f"""
        SELECT gpsno, day, COUNT(*) AS imu_valid_windows,
               SUM(sample_count) AS imu_valid_samples,
               SUM(ems_missing_samples) AS imu_ems_missing_samples,
               SUM(gps_missing_samples) AS imu_gps_missing_samples,
               QUANTILE_CONT(accel_norm_p95,0.5) AS imu_accel_p95_median,
               QUANTILE_CONT(accel_norm_max,0.99) AS imu_accel_peak_p99,
               QUANTILE_CONT(accel_norm_std,0.9) AS imu_accel_variability_p90,
               QUANTILE_CONT(gyro_norm_p95,0.9) AS imu_gyro_p95_p90,
               MAX(gyro_norm_max) AS imu_gyro_peak,
               QUANTILE_CONT(speed_p50,0.5) AS imu_speed_p50
        FROM {part_source} GROUP BY gpsno, day
        """
    daily_part_dir = output / "imu_daily_parts"
    daily_part_dir.mkdir(exist_ok=True)
    print(f"[imu] aggregating {len(window_paths)} vehicle-day partitions", flush=True)
    daily_paths = []
    for index, part_file in enumerate(window_paths):
        daily_path = daily_part_dir / f"part-{index:05d}.parquet"
        if not resume or not up_to_date(daily_path, [part_file]):
            copy_parquet(con, daily_query(f"read_parquet({quoted(part_file)})"), daily_path)
        daily_paths.append(daily_path)
    if not resume or not up_to_date(daily, daily_paths):
        copy_parquet(con, f"SELECT * FROM {parquet_scan(daily_paths)}", daily)
    row = con.execute(f"""
        SELECT COUNT(*), COUNT(DISTINCT gpsno),
               COUNT(*) FILTER (WHERE ts IS NULL OR gpsno=''),
               COUNT(*) FILTER (WHERE ts >= TIMESTAMP '2026-07-31'),
               COUNT(*) FILTER (WHERE ems_speed IS NULL),
               COUNT(*) FILTER (WHERE gps_speed IS NULL),
               COUNT(*) FILTER (WHERE ax IS NULL OR ay IS NULL OR az IS NULL),
               COUNT(*) FILTER (WHERE gx IS NULL OR gy IS NULL OR gz IS NULL)
        FROM {source}
    """).fetchone()
    report = dict(zip(("raw_rows", "raw_vehicles", "invalid_key_or_time_rows", "after_rule_end_rows",
                       "ems_speed_missing_rows", "gps_speed_missing_rows", "accel_invalid_rows",
                       "gyro_invalid_rows"), row))
    report["valid_windows"] = scalar(con, f"SELECT COUNT(*) FROM {window_source}")
    report["cross_part_vehicles"] = overlap
    return report


def build_assembled(con: duckdb.DuckDBPyConnection, output: Path) -> dict[str, object]:
    needed = ("profile.parquet", "events_clean.parquet", "event_daily.parquet", "accidents.parquet",
              "trajectory_daily.parquet", "imu_daily.parquet")
    missing = [name for name in needed if not (output / name).exists()]
    if missing:
        raise FileNotFoundError("Run the required stages first: " + ", ".join(missing))
    p = lambda name: f"read_parquet({quoted(output / name)})"
    coverage = output / "coverage.parquet"
    copy_parquet(con, f"""
        WITH ev AS (
            SELECT gpsno, MIN(CAST(event_ts AS DATE)) AS event_first_day,
                   MAX(CAST(event_ts AS DATE)) AS event_last_day, COUNT(*) AS event_rows
            FROM {p('events_clean.parquet')} GROUP BY gpsno
        ), tr AS (
            SELECT gpsno, MIN(day) AS trajectory_first_day, MAX(day) AS trajectory_last_day,
                   SUM(trajectory_rows) AS trajectory_rows, COUNT(*) AS trajectory_days
            FROM {p('trajectory_daily.parquet')} GROUP BY gpsno
        ), im AS (
            SELECT gpsno, MIN(day) AS imu_first_day, MAX(day) AS imu_last_day,
                   SUM(imu_valid_samples) AS imu_valid_samples, COUNT(*) AS imu_days
            FROM {p('imu_daily.parquet')} GROUP BY gpsno
        )
        SELECT p.gpsno, p.energy_type,
               ev.gpsno IS NOT NULL AS has_event_feed,
               tr.gpsno IS NOT NULL AS has_trajectory,
               im.gpsno IS NOT NULL AS has_imu,
               ev.event_first_day, ev.event_last_day, ev.event_rows,
               tr.trajectory_first_day, tr.trajectory_last_day, tr.trajectory_rows, tr.trajectory_days,
               im.imu_first_day, im.imu_last_day, im.imu_valid_samples, im.imu_days
        FROM {p('profile.parquet')} p
        LEFT JOIN ev USING(gpsno) LEFT JOIN tr USING(gpsno) LEFT JOIN im USING(gpsno)
        ORDER BY p.gpsno
    """, coverage)
    con.execute(f"CREATE OR REPLACE TEMP VIEW coverage_view AS SELECT * FROM {p('coverage.parquet')}")
    daily = output / "daily_features.parquet"
    event_fields = ",\n".join(
        f"CASE WHEN c.has_event_feed THEN COALESCE(e.event_{code}_count,0) END AS event_{code}_count, "
        f"CASE WHEN c.has_event_feed THEN COALESCE(e.event_{code}_episodes,0) END AS event_{code}_episodes"
        for code in EVENT_CODES
    )
    copy_parquet(con, f"""
        WITH calendar AS (
            SELECT c.gpsno, c.energy_type,
                   COALESCE(c.event_first_day <= CAST(day AS DATE), FALSE) AS has_event_feed,
                   COALESCE(c.trajectory_first_day <= CAST(day AS DATE), FALSE) AS has_trajectory,
                   COALESCE(c.imu_first_day <= CAST(day AS DATE), FALSE) AS has_imu,
                   CAST(day AS DATE) AS day
            FROM coverage_view c,
                 GENERATE_SERIES(DATE '2026-06-01', DATE '2026-07-30', INTERVAL 1 DAY) AS d(day)
        )
        SELECT c.gpsno, c.day, c.energy_type,
               c.has_event_feed, c.has_trajectory, c.has_imu,
               e.gpsno IS NOT NULL AS event_recorded_today,
               t.gpsno IS NOT NULL AS trajectory_recorded_today,
               i.gpsno IS NOT NULL AS imu_recorded_today,
               t.gpsno IS NOT NULL AND t.moving_rows=0 AS observed_stationary_day,
               CASE WHEN c.has_event_feed THEN COALESCE(e.event_total_count,0) END AS event_total_count,
               CASE WHEN c.has_event_feed THEN COALESCE(e.event_total_episodes,0) END AS event_total_episodes,
               CASE WHEN c.has_event_feed THEN COALESCE(e.event_other_count,0) END AS event_other_count,
               e.event_speed_p90,
               {event_fields},
               t.trajectory_rows, t.moving_rows, t.distance_valid_rows, t.distance_invalid_rows,
               t.distance_without_runtime_rows, t.distance_km, t.drive_hours,
               t.gps_distance_check_km, t.trip_starts, t.night_distance_km,
               t.speed_p50, t.speed_p90, t.speed_change_p90, t.turn_rate_p90, t.long_gap_rows,
               i.imu_valid_windows, i.imu_valid_samples,
               i.imu_ems_missing_samples, i.imu_gps_missing_samples,
               i.imu_accel_p95_median, i.imu_accel_peak_p99,
               i.imu_accel_variability_p90, i.imu_gyro_p95_p90, i.imu_gyro_peak, i.imu_speed_p50,
               CASE WHEN t.distance_km > 0 AND c.has_event_feed
                    THEN 100.0*COALESCE(e.event_total_count,0)/t.distance_km END AS events_per_100km,
               CASE WHEN t.drive_hours > 0 AND c.has_event_feed
                    THEN COALESCE(e.event_total_count,0)/t.drive_hours END AS events_per_drive_hour
        FROM calendar c
        LEFT JOIN {p('event_daily.parquet')} e USING(gpsno,day)
        LEFT JOIN {p('trajectory_daily.parquet')} t USING(gpsno,day)
        LEFT JOIN {p('imu_daily.parquet')} i USING(gpsno,day)
        ORDER BY c.gpsno,c.day
    """, daily)
    anchors = [PROXY_START + timedelta(days=i) for i in range((PROXY_END - PROXY_START).days + 1)] + [END]
    con.execute("CREATE OR REPLACE TEMP TABLE anchors(anchor_date DATE)")
    con.executemany("INSERT INTO anchors VALUES (?)", [(x,) for x in anchors])
    bags = output / "bag_index.parquet"
    copy_parquet(con, f"""
        WITH labels AS (
            SELECT p.gpsno, a.anchor_date,
                   COUNT(x.event_ts) AS accident_events,
                   MIN(x.event_ts) AS first_accident_ts
            FROM {p('profile.parquet')} p CROSS JOIN anchors a
            LEFT JOIN {p('accidents.parquet')} x ON x.gpsno=p.gpsno
                AND x.event_ts >= CAST(a.anchor_date + INTERVAL 1 DAY AS TIMESTAMP)
                AND x.event_ts < CAST(a.anchor_date + INTERVAL 41 DAY AS TIMESTAMP)
            GROUP BY p.gpsno,a.anchor_date
        )
        SELECT l.gpsno, l.anchor_date,
               GREATEST(DATE '2026-06-01', l.anchor_date - INTERVAL 19 DAY)::DATE AS input_start,
               l.anchor_date AS input_end,
               (l.anchor_date + INTERVAL 40 DAY)::DATE AS future_end,
               CASE WHEN l.anchor_date=DATE '2026-07-30' OR NOT c.has_event_feed THEN NULL
                    WHEN l.accident_events>0 THEN 1 ELSE 0 END AS label,
               CASE WHEN l.anchor_date=DATE '2026-07-30' THEN 'target_unknown'
                    WHEN NOT c.has_event_feed THEN 'unlabeled_no_event_feed'
                    WHEN l.accident_events>0 THEN 'observed_positive'
                    ELSE 'provisional_negative' END AS label_status,
               CASE WHEN l.anchor_date=DATE '2026-07-30' THEN NULL ELSE l.accident_events END AS accident_events,
               CASE WHEN l.anchor_date=DATE '2026-07-30' THEN NULL ELSE l.first_accident_ts END AS first_accident_ts,
               c.has_event_feed, c.has_trajectory, c.has_imu
        FROM labels l JOIN coverage_view c USING(gpsno)
        ORDER BY l.gpsno,l.anchor_date
    """, bags)
    coverage_source = p("coverage.parquet")
    bags_source = p("bag_index.parquet")
    report = {
        "profile_vehicles": scalar(con, f"SELECT COUNT(*) FROM {coverage_source}"),
        "event_joined_vehicles": scalar(con, f"SELECT COUNT(*) FROM {coverage_source} WHERE has_event_feed"),
        "trajectory_joined_vehicles": scalar(con, f"SELECT COUNT(*) FROM {coverage_source} WHERE has_trajectory"),
        "imu_joined_vehicles": scalar(con, f"SELECT COUNT(*) FROM {coverage_source} WHERE has_imu"),
        "daily_rows": scalar(con, f"SELECT COUNT(*) FROM {p('daily_features.parquet')}"),
        "proxy_june20_positive": scalar(con, f"SELECT COUNT(*) FROM {bags_source} WHERE anchor_date=DATE '2026-06-20' AND label=1"),
        "proxy_june20_provisional_negative": scalar(con, f"SELECT COUNT(*) FROM {bags_source} WHERE anchor_date=DATE '2026-06-20' AND label=0"),
        "proxy_june20_unlabeled": scalar(con, f"SELECT COUNT(*) FROM {bags_source} WHERE anchor_date=DATE '2026-06-20' AND label IS NULL"),
    }
    if report["daily_rows"] != report["profile_vehicles"] * (END - START).days + report["profile_vehicles"]:
        raise AssertionError("Daily calendar row count does not match profile vehicles x 60 days")
    return report


def run(args: argparse.Namespace) -> None:
    input_dir = args.input.resolve()
    output = (args.output if args.output else input_dir / OUTPUT_DIRNAME).resolve()
    if input_dir not in output.parents:
        raise ValueError("Output must be a separate subdirectory inside the competition database directory")
    if output.relative_to(input_dir).parts[0].startswith(("3.", "4.")):
        raise ValueError("Output top-level directory must not match the raw IMU or trajectory directory prefixes")
    sources = discover(input_dir)
    raw_directories = {Path(path).parent.resolve() for path in (*sources["imu"], *sources["trajectory"])}
    if any(output == raw_dir or raw_dir in output.parents for raw_dir in raw_directories):
        raise ValueError("Output must not be inside a raw IMU or trajectory directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", file_manifest(sources, args.hash_files))
    temp = args.temp_directory.resolve() if args.temp_directory else output / "duckdb_temp"
    if output not in temp.parents:
        raise ValueError("DuckDB temp directory must be inside the processed output directory")
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(output / "preprocess.duckdb"))
    try:
        con.execute(f"SET threads={args.threads}")
        con.execute(f"SET memory_limit={quoted(args.memory_limit)}")
        con.execute(f"SET temp_directory={quoted(temp)}")
        reports: dict[str, object] = {}
        selected = ["profile", "events", "trajectory", "imu", "assemble"] if args.stage == "all" else [args.stage]
        for stage in selected:
            print(f"[{stage}] started", flush=True)
            if stage == "profile":
                result = build_profile(con, sources["profile"], output)
            elif stage == "events":
                result = build_events(con, sources["events"], output)
            elif stage == "trajectory":
                result = build_trajectory(con, sources["trajectory"], output, args.resume)
            elif stage == "imu":
                result = build_imu(con, sources["imu"], output, args.resume)
            else:
                result = build_assembled(con, output)
            reports[stage] = result
            write_json(output / f"quality_{stage}.json", result)
            print(f"[{stage}] complete: {json.dumps(result, ensure_ascii=False, default=str)}", flush=True)
        write_json(output / "run_config.json", {
            "stage": args.stage, "input": str(input_dir), "output": str(output),
            "resume": args.resume, "temp_directory": str(temp),
            "start": str(START), "end": str(END), "proxy_anchors": [str(PROXY_START), str(PROXY_END)],
            "event_episode_gap_seconds": 60, "imu_window_seconds": 10,
            "trajectory_speed_limit_kmh": 180, "trajectory_accepted_implied_speed_kmh": 160,
            "night_hours": "22:00-06:00", "imu_features": "rotation-invariant only",
            "reports": reports,
        })
    finally:
        con.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Competition source directory")
    parser.add_argument("--output", type=Path, help=f"Generated output directory inside --input; defaults to --input/{OUTPUT_DIRNAME}")
    parser.add_argument("--stage", choices=("all", "profile", "events", "trajectory", "imu", "assemble"), default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="6GB", help="DuckDB memory limit, e.g. 6GB")
    parser.add_argument("--temp-directory", type=Path, help="DuckDB spill directory; defaults to OUTPUT/duckdb_temp")
    parser.add_argument("--hash-files", action="store_true", help="SHA-256 all raw files; adds one full read of 57GB")
    parser.add_argument("--resume", action="store_true", help="Reuse completed trajectory/IMU parts when raw files and code options are unchanged")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        print(f"preprocessing failed: {exc}", file=sys.stderr)
        raise
