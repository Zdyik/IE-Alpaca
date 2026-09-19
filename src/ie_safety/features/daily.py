"""日粒度聚合：只扫一遍原始数据，把数十 GB 压成 ``(gpsno, date)`` 小表。

三个数据集的重活都在这里：

* **轨迹**（数十 GB、轨迹点级）：按 ``(gpsno, date)`` 累加里程/时长/夜间里程/
  高速里程与速度统计。``distance`` 单位是**厘米**，需除以 1e5 转 km。
* **风险事件**（事件级）：按 ``(gpsno, date)`` 透视成 ``c_<事件码>`` 计数，
  并额外统计夜间/凌晨与超高危事件数。
* **IMU**（高频时序）：先估计设备姿态与每车自适应阈值，再检测六类风险场景。

关于 IMU 的两点说明（直接对应出题方参考文档给出的注意事项）：

1. **设备姿态未必水平。** 直接拿 ``accel_x_raw`` 当「前后方向加速度」是错的 ——
   这与设备怎么粘在车上有关。这里先用整段数据的加速度均值估计重力方向，再用
   水平面内方差最大的方向作为车辆前向轴，从而把原始三轴投影到车辆坐标系。
2. **不同车型的阈值不同。** 所以不用绝对阈值，改用**每车自适应**的
   ``median + k × MAD``（MAD 对异常值稳健，不会被事件本身拉偏）。

受限于两遍扫描的成本，姿态与阈值用**每车抽样**估计（默认前 20000 行），再对
全量数据做检测。这一取舍在文档中显式写明。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..io.readers import Source, parse_datetime, to_gpsno_str, to_naive_local_date

logger = logging.getLogger(__name__)

#: 估计姿态与自适应阈值时，每辆车最多抽样的行数
IMU_CALIBRATION_SAMPLE = 20_000
#: 自适应阈值倍数（median + k × MAD）
IMU_THRESHOLD_K = 4.5
#: 疑似碰撞的额外强度要求（相对阈值的倍数）
CRASH_K = 8.0
#: 疑似侧翻的垂直加速度波动要求
ROLLOVER_K = 6.0


# ---------------------------------------------------------------------------
# 轨迹
# ---------------------------------------------------------------------------
def build_daily_exposure(sources: List[Source], cfg: Config, chunksize: int = 2_000_000) -> pd.DataFrame:
    """把轨迹数据聚合成 ``(gpsno, date)`` 的暴露量表。"""
    if not sources:
        raise FileNotFoundError("未发现轨迹数据（列签名需含 distance/run_time/course）")

    div = float(cfg.exposure["distance_unit_divisor"])
    hw_thr = float(cfg.exposure["highway_speed_threshold"])
    night_hours = set(int(h) for h in cfg.temporal["night_hours"])

    parts: List[pd.DataFrame] = []
    for src in sources:
        logger.info("聚合轨迹: %s", src.label)
        for chunk in src.iter_chunks(chunksize=chunksize):
            if "gpsno" not in chunk.columns or "distance" not in chunk.columns:
                continue
            chunk["gpsno"] = to_gpsno_str(chunk["gpsno"])
            tcol = "trigger_time" if "trigger_time" in chunk.columns else None
            if tcol:
                ts = parse_datetime(chunk[tcol], cfg.timezone)
            elif "data_time" in chunk.columns:
                ts = parse_datetime(chunk["data_time"], cfg.timezone)
            else:
                continue
            chunk = chunk.assign(ts=ts).dropna(subset=["gpsno", "ts"])
            if not len(chunk):
                continue
            secs = (
                pd.to_numeric(chunk["run_time"], errors="coerce").fillna(0.0)
                if "run_time" in chunk.columns
                else pd.Series(0.0, index=chunk.index)
            )
            spd = (
                pd.to_numeric(chunk["speed"], errors="coerce")
                if "speed" in chunk.columns
                else pd.Series(np.nan, index=chunk.index)
            )
            chunk = chunk.assign(
                date=to_naive_local_date(chunk["ts"]),
                hour=chunk["ts"].dt.hour,
                dist_km=pd.to_numeric(chunk["distance"], errors="coerce").fillna(0.0) / div,
                secs=secs,
                spd=spd,
            )
            chunk["is_night"] = chunk["hour"].isin(night_hours)
            chunk["is_hw"] = chunk["spd"] >= hw_thr

            # 用命名聚合而不是手工拼 dict：分项 groupby 的索引集合不同，
            # 手工拼会触发对齐产生 NaN，命名聚合则天然正确。
            agg = chunk.groupby(["gpsno", "date"], sort=False).agg(
                km=("dist_km", "sum"),
                secs=("secs", "sum"),
                speed_sum=("spd", "sum"),
                speed_max=("spd", "max"),
                n_points=("spd", "size"),
            )
            agg["hours"] = agg["secs"] / 3600.0
            agg = agg.drop(columns=["secs"])
            night = chunk[chunk["is_night"]].groupby(["gpsno", "date"], sort=False)["dist_km"].sum()
            hw = chunk[chunk["is_hw"]].groupby(["gpsno", "date"], sort=False)["dist_km"].sum()
            agg["night_km"] = night.reindex(agg.index).fillna(0.0)
            agg["highway_km"] = hw.reindex(agg.index).fillna(0.0)
            agg["speed_sum"] = agg["speed_sum"].fillna(0.0)
            parts.append(agg.reset_index())

    if not parts:
        raise ValueError("轨迹数据聚合后为空，请检查列名与时间列")
    out = pd.concat(parts, ignore_index=True)
    out = out.groupby(["gpsno", "date"], as_index=False).sum(numeric_only=True)
    return out.sort_values(["gpsno", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 风险事件
# ---------------------------------------------------------------------------
def build_daily_events(
    sources: List[Source], cfg: Config, chunksize: int = 2_000_000
) -> pd.DataFrame:
    """把风险事件聚合成 ``(gpsno, date)`` 的事件计数表。"""
    if not sources:
        raise FileNotFoundError("未发现风险事件数据（列签名需含 event_type/start_time）")

    night_hours = set(int(h) for h in cfg.temporal["night_hours"])
    dawn_hours = set(int(h) for h in cfg.temporal["dawn_hours"])
    ultra = set(cfg.family_codes("ultra"))

    parts: List[pd.DataFrame] = []
    for src in sources:
        logger.info("聚合风险事件: %s", src.label)
        for chunk in src.iter_chunks(chunksize=chunksize):
            if "gpsno" not in chunk.columns or "event_type" not in chunk.columns:
                continue
            chunk["gpsno"] = to_gpsno_str(chunk["gpsno"])
            tcol = "start_time" if "start_time" in chunk.columns else None
            if tcol is None:
                continue
            ts = parse_datetime(chunk[tcol], cfg.timezone)
            chunk = chunk.assign(ts=ts).dropna(subset=["gpsno", "ts"])
            chunk["event_type"] = pd.to_numeric(chunk["event_type"], errors="coerce")
            chunk = chunk.dropna(subset=["event_type"])
            if not len(chunk):
                continue
            chunk["event_type"] = chunk["event_type"].astype(int)
            chunk = chunk.assign(
                date=to_naive_local_date(chunk["ts"]),
                hour=chunk["ts"].dt.hour,
                n_events=1,
                n_events_night=chunk["ts"].dt.hour.isin(night_hours).astype(int),
                n_events_dawn=chunk["ts"].dt.hour.isin(dawn_hours).astype(int),
                n_ultra=chunk["event_type"].isin(ultra).astype(int),
            )

            pivot = (
                chunk.pivot_table(
                    index=["gpsno", "date"],
                    columns="event_type",
                    values="n_events",
                    aggfunc="sum",
                    fill_value=0,
                )
                .add_prefix("c_")
                .reset_index()
            )
            base = (
                chunk.groupby(["gpsno", "date"], as_index=False)[
                    ["n_events", "n_events_night", "n_events_dawn", "n_ultra"]
                ].sum()
            )
            parts.append(base.merge(pivot, on=["gpsno", "date"], how="outer"))

    if not parts:
        raise ValueError("风险事件聚合后为空，请检查列名与时间列")

    out = pd.concat(parts, ignore_index=True)
    num_cols = [c for c in out.columns if c not in ("gpsno", "date")]
    out[num_cols] = out[num_cols].fillna(0)
    out = out.groupby(["gpsno", "date"], as_index=False)[num_cols].sum()
    return out.sort_values(["gpsno", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# IMU
# ---------------------------------------------------------------------------
def _estimate_vehicle_frame(acc: np.ndarray) -> tuple:
    """估计设备坐标系 → 车辆坐标系的投影基。

    返回 ``(forward, lateral, vertical)`` 三个单位向量。

    做法（对应出题方参考文档第 2 条注意事项）：

    1. 重力方向 = 整段加速度的均值方向 —— 静止/匀速时加速度主要就是重力，
       因此均值方向近似指向地面，取反为垂直轴。
    2. 车辆前向 = 去除重力分量后，水平面内**方差最大**的方向。对车辆而言，
       纵向（加速/刹车）的加速度方差远大于横向，因此这个假设是合理的。
    3. 横向 = 垂直 × 前向。
    """
    mean_acc = acc.mean(axis=0)
    norm = np.linalg.norm(mean_acc)
    if not np.isfinite(norm) or norm < 1e-6:
        # 退化情形：没有可辨识的重力方向，退回标准轴
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])

    vertical = mean_acc / norm
    resid = acc - np.outer(acc @ vertical, vertical)
    # 水平面内方差最大的方向 = 残差协方差矩阵的主特征向量
    cov = np.cov(resid.T) if resid.shape[0] > 1 else np.eye(3)
    vals, vecs = np.linalg.eigh(cov)
    forward = vecs[:, int(np.argmax(vals))]
    forward = forward - (forward @ vertical) * vertical
    fn = np.linalg.norm(forward)
    if not np.isfinite(fn) or fn < 1e-6:
        forward = np.array([1.0, 0.0, 0.0])
    else:
        forward = forward / fn
    lateral = np.cross(vertical, forward)
    return forward, lateral, vertical


def _adaptive_threshold(x: np.ndarray, k: float = IMU_THRESHOLD_K) -> float:
    """``median + k × MAD`` 自适应阈值（MAD 对异常值稳健）。"""
    x = x[np.isfinite(x)]
    if x.size < 50:
        return float("inf")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad <= 0:
        mad = float(np.std(x)) or 1e-6
    return med + k * 1.4826 * mad


def build_daily_imu(
    sources: List[Source], cfg: Config, chunksize: int = 2_000_000
) -> Optional[pd.DataFrame]:
    """从 IMU 原始信号里挖掘六类风险场景，聚合成 ``(gpsno, date)``。

    这是**提升阶梯**而非保底路径：IMU 是压缩包、只有部分车辆覆盖、且需要姿态
    校准与自适应阈值，一周内做对的风险高。因此本函数在缺少数据时返回 ``None``
    而不是抛错，让保底路径（事件 + 轨迹 + 画像）能独立跑通。

    实现分两遍：第一遍抽样估计姿态与阈值，第二遍全量检测。
    """
    if not sources:
        logger.info("未发现 IMU 数据，跳过（保底路径不依赖 IMU）")
        return None

    accel_cols = ["accel_x", "accel_y", "accel_z"]
    gyro_cols = ["gyro_x", "gyro_y", "gyro_z"]

    # ---------------- 第一遍：抽样估计每车的姿态与阈值 ----------------
    calib: Dict[str, Dict[str, object]] = {}
    for src in sources:
        for chunk in src.iter_chunks(chunksize=chunksize):
            need = set(accel_cols + gyro_cols + ["gpsno"])
            if not need.issubset(chunk.columns):
                continue
            chunk = chunk[[c for c in ["gpsno", "data_time", "accel_x", "accel_y", "accel_z",
                                       "gyro_x", "gyro_y", "gyro_z"] if c in chunk.columns]]
            chunk["gpsno"] = to_gpsno_str(chunk["gpsno"])
            for gid, sub in chunk.groupby("gpsno", sort=False):
                if gid in calib:
                    continue
                arr = sub[accel_cols].apply(pd.to_numeric, errors="coerce").dropna().values
                gyr = sub[gyro_cols].apply(pd.to_numeric, errors="coerce").dropna().values
                if arr.shape[0] < 200:
                    continue
                acc = arr[:IMU_CALIBRATION_SAMPLE]
                forward, lateral, vertical = _estimate_vehicle_frame(acc)
                fwd = (arr @ forward)
                lat = (arr @ lateral)
                # 去除重力分量后的垂直轴波动
                vert = (arr @ vertical)
                calib[gid] = {
                    "forward": forward,
                    "lateral": lateral,
                    "vertical": vertical,
                    "thr_fwd": _adaptive_threshold(fwd),
                    "thr_lat": _adaptive_threshold(lat),
                    "thr_vert": _adaptive_threshold(vert),
                    "thr_gyro_z": _adaptive_threshold(gyr[:, 2]) if gyr.size else float("inf"),
                }
            if len(calib) % 50 == 0:
                logger.info("IMU 姿态/阈值标定中，已完成 %d 台", len(calib))

    if not calib:
        logger.warning("IMU 数据不足以标定任何车辆，跳过")
        return None

    # ---------------- 第二遍：全量检测 ----------------
    parts: List[pd.DataFrame] = []
    for src in sources:
        for chunk in src.iter_chunks(chunksize=chunksize):
            if not set(accel_cols + gyro_cols + ["gpsno"]).issubset(chunk.columns):
                continue
            chunk["gpsno"] = to_gpsno_str(chunk["gpsno"])
            tcol = "data_time" if "data_time" in chunk.columns else None
            if tcol is None:
                continue
            ts = parse_datetime(chunk[tcol], cfg.timezone)
            chunk = chunk.assign(ts=ts).dropna(subset=["gpsno", "ts"])
            if not len(chunk):
                continue
            chunk = chunk.assign(date=to_naive_local_date(chunk["ts"]))

            rows = []
            for gid, sub in chunk.groupby("gpsno", sort=False):
                c = calib.get(gid)
                if c is None:
                    continue
                a = sub[accel_cols].apply(pd.to_numeric, errors="coerce").values
                gyr = sub[gyro_cols].apply(pd.to_numeric, errors="coerce").values
                if a.shape[0] < 3:
                    continue
                fwd = a @ c["forward"]
                lat = a @ c["lateral"]
                vert = a @ c["vertical"]
                gz = gyr[:, 2]

                hb = fwd < -float(c["thr_fwd"])
                ha = fwd > float(c["thr_fwd"])
                # 急转弯与猛打方向共用横向信号，但一个强调持续性、一个强调突发性
                st = (np.abs(lat) > float(c["thr_lat"])) & (np.abs(gz) > float(c["thr_gyro_z"]))
                crash = np.abs(fwd) > CRASH_K * float(c["thr_fwd"]) if np.isfinite(c["thr_fwd"]) else np.zeros(len(a), bool)
                roll = np.abs(vert - np.median(vert)) > ROLLOVER_K * float(c["thr_vert"]) if np.isfinite(c["thr_vert"]) else np.zeros(len(a), bool)

                d = sub["date"]
                rows.append(
                    pd.DataFrame(
                        {
                            "gpsno": gid,
                            "imu_harsh_brake": hb.astype(int),
                            "imu_harsh_accel": ha.astype(int),
                            "imu_sharp_turn": st.astype(int),
                            "imu_swerve": st.astype(int),
                            "imu_suspected_crash": crash.astype(int),
                            "imu_suspected_rollover": roll.astype(int),
                        },
                        index=sub.index,
                    ).assign(date=d)
                )
            if rows:
                df = pd.concat(rows, ignore_index=True)
                parts.append(df.groupby(["gpsno", "date"], as_index=False).sum(numeric_only=True))

    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    num_cols = [c for c in out.columns if c not in ("gpsno", "date")]
    out = out.groupby(["gpsno", "date"], as_index=False)[num_cols].sum()
    # 用「有 IMU 记录」的天数作为覆盖度证据：即使当天没有事件也要记一行
    return out.sort_values(["gpsno", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------------
def save_daily(df: Optional[pd.DataFrame], path: Path) -> Optional[Path]:
    if df is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_daily(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path)
