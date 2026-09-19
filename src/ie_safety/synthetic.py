"""合成数据生成器 —— 让整条流水线在**没有真实数据**时也能被端到端验证。

## 为什么需要它

赛题数据需要从腾讯云 COS 分享链接手动获取，在拿到之前，代码是**完全未经验证**
的。未经验证的代码交上去，等于把「能不能跑」这件事赌在运气上。因此这里按赛题
文档给出的字段规范生成一份**schema 完全一致**的合成数据，用途有三：

1. **冒烟测试**：证明读取 → 审计 → 特征 → 建模 → 评分 → 报告这条链路真的通。
2. **负对照验证**：合成数据有**已知的生成过程**，因此可以验证「打乱标签后 AUC
   必须回到 0.5」这类断言确实成立 —— 在真实数据上无法预先知道正确答案，但在
   合成数据上可以。
3. **接口冻结**：真实数据到位后只换输入，不改代码。

## 生成过程（刻意做成「有信号但不完美」）

* 每台车有一个潜在风险倾向 ``z ~ N(0, 1)``；
* 风险倍率 ``m = exp(0.55 z)``，同时驱动**前 20 天的特征窗口**与**后 40 天的结果
  窗口** —— 所以特征对标签确实有预测力（AUC 应显著高于 0.5）；
* 事件计数服从泊松分布，均值 = 族基准频次 × 风险倍率 × 暴露量因子；
* 超高危事件额外乘一个 ``m`` 的幂，让「历史出险」成为强预测因子；
* 结果窗口的标签由同一 ``z`` 加噪声决定，**因此 AUC 有上限而不是 1.0** ——
  这正是我们想要的：一个「信号充足但非完美」的可验证场景。

> ⚠️ **合成数据上的任何数字都不是比赛结果。** 它只能说明代码能跑、协议自洽。
> 报告与文档中必须显式标注这一点。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# 23 类事件码及其族归属（必须与 configs/base.yml 保持一致）
FAMILY_CODES: Dict[str, list] = {
    "ultra": [11803, 11804],
    "collision": [30000, 30005, 30017, 30002, 30003],
    "speed": [11401, 11402, 11403, 11405, 11406],
    "fatigue": [41001, 41029, 41002, 41009],
    "distraction": [41003, 41004, 41023, 41005],
    "blindspot": [60292, 60294],
}
COMPLIANCE_CODES = [41006, 41021]

CODE_NAMES = {
    11803: "事故", 11804: "未遂事故", 30000: "前碰撞预警", 30005: "车距过近",
    30017: "长时间压线", 30002: "左车道偏移", 30003: "右车道偏移",
    11401: "路口超速", 11402: "主路弯道超速", 11403: "危险路段超速报警",
    11405: "匝道超速", 11406: "高速长下坡超速", 41001: "疲劳(闭眼)", 41029: "困倦",
    41002: "打哈欠", 41009: "频繁低头", 41003: "注意力分散", 41004: "打电话",
    41023: "看手机", 41005: "抽烟", 60292: "后盲区报警", 60294: "右盲区报警",
    41006: "摄像头遮挡", 41021: "摄像头角度扭转",
}

# 各族「每车每日」的事件基准频次。刻意让疲劳/分心/超速 ≫ 跟车/盲区 ≫ 超高危，
# 从而满足 audit/anchors.py 里的频次序关系锚点。
FAMILY_DAILY_RATE = {
    "fatigue": 2.4,
    "distraction": 1.6,
    "speed": 1.1,
    "collision": 0.35,
    "blindspot": 0.12,
    "ultra": 0.05,
}

ENERGY_TYPES = ["电力", "柴油", "天然气"]


@dataclass
class SyntheticSpec:
    n_vehicles: int = 120
    n_days: int = 60
    feature_days: int = 20
    outcome_days: int = 40
    traj_points_per_day: int = 48
    imu_vehicle_frac: float = 0.2
    imu_days: int = 6
    imu_hz_samples_per_day: int = 1440
    seed: int = 42
    start_date: str = "2026-06-01"


def _gpsno(i: int) -> str:
    # 刻意保留前导零的写法，用于验证读取层的字符串化逻辑
    return f"{9500000 + i * 7919:09d}"


def _poisson(rng: np.random.Generator, lam: float) -> int:
    return int(rng.poisson(max(lam, 0.0)))


def generate(spec: SyntheticSpec, out_dir: Path) -> Dict[str, Path]:
    """生成四个数据集并落盘，返回 ``{kind: path}``。"""
    rng = np.random.default_rng(spec.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpsnos = [_gpsno(i) for i in range(spec.n_vehicles)]
    z = rng.normal(0.0, 1.0, size=spec.n_vehicles)
    risk_mult = np.exp(0.55 * z)

    base_km = rng.uniform(120.0, 420.0, size=spec.n_vehicles)          # 日均里程
    energy = rng.choice(ENERGY_TYPES, size=spec.n_vehicles, p=[0.35, 0.5, 0.15])
    night_frac = np.clip(rng.normal(0.14, 0.07, size=spec.n_vehicles), 0.0, 0.6)
    highway_frac = np.clip(rng.normal(0.28, 0.14, size=spec.n_vehicles), 0.0, 0.9)

    dates = pd.date_range(spec.start_date, periods=spec.n_days, freq="D")

    # ------------------------------------------------------------------
    # 1. 车辆画像（近半年月均；比率字段刻意写成带百分号的字符串，
    #    因为赛题文档的示例就是 "4.21%"，而字段声明却是 double）
    # ------------------------------------------------------------------
    profile = pd.DataFrame(
        {
            "gpsno": gpsnos,
            "能源类型": energy,
            "月均行驶里程": np.round(base_km * 30 * rng.normal(1.0, 0.08, spec.n_vehicles), 2),
            "月均行驶时长": np.round(base_km * 30 / rng.uniform(38, 55, spec.n_vehicles), 2),
            "月平均停留次数": np.round(rng.uniform(15, 60, spec.n_vehicles), 2),
            "高速里程占比": [f"{v * 100:.2f}%" for v in highway_frac],
            "早晨行驶时长占比": [f"{np.clip(rng.normal(0.18, 0.05), 0, 1) * 100:.2f}%" for _ in gpsnos],
            "黄昏行驶时长占比": [f"{np.clip(rng.normal(0.24, 0.06), 0, 1) * 100:.2f}%" for _ in gpsnos],
            "夜间里程占比": [f"{v * 100:.2f}%" for v in night_frac],
            "夜间行驶时长占比": [f"{v * 100:.2f}%" for v in night_frac],
        }
    )
    profile_path = out_dir / "vehicle_profile.csv"
    profile.to_csv(profile_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 2. 风险事件（事件级）
    # ------------------------------------------------------------------
    ev_rows = []
    veh_idx = {g: i for i, g in enumerate(gpsnos)}
    for di, day in enumerate(dates):
        for gi, g in enumerate(gpsnos):
            m = risk_mult[gi]
            exposure = base_km[gi] / 250.0  # 相对暴露量，长里程车事件更多
            # 巡航参数：夜间比例高的车，夜间事件更多
            for fam, codes in FAMILY_CODES.items():
                lam = FAMILY_DAILY_RATE[fam] * m * exposure
                if fam == "ultra":
                    lam *= m  # 超高危事件对潜在风险更敏感（复发效应）
                n = _poisson(rng, lam)
                if n <= 0:
                    continue
                for _ in range(n):
                    code = int(rng.choice(codes))
                    if rng.random() < night_frac[gi]:
                        hour = int(rng.integers(22, 24)) if rng.random() < 0.6 else int(rng.integers(2, 6))
                    else:
                        hour = int(rng.integers(6, 22))
                    ts = day + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 60)))
                    ev_rows.append(
                        (
                            g, float(23.1 + rng.normal(0, 0.3)), float(113.2 + rng.normal(0, 0.3)),
                            code, CODE_NAMES.get(code, str(code)),
                            int(np.clip(rng.normal(62, 18), 10, 120)),
                            ts.strftime("%Y-%m-%d %H:%M:%S"),
                        )
                    )
            # 合规类事件（摄像头遮挡/角度异常）：与风险无关，但与「数据可见性」有关
            if rng.random() < 0.02:
                code = int(rng.choice(COMPLIANCE_CODES))
                hour = int(rng.integers(0, 24))
                ts = day + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
                ev_rows.append(
                    (g, float(23.1), float(113.2), code, CODE_NAMES[code],
                     int(np.clip(rng.normal(50, 15), 0, 120)), ts.strftime("%Y-%m-%d %H:%M:%S"))
                )

    events = pd.DataFrame(
        ev_rows, columns=["gpsno", "lat", "lng", "event_type", "event_name", "speed", "start_time"]
    )
    events_path = out_dir / "risk_events.csv"
    events.to_csv(events_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 3. 轨迹（轨迹点级）
    # ------------------------------------------------------------------
    traj_rows = []
    step_km = spec.traj_points_per_day
    for di, day in enumerate(dates):
        for gi, g in enumerate(gpsnos):
            # 每 5 天里有 1 天停驶，制造覆盖度差异
            if rng.random() < 0.2:
                continue
            day_km = base_km[gi] * rng.normal(1.0, 0.2)
            for k in range(step_km):
                hour = int(k * 24 / step_km)
                is_night = hour >= 22 or hour <= 5
                spd = float(np.clip(rng.normal(72 if highway_frac[gi] > 0.3 else 48, 16), 0, 110))
                seg_km = max(day_km, 1.0) / step_km
                run_time = int(np.clip(seg_km / max(spd, 5.0) * 3600, 5, 900))
                ts = day + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
                traj_rows.append(
                    (
                        g, int(seg_km * 100000), run_time, ts.strftime("%Y-%m-%d %H:%M:%S"),
                        int(spd), int(rng.integers(0, 360)),
                        float(23.1 + rng.normal(0, 0.2)), float(113.2 + rng.normal(0, 0.2)),
                    )
                )
                _ = is_night
    trajectory = pd.DataFrame(
        traj_rows,
        columns=["gpsno", "distance", "run_time", "trigger_time", "speed", "course", "lat", "lng"],
    )
    trajectory_path = out_dir / "trajectory.csv"
    trajectory.to_csv(trajectory_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 4. IMU（只覆盖部分车辆、部分时段 —— 刻意复现赛题的「部分车辆不覆盖」）
    # ------------------------------------------------------------------
    n_imu = max(2, int(spec.n_vehicles * spec.imu_vehicle_frac))
    imu_vehicles = list(rng.choice(gpsnos, size=n_imu, replace=False))
    imu_rows = []
    g_vec = np.array([0.0, 0.0, 1.0])          # 真值重力方向（车辆坐标系）
    # 每台车的设备安装姿态不同：绕 z 转一个随机角，再叠一点倾斜
    for g in imu_vehicles:
        gi = veh_idx[g]
        theta = float(rng.uniform(0, 2 * np.pi))
        tilt = float(rng.normal(0, 0.12))
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        tilt_m = np.array(
            [[1.0, 0.0, 0.0], [0.0, np.cos(tilt), -np.sin(tilt)], [0.0, np.sin(tilt), np.cos(tilt)]]
        )
        mount = rot @ tilt_m

        for di in range(min(spec.imu_days, spec.n_days)):
            day = dates[di]
            n = spec.imu_hz_samples_per_day
            t = np.arange(n)
            # 基础信号：重力 + 轻微路面噪声
            acc = np.tile(g_vec, (n, 1)) + rng.normal(0, 0.02, size=(n, 3))
            gyro = rng.normal(0, 0.05, size=(n, 3))
            # 注入急刹车：前向强负向脉冲（约 1g）
            n_brake = _poisson(rng, 1.5 * risk_mult[gi])
            for _ in range(n_brake):
                s = int(rng.integers(100, n - 60))
                acc[s : s + 25, 0] -= 0.85
            # 注入急加速
            n_acc = _poisson(rng, 1.2 * risk_mult[gi])
            for _ in range(n_acc):
                s = int(rng.integers(100, n - 60))
                acc[s : s + 20, 0] += 0.55
            # 注入急转弯：横向 + 航向角速度同时变化（双约束）
            n_turn = _poisson(rng, 0.8 * risk_mult[gi])
            for _ in range(n_turn):
                s = int(rng.integers(100, n - 120))
                acc[s : s + 60, 1] += 0.40
                gyro[s : s + 60, 2] += 0.9
            # 疑似侧翻：垂直轴大幅波动
            if rng.random() < 0.05 * risk_mult[gi]:
                s = int(rng.integers(100, n - 120))
                acc[s : s + 80, 2] += 1.6

            # 把车辆坐标系的信号投影到设备坐标系（模拟真实安装姿态）
            acc_dev = acc @ mount.T
            gyro_dev = gyro @ mount.T
            ts = day + pd.to_timedelta(t, unit="s")
            imu_rows.append(
                pd.DataFrame(
                    {
                        "gpsno": g,
                        "imei": f"16102059{g[-6:]}",
                        "data_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "data_date": ts.strftime("%Y%m%d"),
                        "ems_speed": np.clip(rng.normal(60, 15, n), 0, 120).round(1),
                        "gps_speed": np.clip(rng.normal(59, 15, n), 0, 120).round(1),
                        "accel_x_raw": acc_dev[:, 0].round(4),
                        "accel_y_raw": acc_dev[:, 1].round(4),
                        "accel_z_raw": acc_dev[:, 2].round(4),
                        "gyro_x_raw": gyro_dev[:, 0].round(4),
                        "gyro_y_raw": gyro_dev[:, 1].round(4),
                        "gyro_z_raw": gyro_dev[:, 2].round(4),
                    }
                )
            )
    imu = pd.concat(imu_rows, ignore_index=True) if imu_rows else pd.DataFrame()
    imu_path = out_dir / "imu.csv"
    imu.to_csv(imu_path, index=False, encoding="utf-8-sig")

    # 生成真值文件（仅合成场景有；真实数据没有，用于验证负对照）
    truth = pd.DataFrame(
        {
            "gpsno": gpsnos,
            "latent_z": z,
            "risk_multiplier": risk_mult,
            "true_daily_km": base_km,
        }
    )
    truth_path = out_dir / "_ground_truth.csv"
    truth.to_csv(truth_path, index=False, encoding="utf-8-sig")

    return {
        "profile": profile_path,
        "events": events_path,
        "trajectory": trajectory_path,
        "imu": imu_path,
        "truth": truth_path,
    }


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="生成 schema 一致的合成赛题数据")
    ap.add_argument("--out", default="data/raw/synthetic")
    ap.add_argument("--vehicles", type=int, default=120)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    spec = SyntheticSpec(n_vehicles=args.vehicles, n_days=args.days, seed=args.seed)
    paths = generate(spec, Path(args.out))
    print("合成数据已生成：")
    for k, v in paths.items():
        print(f"  {k:11s} {v}")
    print("\n注意：这是**合成数据**，任何在其上得到的指标都不代表比赛结果。")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
