"""脚本共用工具：路径解析、数据来源发现、日粒度表加载。"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# 允许直接以 `python scripts/xxx.py` 运行而无需先安装包
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ie_safety.config import Config, load_config  # noqa: E402
from ie_safety.textio import write_json_lf, write_text_lf  # noqa: E402

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")


def get_config(path: Optional[str] = None) -> Config:
    cfg = load_config(path) if path else load_config()
    cfg.ensure_dirs()
    return cfg


def find_raw_dir(cfg: Config, override: Optional[str] = None) -> Path:
    """定位原始数据目录。

    优先用显式传入的路径；否则在 ``data/raw`` 下自动挑选**包含四个数据集**的
    子目录（真实数据与合成数据可以共存，避免误用合成数据）。
    """
    raw = Path(override) if override else cfg.resolve("raw")
    if override:
        return raw
    from ie_safety.io import discover_datasets

    candidates = [raw] + sorted([p for p in raw.iterdir() if p.is_dir()]) if raw.exists() else []
    best, best_score = raw, -1
    for cand in candidates:
        found = discover_datasets(cand)
        score = sum(1 for k in ("profile", "events", "trajectory", "imu") if found.get(k))
        if score > best_score:
            best, best_score = cand, score
    if best_score <= 0:
        raise SystemExit(
            f"在 {raw} 下没有发现任何可识别的数据集。\n"
            "请把赛题数据放入 data/raw/（四个数据集：车辆画像 / 风险事件 / IMU / 轨迹），\n"
            "或运行 `python scripts/01b_make_synthetic.py` 生成合成数据做冒烟验证。"
        )
    return best


def load_daily_tables(cfg: Config) -> Dict[str, object]:
    """加载日粒度聚合表；不存在则提示先跑 02_audit.py。"""
    from ie_safety.features.daily import load_daily

    proc = cfg.resolve("processed")
    tables = {
        "exposure": load_daily(proc / "daily_exposure.parquet"),
        "events": load_daily(proc / "daily_events.parquet"),
        "imu": load_daily(proc / "daily_imu.parquet"),
    }
    if tables["exposure"] is None or tables["events"] is None:
        raise SystemExit(
            "未找到日粒度聚合表（data/processed/daily_*.parquet）。\n"
            "请先运行 `python scripts/02_audit.py`。"
        )
    return tables


def rel_or_abs(p: Path) -> str:
    """尽量返回相对项目根的路径。

    报告与清单会被提交进公开仓库，写绝对路径既无必要、又泄漏本机目录结构，
    因此优先转相对路径；不在项目内时才退回绝对路径。
    """
    try:
        return str(Path(p).resolve().relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(p)


def write_json(path: Path, obj: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_json_lf(path, obj)


def load_json(path: Path) -> Optional[object]:
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def markdown_table(rows, headers=None) -> str:
    """把 ``list[dict]`` 或 ``DataFrame`` 渲染成 markdown 表格。"""
    import pandas as pd

    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        return "（无数据）"
    cols = list(headers) if headers else list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df[cols].iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def synthetic_notice(cfg: Config, raw_dir: Path) -> str:
    """若用的是合成数据，返回一段必须在报告里出现的声明。"""
    if "synthetic" in str(raw_dir).lower():
        return (
            "> ⚠️ **本报告基于合成数据（smoke test），所有数字均不代表比赛结果。**\n"
            "> 合成数据的用途是验证代码链路与验证协议自洽（例如「打乱标签后 AUC 必须回到 0.5」"
            "这类断言在合成数据上可以被验证）。真实结论必须以赛题数据重跑为准。\n"
        )
    return ""
