"""数据读取、列名归一与类型规整。

三个必须防御的现实问题：

1. **列名不确定。** 赛题文档用的是英文字段名（``gpsno`` / ``event_type``），但
   表格里的中文名是「设备号」「事件类型编码」。真实文件到底用哪套无法预知，
   因此这里用**别名表 + 列签名**做识别，而不是硬编码文件名与列名。
2. **文件名不确定。** 四个数据集的实际文件名未给出，因此 ``discover_datasets``
   按**列签名**扫描 ``data/raw``（含 zip 内部成员）来归类。
3. **编码不确定。** 国内物流数据常见 GBK/GB18030，这里按 ``utf-8-sig → gbk``
   顺序试读。

另外，``gpsno`` 是四表 join 键，必须全程字符串化：它可能带前导零，也可能被
读成 float（``9542389.0``），两种都会导致 join 静默失败。
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# 列名别名表：canonical -> 可能的写法
# ---------------------------------------------------------------------------
CANONICAL_ALIASES: Dict[str, List[str]] = {
    # 通用
    "gpsno": ["gpsno", "设备号", "设备编号", "gps_no"],
    # 车辆画像
    "energy_type": ["能源类型", "energy_type", "能源"],
    "month_km": ["月均行驶里程", "month_km", "月均里程", "avg_month_km"],
    "month_hours": ["月均行驶时长", "month_hours", "月均时长", "avg_month_hours"],
    "month_stops": ["月平均停留次数", "month_stops", "月均停留次数"],
    "highway_km_ratio": ["高速里程占比", "highway_km_ratio"],
    "morning_hours_ratio": ["早晨行驶时长占比", "morning_hours_ratio"],
    "dusk_hours_ratio": ["黄昏行驶时长占比", "dusk_hours_ratio"],
    "night_km_ratio": ["夜间里程占比", "night_km_ratio"],
    "night_hours_ratio": ["夜间行驶时长占比", "night_hours_ratio"],
    # 风险事件
    "lat": ["lat", "开始纬度", "纬度", "latitude"],
    "lng": ["lng", "开始经度", "经度", "longitude", "lon"],
    "event_type": ["event_type", "事件类型编码", "事件类型", "event_code"],
    "event_name": ["event_name", "事件名称", "事件名"],
    "speed": ["speed", "速度", "事件速度"],
    "start_time": ["start_time", "事件开始时间", "开始时间", "发生时间"],
    # IMU
    "imei": ["imei", "设备串号"],
    "data_time": ["data_time", "数据时间"],
    "data_date": ["data_date", "数据日期"],
    "ems_speed": ["ems_speed", "ems速度", "ems_spe"],
    "gps_speed": ["gps_speed", "gps速度", "gps_spe"],
    "accel_x": ["accel_x_raw", "accel_x", "加速度x", "加速度X"],
    "accel_y": ["accel_y_raw", "accel_y", "加速度y", "加速度Y"],
    "accel_z": ["accel_z_raw", "accel_z", "加速度z", "加速度Z"],
    "gyro_x": ["gyro_x_raw", "gyro_x", "角速度x", "角速度X"],
    "gyro_y": ["gyro_y_raw", "gyro_y", "角速度y", "角速度Y"],
    "gyro_z": ["gyro_z_raw", "gyro_z", "角速度z", "角速度Z"],
    # 轨迹
    "distance": ["distance", "里程", "行驶距离"],
    "run_time": ["run_time", "运行时长", "行驶时长"],
    "trigger_time": ["trigger_time", "数据产生时间", "采集时间"],
    "course": ["course", "航向", "航向角"],
}

# 各类数据集的列签名：命中越多越可信
KIND_SIGNATURES: Dict[str, set] = {
    "events": {"event_type", "event_name", "start_time"},
    "trajectory": {"distance", "run_time", "course"},
    "imu": {"accel_x", "accel_y", "accel_z", "gyro_x", "gyro_z"},
    "profile": {"energy_type", "month_km", "month_hours", "night_km_ratio"},
}

#: 归类所需的最低命中数
MIN_SIGNATURE_HITS = 2

DATA_SUFFIXES = {".csv", ".txt", ".tsv", ".parquet", ".xlsx", ".xls"}
ARCHIVE_SUFFIXES = {".zip"}


def _norm_key(s: Any) -> str:
    """把列名压成比较用的规范键：小写、去掉空格/下划线/连字符/全角括号。"""
    t = str(s).strip().lower()
    for ch in (" ", "_", "-", "\u3000", "(", ")", "（", "）", "\ufeff", '"', "'"):
        t = t.replace(ch, "")
    return t


_ALIAS_INDEX: Dict[str, str] = {}
for _canon, _aliases in CANONICAL_ALIASES.items():
    for _a in _aliases:
        _ALIAS_INDEX.setdefault(_norm_key(_a), _canon)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把列名映射为规范名；未识别的列保持原名。"""
    mapping: Dict[str, str] = {}
    used: set = set()
    for col in df.columns:
        canon = _ALIAS_INDEX.get(_norm_key(col))
        if canon and canon not in used:
            mapping[col] = canon
            used.add(canon)
    return df.rename(columns=mapping)


# ---------------------------------------------------------------------------
# 类型规整
# ---------------------------------------------------------------------------
def to_gpsno_str(series: pd.Series) -> pd.Series:
    """``gpsno`` 一律字符串化，并抹掉 ``.0`` 尾巴与空白。"""
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)
    s = s.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
    return s.replace({"nan": None, "None": None, "": None})


def parse_datetime(series: pd.Series, tz: Optional[str] = None) -> pd.Series:
    """解析时间列并统一时区（默认按配置的 ``Asia/Shanghai``）。"""
    dt = pd.to_datetime(series, errors="coerce")
    if tz:
        dt = dt.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    return dt


def to_naive_local_date(series: pd.Series) -> pd.Series:
    """把时间列压成**本地朴素日期**（丢掉时区，但保留本地墙钟时间）。

    这是本项目的时间不变量。理由：聚合表要在窗口之间反复比较，而
    ``tz-aware`` 与 ``tz-naive`` 混用会在 pandas 里引出成片的
    ``Invalid comparison between dtype=datetime64[us] and Timestamp``，
    且极易在 ``.values`` 转换处被静默降级（tz 会被丢掉并当成 UTC）。
    统一成朴素本地日期后，整类问题消失。
    """
    dt = pd.to_datetime(series, errors="coerce")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return dt.dt.normalize()


def naive_ts(ts) -> pd.Timestamp:
    """标量版本：把任意时间戳转成朴素本地日期（零点）。"""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t.normalize()


def parse_ratio(series: pd.Series) -> pd.Series:
    """解析比率字段。

    赛题文档的示例值是 ``4.21%``、``18.41%`` 这类**带百分号的字符串**，而字段
    声明却是 ``double``。因此这里同时处理三种写法：``"4.21%"``、``4.21``（视为
    百分数）、``0.0421``（视为小数）。判据是：去掉百分号后若量级 > 1 则除以 100。
    """
    if series.dtype.kind in "if":
        num = series.astype(float)
        return num.where(num.abs() <= 1.0, num / 100.0)

    s = series.astype(str).str.strip()
    had_pct = s.str.contains("%", na=False)
    num = pd.to_numeric(s.str.replace("%", "", regex=False), errors="coerce")
    # 带百分号的：数值即百分数，一律 /100（"4.21%" -> 0.0421）
    # 不带的：先原样保留，仅在量级 > 1 时按百分数处理（4.21 -> 0.0421；0.0421 不变）
    out = num.where(~had_pct, other=num / 100.0)
    out = pd.Series(out, index=series.index)
    big = (~had_pct) & (num.abs() > 1.0)
    out = out.where(~big, num / 100.0)
    return out


# ---------------------------------------------------------------------------
# 文件发现
# ---------------------------------------------------------------------------
@dataclass
class Source:
    """一个可读的数据来源（可能是普通文件，也可能是 zip 内部成员）。"""

    path: Path
    member: Optional[str] = None
    kind: str = ""
    columns: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.path.name}::{self.member}" if self.member else self.path.name

    def iter_chunks(self, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
        """分块产出数据，列名已归一。"""
        if self.path.suffix.lower() == ".parquet":
            yield normalize_columns(pd.read_parquet(self.path))
            return

        if self.path.suffix.lower() in {".xlsx", ".xls"}:
            yield normalize_columns(pd.read_excel(self.path))
            return

        raw: Any
        if self.member:
            with zipfile.ZipFile(self.path) as zf:
                with zf.open(self.member) as fh:
                    raw = fh.read()
            for chunk in _read_text_chunks(raw, chunksize):
                yield normalize_columns(chunk)
            return

        for chunk in _read_text_chunks(self.path, chunksize):
            yield normalize_columns(chunk)

    def read(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """整体读入（``nrows`` 仅对文本文件生效）。"""
        if nrows is not None and self.path.suffix.lower() in {".csv", ".txt", ".tsv"} and not self.member:
            return normalize_columns(_read_text(self.path, nrows=nrows))
        parts = list(self.iter_chunks())
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        return df if nrows is None else df.head(nrows)


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _read_text(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    head = path.read_bytes() if nrows is None else path.open("rb").read()
    text = _decode(head)
    sep = "\t" if path.suffix.lower() == ".tsv" else None
    return pd.read_csv(
        io.StringIO(text),
        sep=sep,
        engine="python",
        nrows=nrows,
        dtype={"gpsno": str},
        low_memory=False,
    )


def _read_text_chunks(src: Any, chunksize: int) -> Iterator[pd.DataFrame]:
    if isinstance(src, bytes):
        text = _decode(src)
        reader = pd.read_csv(
            io.StringIO(text), engine="python", chunksize=chunksize, dtype={"gpsno": str}
        )
    else:
        # 大文件走 C 引擎，但先用少量字节探测编码
        with open(src, "rb") as fh:
            probe = fh.read(65536)
        enc = "utf-8-sig"
        for cand in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                probe.decode(cand)
                enc = cand
                break
            except UnicodeDecodeError:
                continue
        reader = pd.read_csv(
            src,
            engine="c",
            chunksize=chunksize,
            encoding=enc,
            dtype={"gpsno": str},
            low_memory=False,
        )
    for chunk in reader:
        yield chunk


def _classify(columns: Iterable[Any]) -> tuple:
    """按列签名给数据集归类，返回 ``(kind, hits)``。"""
    norm = {_norm_key(c) for c in columns}
    canon = {_ALIAS_INDEX.get(c, "") for c in norm}
    canon.discard("")
    best_kind, best_hits = "", 0
    for kind, sig in KIND_SIGNATURES.items():
        hits = len(sig & canon)
        if hits > best_hits:
            best_kind, best_hits = kind, hits
    return best_kind, best_hits


def discover_datasets(raw_dir: Path, probe_rows: int = 5) -> Dict[str, List[Source]]:
    """扫描 ``raw_dir``（含 zip 成员），按列签名归类四个数据集。

    返回 ``{"profile": [...], "events": [...], "trajectory": [...], "imu": [...]}``。
    未识别或命中不足的文件会被忽略，但可通过返回值中的 ``?`` 键查看。
    """
    found: Dict[str, List[Source]] = {
        "profile": [],
        "events": [],
        "trajectory": [],
        "imu": [],
        "?": [],
    }
    if not raw_dir.exists():
        return found

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()

        if suffix in ARCHIVE_SUFFIXES:
            try:
                with zipfile.ZipFile(path) as zf:
                    for name in zf.namelist():
                        if name.endswith("/") or Path(name).suffix.lower() not in DATA_SUFFIXES:
                            continue
                        with zf.open(name) as fh:
                            head = _decode(fh.read(65536))
                        try:
                            cols = pd.read_csv(io.StringIO(head), nrows=probe_rows, engine="python").columns
                        except Exception:
                            continue
                        src = Source(path=path, member=name, columns=[str(c) for c in cols])
                        src.kind, hits = _classify(cols)
                        (found[src.kind] if hits >= MIN_SIGNATURE_HITS else found["?"]).append(src)
            except zipfile.BadZipFile:
                continue
            continue

        if suffix not in DATA_SUFFIXES:
            continue

        try:
            if suffix == ".parquet":
                cols = list(pd.read_parquet(path).columns)
            elif suffix in {".xlsx", ".xls"}:
                cols = list(pd.read_excel(path, nrows=probe_rows).columns)
            else:
                with open(path, "rb") as fh:
                    head = _decode(fh.read(65536))
                cols = list(pd.read_csv(io.StringIO(head), nrows=probe_rows, engine="python").columns)
        except Exception:
            continue

        src = Source(path=path, columns=[str(c) for c in cols])
        src.kind, hits = _classify(cols)
        (found[src.kind] if hits >= MIN_SIGNATURE_HITS else found["?"]).append(src)

    return found


# ---------------------------------------------------------------------------
# 面向业务的高层读取器
# ---------------------------------------------------------------------------
def load_profile(sources: List[Source]) -> pd.DataFrame:
    """读取车辆画像（500 行级别，可整表读入）。"""
    if not sources:
        raise FileNotFoundError("未发现车辆画像数据（列签名需含 能源类型/月均行驶里程 等）")
    frames = []
    for src in sources:
        df = src.read()
        if "gpsno" not in df.columns:
            continue
        df["gpsno"] = to_gpsno_str(df["gpsno"])
        for col in (
            "highway_km_ratio",
            "morning_hours_ratio",
            "dusk_hours_ratio",
            "night_km_ratio",
            "night_hours_ratio",
        ):
            if col in df.columns:
                df[col] = parse_ratio(df[col])
        if "energy_type" in df.columns:
            df["energy_type"] = df["energy_type"].astype(str).str.strip()
        frames.append(df)
    if not frames:
        raise ValueError("车辆画像文件中找不到 gpsno 列")
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["gpsno"], keep="first")


def load_events(sources: List[Source], tz: Optional[str] = None) -> pd.DataFrame:
    """读取风险事件（事件级，量级可控，整表读入）。"""
    if not sources:
        raise FileNotFoundError("未发现风险事件数据（列签名需含 event_type/start_time）")
    frames = []
    for src in sources:
        for chunk in src.iter_chunks(chunksize=1_000_000):
            if "gpsno" not in chunk.columns or "event_type" not in chunk.columns:
                continue
            chunk["gpsno"] = to_gpsno_str(chunk["gpsno"])
            chunk["event_type"] = pd.to_numeric(chunk["event_type"], errors="coerce").astype("Int64")
            if "start_time" in chunk.columns:
                chunk["ts"] = parse_datetime(chunk["start_time"], tz)
            frames.append(chunk)
    if not frames:
        raise ValueError("风险事件文件中找不到 gpsno / event_type 列")
    ev = pd.concat(frames, ignore_index=True)
    ev = ev.dropna(subset=["gpsno", "event_type"])
    ev["event_type"] = ev["event_type"].astype(int)
    return ev


def iter_events(sources: List[Source], chunksize: int = 1_000_000) -> Iterator[pd.DataFrame]:
    """分块产出风险事件，供聚合型消费方使用。"""
    for src in sources:
        for chunk in src.iter_chunks(chunksize=chunksize):
            if "gpsno" not in chunk.columns or "event_type" not in chunk.columns:
                continue
            chunk["gpsno"] = to_gpsno_str(chunk["gpsno"])
            chunk["event_type"] = pd.to_numeric(chunk["event_type"], errors="coerce")
            yield chunk.dropna(subset=["gpsno", "event_type"])
