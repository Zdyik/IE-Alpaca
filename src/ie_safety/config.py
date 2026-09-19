"""配置加载与全局工程约定。

本模块是读取 ``configs/base.yml`` 的唯一入口。把它集中在一处的原因有二：

1. **赛题标签口径在第 1 天数据审计后才会最终确定。** 窗口长度、标签事件码、
   降级代理标签全部可配置，口径变更只需改 YAML，不引发代码改写。
2. **工程约定必须一致。** 随机种子、时区、``gpsno`` 的字符串类型、路径解析
   集中定义，避免各脚本各写一套导致复现失败。

设计上刻意不做 schema 校验框架：这是一个一次性比赛项目，配置写错时应尽快
报错而不是被层层包装。所有访问器在缺键时直接抛 ``KeyError``。
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

#: 项目根目录（src/ie_safety/config.py → parents[2]）
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "base.yml"


class Config:
    """``configs/base.yml`` 的轻量包装，附带路径解析与常用快捷访问器。"""

    def __init__(self, data: Dict[str, Any], source_path: Path) -> None:
        self._d = data
        self.source_path = source_path

    # ------------------------------------------------------------------ 基础
    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def raw(self) -> Dict[str, Any]:
        return self._d

    def resolve(self, key: str, ensure: bool = False) -> Path:
        """把 ``paths`` 段里的相对路径解析为项目根下的绝对路径。"""
        rel = self._d["paths"][key]
        p = (PROJECT_ROOT / rel).resolve()
        if ensure:
            p.mkdir(parents=True, exist_ok=True)
        return p

    def ensure_dirs(self) -> None:
        for key in ("raw", "interim", "processed", "outputs", "reports", "docs", "figures"):
            if key in self._d.get("paths", {}):
                self.resolve(key, ensure=True)

    # ------------------------------------------------------------ project 段
    @property
    def seed(self) -> int:
        return int(self._d["project"]["seed"])

    @property
    def timezone(self) -> str:
        return str(self._d["project"]["timezone"])

    # ---------------------------------------------------------- windowing 段
    @property
    def feature_days(self) -> int:
        return int(self._d["windowing"]["feature_days"])

    @property
    def outcome_days(self) -> int:
        return int(self._d["windowing"]["outcome_days"])

    @property
    def dev_origins(self) -> List[int]:
        return [int(x) for x in self._d["windowing"]["dev_origins"]]

    @property
    def vault_origin(self) -> int:
        return int(self._d["windowing"]["vault_origin"])

    @property
    def observation_days(self) -> int:
        return int(self._d["data"]["observation_days"])

    # ------------------------------------------------------------- labels 段
    @property
    def accident_codes(self) -> List[int]:
        return [int(x) for x in self._d["labels"]["accident_codes"]]

    @property
    def near_miss_codes(self) -> List[int]:
        return [int(x) for x in self._d["labels"]["near_miss_codes"]]

    @property
    def proxy_enabled(self) -> bool:
        return bool(self._d["labels"].get("proxy_enabled", False))

    @property
    def proxy_extra_codes(self) -> List[int]:
        return [int(x) for x in self._d["labels"].get("proxy_extra_codes", [])]

    def label_codes(self) -> List[int]:
        """返回当前生效的标签事件码集合。

        默认 = 事故 + 未遂事故。若审计判定事件过于稀疏而启用代理标签，
        则并入 ``proxy_extra_codes``（前碰撞预警）。降级决策写入
        ``reports/label_decision.json``，文档必须显式说明。
        """
        codes = list(self.accident_codes)
        if bool(self._d["labels"].get("include_near_miss", True)):
            codes += self.near_miss_codes
        if self.proxy_enabled:
            codes += self.proxy_extra_codes
        return sorted(set(codes))

    # ------------------------------------------------------------- 事件族
    @property
    def families(self) -> Dict[str, Dict[str, Any]]:
        return self._d["event_families"]

    def family_codes(self, family: str) -> List[int]:
        return [int(x) for x in self._d["event_families"][family]["codes"]]

    def all_family_names(self) -> List[str]:
        return list(self._d["event_families"].keys())

    @property
    def compliance_codes(self) -> List[int]:
        return [int(x) for x in self._d["compliance"]["codes"]]

    @property
    def code_catalog(self) -> Dict[int, Dict[str, Any]]:
        return {int(k): v for k, v in self._d["code_catalog"].items()}

    def code_weight(self, code: int) -> float:
        entry = self.code_catalog.get(int(code))
        return float(entry["weight"]) if entry else 1.0

    def code_name(self, code: int) -> str:
        entry = self.code_catalog.get(int(code))
        return str(entry["name"]) if entry else f"未知事件码{code}"

    def code_to_family(self) -> Dict[int, str]:
        """事件码 → 族名。注意族之间**不互斥**，此处只用于特征聚合。"""
        out: Dict[int, str] = {}
        for fam, spec in self._d["event_families"].items():
            for c in spec["codes"]:
                out.setdefault(int(c), fam)
        return out

    # ------------------------------------------------------------- 其它段
    @property
    def exposure(self) -> Dict[str, Any]:
        return self._d["exposure"]

    @property
    def temporal(self) -> Dict[str, Any]:
        return self._d["temporal"]

    @property
    def censoring(self) -> Dict[str, Any]:
        return self._d["censoring"]

    @property
    def validation(self) -> Dict[str, Any]:
        return self._d["validation"]

    @property
    def min_effect_delta_auc(self) -> float:
        return float(self._d["validation"]["min_effect_delta_auc"])

    @property
    def max_comparisons(self) -> int:
        return int(self._d["validation"]["max_comparisons"])

    @property
    def models(self) -> Dict[str, Any]:
        return self._d["models"]

    @property
    def scoring(self) -> Dict[str, Any]:
        return self._d["scoring"]

    @property
    def audit(self) -> Dict[str, Any]:
        return self._d["audit"]

    # ------------------------------------------------------------- 序列化
    def dump(self) -> str:
        return yaml.safe_dump(self._d, allow_unicode=True, sort_keys=False)


def load_config(path: Optional[Path | str] = None) -> Config:
    """读取配置文件并返回 :class:`Config`。"""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"找不到配置文件: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data, p)


def set_global_seed(seed: int) -> None:
    """固定所有可见随机源。

    ``PYTHONHASHSEED`` 必须在解释器启动前设置才生效，因此这里只做提醒式写入
    （对已启动进程无影响），真正的可复现性依赖脚本内固定种子与稳定排序。
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy 必然存在，仅为防御
        pass


def seed_everything(cfg: Config) -> int:
    """按配置固定种子，返回实际使用的种子值（便于写入报告）。"""
    set_global_seed(cfg.seed)
    try:  # LightGBM 的确定性依赖它的 seed，但该包可能尚未安装
        import lightgbm as lgb  # noqa: F401

        os.environ.setdefault("LIGHTGBM_SEED", str(cfg.seed))
    except Exception:
        pass
    return cfg.seed
