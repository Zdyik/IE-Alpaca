"""文本读写工具：跨平台确定性换行符。

## 为什么需要这个小模块

Python 的 ``open(path, "w")`` 在 Windows 上会把 ``\\n`` 翻译成 ``\\r\\n``。这意味着
同一份脚本在 Windows 与 Linux 上产出的报告文件**字节不同**，而本仓库的
``.gitattributes`` 声明了 ``* text=auto eol=lf``。

后果不只是「diff 难看」：PR 的 patch 系列在应用时的归一化结果会与本地仓库
不一致（本地 CRLF、应用后 LF），于是「patch 能否忠实复现仓库」这件事就无法验证。
这个问题就是被 `git am` 的端到端验证抓出来的。

因此所有**生成型产出**（报告、安全卡、JSON、CSV 摘要）一律走本模块，显式写 ``\\n``。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def write_text_lf(path: PathLike, text: str) -> Path:
    """以 UTF-8 + LF 写入文本，并确保父目录存在。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return p


def append_text_lf(path: PathLike, text: str) -> Path:
    """以 UTF-8 + LF 追加文本。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return p


def write_json_lf(path: PathLike, obj: Any, indent: int = 2) -> Path:
    """以 UTF-8 + LF 写 JSON（``ensure_ascii=False``，中文保持可读）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=str)
    return p


def read_text(path: PathLike) -> str:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()
