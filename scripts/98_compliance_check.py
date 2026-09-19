"""98 · 合规校验：提交前扫描待入库文件，命中即中止。

## 为什么需要自动校验

赛题明文规定「不得泄露任何数据中涉及的信息」，而本仓库是**公开**的。三份赛题
PDF 里写着数据集的下载链接与提取码，结果 CSV 里是逐台车的 ID 与预测值 —— 这些
一旦推上公开仓库就构成泄露。

人工检查会漏。所以把规则写成脚本，并挂进 `run_all.ps1` 的最后一步。

命中任一规则即**拒绝通过**（退出码非 0），必须清理后重跑。

用法::

    python scripts/98_compliance_check.py
    python scripts/98_compliance_check.py --staged     # 只看 git 暂存区
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from _common import get_config, setup_logging

ROOT = Path(__file__).resolve().parents[1]

#: 禁止入库的文件扩展名
FORBIDDEN_SUFFIXES = {".pdf", ".parquet", ".zip", ".7z", ".gz", ".xlsx", ".xls"}

#: 禁止入库的路径前缀（相对仓库根）
FORBIDDEN_PREFIXES = ("data/", "outputs/", ".venv/", ".pkgs/", ".tools/", ".tmp/")

#: 内容级规则：(名称, 正则)
CONTENT_RULES: List[Tuple[str, str]] = [
    ("腾讯云 COS 分享域名（数据访问入口）", r"cosbrowser\.cloud\.tencent\.com"),
    ("COS 分享 token 参数", r"token=[A-Za-z0-9%+/=]{30,}"),
    ("数据 Bucket 名", r"bigdata-dlc-\d+"),
    ("提取码字样", r"提取码"),
    ("疑似车辆 ID 长数字串（>=7 位）", r"\b9\d{8,}\b"),
    ("Windows 绝对路径", r"[A-Za-z]:\\\\?(?:Users|desktop|python)"),
    ("本机用户名", r"HONOR"),
    ("疑似 GitHub/云厂商密钥", r"(gh[pousr]_[A-Za-z0-9]{20,}|AKID[A-Za-z0-9]{20,})"),
]

#: 允许出现「长数字」的白名单文件（文档里的年份、版本号等）
CONTENT_WHITELIST = {
    "README.md",
    "requirements.txt",
    "configs/base.yml",
    "scripts/98_compliance_check.py",
}


def _git_files(staged_only: bool) -> List[str]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"] if staged_only else [
        "git", "ls-files"
    ]
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
        return [f for f in out.stdout.splitlines() if f.strip()]
    except Exception:
        # 尚未 git init 时，退化为扫描工作区（排除被 .gitignore 覆盖的目录）
        skip = {".git", ".venv", ".pkgs", ".tools", ".tmp", "data", "outputs", "__pycache__"}
        files = []
        for p in ROOT.rglob("*"):
            if p.is_file() and not any(part in skip for part in p.relative_to(ROOT).parts):
                files.append(str(p.relative_to(ROOT)).replace("\\", "/"))
        return files


def check_file(rel: str) -> List[str]:
    problems: List[str] = []
    p = ROOT / rel
    if not p.exists():
        return problems

    if p.suffix.lower() in FORBIDDEN_SUFFIXES:
        problems.append(f"禁止入库的扩展名 {p.suffix}")
    for pref in FORBIDDEN_PREFIXES:
        if rel.replace("\\", "/").startswith(pref):
            problems.append(f"禁止入库的路径前缀 {pref}")

    if p.suffix.lower() not in {".md", ".py", ".yml", ".yaml", ".txt", ".ps1", ".json", ".csv", ".toml", ".cfg", ""}:
        return problems
    if p.stat().st_size > 2 * 1024 * 1024:
        problems.append("文件过大（>2MB），可能误入数据文件")
        return problems

    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return problems

    whitelisted = rel in CONTENT_WHITELIST or rel.startswith("docs/") and "01_数据审计" not in rel
    for name, pattern in CONTENT_RULES:
        for m in re.finditer(pattern, text):
            # 文档里的示例数字允许，但真实车辆 ID 不允许：只对长数字串做行内容判断
            if name.startswith("疑似车辆 ID"):
                line = text[: m.start()].splitlines()[-1] if m.start() else ""
                if whitelisted and ("示例" in line or "例" in line or "9500000" in line):
                    continue
            problems.append(f"{name}：命中 `{m.group(0)[:40]}`")
            break
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="入库合规校验")
    ap.add_argument("--staged", action="store_true", help="只检查 git 暂存区")
    args = ap.parse_args(argv)
    setup_logging()

    files = _git_files(args.staged)
    print(f"检查 {len(files)} 个待入库文件 ...")

    failed = 0
    for rel in files:
        problems = check_file(rel)
        if problems:
            failed += 1
            print(f"\n✗ {rel}")
            for pr in problems:
                print(f"    - {pr}")

    if failed:
        print(
            f"\n合规校验未通过：{failed} 个文件命中规则。\n"
            "赛题规定不得泄露数据中涉及的信息，而本仓库是公开的。请移除上述文件/内容后重跑。"
        )
        return 1

    print("\n✓ 合规校验通过：未发现数据文件、赛题 PDF、车辆级结果或密钥。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
