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
#:
#: 设计原则：规则要匹配**真实的数据/密钥**，而不是「描述这些规则的文字」。
#: 否则合规脚本自身、以及解释合规要求的文档都会误报，最后没人再信任这个检查。
CONTENT_RULES: List[Tuple[str, str]] = [
    ("腾讯云 COS 分享链接（含 id 参数）", r"share/\?id=[A-Za-z0-9]{16,}"),
    ("腾讯云 COS 分享域名（数据访问入口）", r"cosbrowser\.cloud\.tencent\.com"),
    ("COS 分享 token 参数", r"token=[A-Za-z0-9%+/=]{30,}"),
    ("数据 Bucket 名", r"bigdata-dlc-\d+"),
    ("提取码（形如「提取码: ab12cd」）", r"提取码\s*[:：=]?\s*[0-9a-zA-Z]{6}\b"),
    ("疑似车辆 ID 长数字串（>=9 位）", r"\b9\d{8,}\b"),
    ("Windows 绝对路径中的本机用户名", r"[A-Za-z]:\\+Users\\+[A-Za-z0-9_.\-]+"),
    ("疑似 GitHub / 云厂商密钥", r"(gh[pousr]_[A-Za-z0-9]{20,}|AKID[A-Za-z0-9]{20,})"),
]

#: 允许出现 COS **域名**（但不允许分享链接）的文件。
#: `00_probe_share.py` 需要用公开的分享站 API 查询配置；知道域名并不等于获得数据
#: 访问权，真正的凭据是分享 id 与提取码，它们由上面第一、五条规则单独拦截。
DOMAIN_ALLOWED = {"scripts/00_probe_share.py"}

#: 允许出现「长数字」的白名单文件（版本号、示例 ID 等）
CONTENT_WHITELIST = {
    "README.md",
    "requirements.txt",
    "configs/base.yml",
    "scripts/98_compliance_check.py",
    "scripts/01b_make_synthetic.py",
    "src/ie_safety/synthetic.py",
}


def _git_files(staged_only: bool) -> List[str]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"] if staged_only else [
        "git", "ls-files"
    ]
    try:
        # 必须显式指定 utf-8：否则 Windows 下 subprocess 会用系统 GBK 解码 git 输出的
        # UTF-8 中文文件名，在读线程里抛 UnicodeDecodeError，stdout 变成 None，
        # 于是静默退化成工作区扫描 —— 那会让校验看起来「跑了」，其实扫错了对象。
        out = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True
        )
        files = [f for f in (out.stdout or "").splitlines() if f.strip()]
        if files:
            return files
    except Exception as exc:
        print(f"[提示] git 文件列表获取失败（{type(exc).__name__}），退化为工作区扫描。")

    # 退化路径：尚未 git init，或 git 不可用
    skip = {".git", ".venv", ".pkgs", ".tools", ".tmp", "data", "outputs",
            "__pycache__", "push-materials", "pdf"}
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

    whitelisted = rel.replace("\\", "/") in CONTENT_WHITELIST
    domain_ok = rel.replace("\\", "/") in DOMAIN_ALLOWED
    for name, pattern in CONTENT_RULES:
        # 「长数字串」这条对白名单文件不适用：合成数据生成器与合规脚本里本来就有
        # 伪造的车辆 ID 示例，它们不是真实数据。
        if name.startswith("疑似车辆 ID") and whitelisted:
            continue
        if name.startswith("腾讯云 COS 分享域名") and domain_ok:
            continue
        # 合规脚本必须描述规则本身，因此它的注释里会写出规则的形状（用的是明显的
        # 假示例 ab12cd）。这条自引用例外只对它一个文件生效。
        if rel.replace("\\", "/") == "scripts/98_compliance_check.py" and name.startswith("提取码"):
            continue
        m = re.search(pattern, text)
        if m:
            problems.append(f"{name}：命中 `{m.group(0)[:40]}`")
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
