"""06 · 把 docs/ 下的说明文档导出为 PDF（赛题提交物要求 PDF 格式）。

用 reportlab 自带的中日韩字体 ``STSong-Light`` 渲染中文，避免额外字体依赖。
解析的是本项目自己产出的 Markdown 子集（标题 / 段落 / 列表 / 引用 / 表格），
不追求通用性 —— 通用 Markdown 渲染器的引入成本远高于本项目所需。

用法::

    python scripts/06_docs.py [--docs docs] [--out docs/pdf]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from _common import get_config, setup_logging

CJK_FONT = "STSong-Light"


def _register_font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    try:
        pdfmetrics.registerFont(UnicodeCIDFont(CJK_FONT))
        return CJK_FONT
    except Exception:  # pragma: no cover
        return "Helvetica"


def _styles(font: str):
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm

    return {
        "title": ParagraphStyle("title", fontName=font, fontSize=20, leading=26, spaceAfter=6 * mm),
        "h1": ParagraphStyle("h1", fontName=font, fontSize=16, leading=22, spaceBefore=5 * mm,
                             spaceAfter=3 * mm),
        "h2": ParagraphStyle("h2", fontName=font, fontSize=13, leading=18, spaceBefore=4 * mm,
                             spaceAfter=2 * mm),
        "h3": ParagraphStyle("h3", fontName=font, fontSize=11.5, leading=16, spaceBefore=3 * mm,
                             spaceAfter=2 * mm),
        "body": ParagraphStyle("body", fontName=font, fontSize=10, leading=15, spaceAfter=2 * mm),
        "quote": ParagraphStyle("quote", fontName=font, fontSize=9.5, leading=14,
                                leftIndent=6 * mm, textColor="#444444", spaceAfter=2 * mm),
        "bullet": ParagraphStyle("bullet", fontName=font, fontSize=10, leading=15,
                                 leftIndent=6 * mm, bulletIndent=2 * mm, spaceAfter=1 * mm),
        "cell": ParagraphStyle("cell", fontName=font, fontSize=8.5, leading=11),
        "cellh": ParagraphStyle("cellh", fontName=font, fontSize=8.5, leading=11, textColor="#ffffff"),
    }


def _escape(text: str) -> str:
    """reportlab 的 Paragraph 用 XML 解析，必须转义；同时保留 **粗体** 语义。"""
    t = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", t)
    return t


def _split_row(line: str):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def markdown_to_flowables(md: str, font: str):
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    st = _styles(font)
    flow = []
    lines = md.splitlines()
    i = 0
    first_heading = True

    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            i += 1
            continue

        # 表格
        if line.lstrip().startswith("|") and i + 1 < len(lines) and set(lines[i + 1].strip()) <= set("|-: "):
            header = _split_row(line)
            rows = []
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            data = [[Paragraph(_escape(c), st["cellh"]) for c in header]]
            for r in rows:
                r = (r + [""] * len(header))[: len(header)]
                data.append([Paragraph(_escape(c), st["cell"]) for c in r])
            tbl = Table(data, repeatRows=1, hAlign="LEFT")
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f4858")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9aa5ad")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f7")]),
                    ]
                )
            )
            flow.append(tbl)
            flow.append(Spacer(1, 3 * mm))
            continue

        if line.startswith("# "):
            flow.append(Paragraph(_escape(line[2:]), st["title"] if first_heading else st["h1"]))
            first_heading = False
        elif line.startswith("## "):
            flow.append(Paragraph(_escape(line[3:]), st["h1"]))
        elif line.startswith("### "):
            flow.append(Paragraph(_escape(line[4:]), st["h2"]))
        elif line.startswith("#### "):
            flow.append(Paragraph(_escape(line[5:]), st["h3"]))
        elif line.lstrip().startswith(">"):
            flow.append(Paragraph(_escape(line.lstrip()[1:].strip()), st["quote"]))
        elif re.match(r"^\s*[-*]\s+", line):
            flow.append(Paragraph(_escape(re.sub(r"^\s*[-*]\s+", "", line)), st["bullet"], bulletText="•"))
        elif re.match(r"^\s*\d+\.\s+", line):
            flow.append(Paragraph(_escape(re.sub(r"^\s*\d+\.\s+", "", line)), st["bullet"], bulletText="·"))
        else:
            flow.append(Paragraph(_escape(line), st["body"]))
        i += 1

    return flow


def convert(md_path: Path, pdf_path: Path, font: str) -> bool:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate

    md = md_path.read_text(encoding="utf-8")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title=md_path.stem,
    )
    doc.build(markdown_to_flowables(md, font))
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="导出说明文档 PDF")
    ap.add_argument("--docs", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    setup_logging()
    cfg = get_config()

    docs = Path(args.docs) if args.docs else cfg.resolve("docs")
    out = Path(args.out) if args.out else (docs / "pdf")

    font = _register_font()
    files = sorted(p for p in docs.glob("*.md"))
    if not files:
        print(f"{docs} 下没有 Markdown 文档，跳过。")
        return 0

    ok = 0
    for md in files:
        target = out / (md.stem + ".pdf")
        try:
            convert(md, target, font)
            print(f"  ✓ {md.name} → {target}")
            ok += 1
        except Exception as exc:
            print(f"  ✗ {md.name} 转换失败：{type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"\n完成 {ok}/{len(files)} 份。")
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
