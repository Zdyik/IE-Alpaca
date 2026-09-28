"""Create the formal PDF explanation for a completed V12 task-two score run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)


REPO = Path(__file__).resolve().parent
DEFAULT_DB = Path(r"E:\Databases\2026 清华IE亮剑-算法赛道赛题")
FONT = "MicrosoftYaHei"
FONT_BOLD = "MicrosoftYaHeiBold"
NAVY = colors.HexColor("#15334A")
BLUE = colors.HexColor("#277DA1")
TEAL = colors.HexColor("#43AA8B")
AMBER = colors.HexColor("#F9C74F")
ORANGE = colors.HexColor("#F8961E")
RED = colors.HexColor("#E45756")
LIGHT = colors.HexColor("#F2F6F8")
MID = colors.HexColor("#D7E2E8")
TEXT = colors.HexColor("#24343D")


def register_fonts() -> None:
    regular = Path(r"C:\Windows\Fonts\msyh.ttc")
    bold = Path(r"C:\Windows\Fonts\msyhbd.ttc")
    if not regular.exists() or not bold.exists():
        raise FileNotFoundError("Microsoft YaHei fonts are required to render the Chinese report")
    pdfmetrics.registerFont(TTFont(FONT, str(regular), subfontIndex=0))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(bold), subfontIndex=0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paragraph(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text), style)


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName=FONT_BOLD,
                                fontSize=24, leading=34, textColor=NAVY, alignment=TA_LEFT,
                                spaceAfter=12),
        "subtitle": ParagraphStyle("subtitle", fontName=FONT, fontSize=11, leading=18,
                                   textColor=BLUE, spaceAfter=18),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName=FONT_BOLD,
                             fontSize=16, leading=23, textColor=NAVY, spaceBefore=8,
                             spaceAfter=8, keepWithNext=True),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName=FONT_BOLD,
                             fontSize=12.5, leading=19, textColor=BLUE, spaceBefore=6,
                             spaceAfter=5, keepWithNext=True),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName=FONT,
                               fontSize=9.3, leading=16, textColor=TEXT, spaceAfter=6),
        "small": ParagraphStyle("small", fontName=FONT, fontSize=7.7, leading=12,
                                textColor=TEXT),
        "table": ParagraphStyle("table", fontName=FONT, fontSize=7.4, leading=10.5,
                                textColor=TEXT),
        "table_bold": ParagraphStyle("table_bold", fontName=FONT_BOLD, fontSize=7.4,
                                     leading=10.5, textColor=colors.white,
                                     alignment=TA_CENTER),
        "callout": ParagraphStyle("callout", fontName=FONT_BOLD, fontSize=10.2,
                                  leading=17, textColor=NAVY, leftIndent=7, rightIndent=7,
                                  spaceBefore=4, spaceAfter=4),
        "metric": ParagraphStyle("metric", fontName=FONT_BOLD, fontSize=15,
                                 leading=18, alignment=TA_CENTER, textColor=NAVY),
        "metric_label": ParagraphStyle("metric_label", fontName=FONT, fontSize=7.8,
                                       leading=11, alignment=TA_CENTER, textColor=TEXT),
    }


def table(data: list[list[object]], widths: list[float], sty: dict[str, ParagraphStyle],
          aligns: dict[int, str] | None = None, repeat_rows: int = 1) -> Table:
    rows = []
    for r, values in enumerate(data):
        row = []
        for value in values:
            row.append(paragraph(value, sty["table_bold"] if r == 0 else sty["table"]))
        rows.append(row)
    result = Table(rows, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), .35, MID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for column, align in (aligns or {}).items():
        commands.append(("ALIGN", (column, 1), (column, -1), align))
    result.setStyle(TableStyle(commands))
    return result


def metric_cards(values: list[tuple[str, str]], sty: dict[str, ParagraphStyle]) -> Table:
    content = []
    for value, label in values:
        content.append(Table([[paragraph(value, sty["metric"])],
                              [paragraph(label, sty["metric_label"])]],
                             colWidths=[42 * mm], rowHeights=[10 * mm, 10 * mm],
                             style=TableStyle([
                                 ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                                 ("BOX", (0, 0), (-1, -1), .6, MID),
                                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                             ])))
    return Table([content], colWidths=[44 * mm] * len(content), hAlign="LEFT",
                 style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))


def grade_chart(fleet: pd.DataFrame) -> Drawing:
    palette = {"A": TEAL, "B": colors.HexColor("#90BE6D"), "C": AMBER,
               "D": ORANGE, "E": RED, "U": colors.HexColor("#8D99AE")}
    width, height = 475, 150
    drawing = Drawing(width, height)
    values = {str(row.grade): int(row.vehicles) for row in fleet.itertuples()}
    maximum = max(values.values())
    x0, y0, bar_width, gap, plot_height = 42, 24, 48, 20, 102
    drawing.add(String(2, 132, "车辆数", fontName=FONT, fontSize=7, fillColor=TEXT))
    for index, grade_name in enumerate(("A", "B", "C", "D", "E", "U")):
        value = values.get(grade_name, 0)
        bar_height = plot_height * value / maximum
        x = x0 + index * (bar_width + gap)
        drawing.add(Rect(x, y0, bar_width, bar_height, fillColor=palette[grade_name], strokeColor=None))
        drawing.add(String(x + bar_width / 2, y0 - 13, grade_name, textAnchor="middle",
                           fontName=FONT_BOLD, fontSize=8, fillColor=TEXT))
        drawing.add(String(x + bar_width / 2, y0 + bar_height + 4, str(value), textAnchor="middle",
                           fontName=FONT_BOLD, fontSize=8, fillColor=NAVY))
    return drawing


class NumberedDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs: object) -> None:
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=self.header_footer))

    @staticmethod
    def header_footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(MID)
        canvas.setLineWidth(.5)
        canvas.line(doc.leftMargin, A4[1] - 15 * mm, A4[0] - doc.rightMargin, A4[1] - 15 * mm)
        canvas.setFont(FONT, 7.3)
        canvas.setFillColor(colors.HexColor("#607480"))
        canvas.drawString(doc.leftMargin, A4[1] - 11.5 * mm, "IE亮剑 · 任务二安全评价模型")
        canvas.drawRightString(A4[0] - doc.rightMargin, A4[1] - 11.5 * mm, "V12 State-MAE Scorecard")
        canvas.line(doc.leftMargin, 14 * mm, A4[0] - doc.rightMargin, 14 * mm)
        canvas.drawString(doc.leftMargin, 9.5 * mm, "开发集历史 OOF 验证；不代表官方测试成绩")
        canvas.drawRightString(A4[0] - doc.rightMargin, 9.5 * mm, f"第 {doc.page} 页")
        canvas.restoreState()


def resolve_run(run_dir: Path | None, database_root: Path) -> Path:
    if run_dir is not None:
        return run_dir.resolve()
    latest = json.loads((database_root / "任务二安全评价结果" / "latest.json").read_text(encoding="utf-8"))
    return Path(latest["run_dir"]).resolve()


def build_report(run_dir: Path, output: Path) -> None:
    register_fonts()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    deliverables = run_dir / "deliverables"
    fleet = pd.read_csv(deliverables / "task2_fleet_summary.csv", encoding="utf-8-sig")
    events = pd.read_csv(deliverables / "task2_event_weights.csv", encoding="utf-8-sig")
    outcomes = pd.read_csv(run_dir / "diagnostics" / "oof_grade_outcomes.csv", encoding="utf-8-sig")
    cards = pd.read_csv(deliverables / "task2_driver_scorecards.csv", encoding="utf-8-sig")
    history = pd.read_csv(deliverables / "task2_daily_score_history.csv", encoding="utf-8-sig")
    if len(cards) != 500 or cards.gpsno.nunique() != 500 or len(history) != 23500:
        raise ValueError("task2 run has incomplete score outputs")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    sty = styles()
    doc = NumberedDocTemplate(str(output), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm,
                              topMargin=21 * mm, bottomMargin=19 * mm,
                              title="任务二安全评价模型说明", author="IE-Alpaca")
    story: list[object] = []
    story += [Spacer(1, 8 * mm), paragraph("任务二安全评价模型说明", sty["title"]),
              paragraph("基于 V12 人类先验引导驾驶状态表征 · 版本化可复现交付", sty["subtitle"]),
              HRFlowable(width="100%", thickness=2, color=BLUE, spaceBefore=1, spaceAfter=13)]
    anchor = metrics["anchor20_oof"]
    story.append(metric_cards([
        ("500", "正式评分车辆"),
        (f"{anchor['safety_score_risk_auc']:.4f}", "历史 OOF 风险 AUC"),
        (f"{metrics['score_mean']:.2f}", "车队平均安全分"),
        (str(metrics["grade_counts"].get("U", 0)), "证据不足车辆"),
    ], sty))
    story += [Spacer(1, 6 * mm), paragraph("执行摘要", sty["h1"]),
              paragraph("本模型复用 V12 State-MAE 学到的驾驶状态和未来 40 天事故/未遂风险，把综合概率、五类可解释行为状态与数据可信度转换为 0–100 分。每一分扣分均可追溯到风险分位、具体事件和固定公式；低证据车辆单列 U 级，避免把设备离线解释为安全。", sty["body"]),
              paragraph("最终安全分在 5.25–92.50 之间。历史 20→40 天严格 OOF 检验中，100−安全分的 ROC-AUC 为 0.7767；A 到 E 级未来事件率从 9.6% 单调上升到 69.6%。第 60 天之后没有本地真值，正式得分的官方表现只能由赛方评测确认。", sty["body"]),
              paragraph(f"运行编号：{manifest['run_id']}　源模型：{manifest['source_task1_run_id']}　评分设备：{manifest.get('device', 'unknown')}", sty["small"]),
              PageBreak()]

    story += [paragraph("1　设计目标与整体结构", sty["h1"]),
              paragraph("赛题要求模型能区分风险、解释得分、支持运营并动态更新。本方案把这些要求落到同一条数据链：", sty["body"])]
    flow = [["步骤", "实际处理", "输出"],
            ["① 驾驶状态表征", "V12 将逐日记录编码为暴露、危险行为、控制、近端险情、严重历史、观测质量六种状态", "未来 40 天风险概率"],
            ["② 事件重要性学习", "逐个屏蔽 24 类事件，测量 V12 概率变化，并向均匀权重收缩", "状态内事件权重"],
            ["③ 可解释计分", "六项风险映射到冻结开发参考分布，再按固定上限扣分", "0–100 安全分"],
            ["④ 运营分层", "按分数与证据置信度划分 A/B/C/D/E/U", "管理动作和复评节奏"],
            ["⑤ 动态更新", "每天仅使用当日及之前记录重算", "日/周/月趋势"]]
    story += [table(flow, [21*mm, 104*mm, 50*mm], sty), Spacer(1, 5*mm),
              paragraph("任务一概率是评分主轴，行为项提供可解释修正。暴露量不直接扣分，只用于把事件次数换成每千公里、每百小时风险，避免高里程司机因驾驶机会多而被机械处罚。", sty["body"]),
              paragraph("2　特征工程", sty["h1"]),
              paragraph("每个事件分别保留累计次数、每千公里次数、每百小时次数、事件活跃天占比以及最近 14 天相对过去的上升幅度。极端值经 log1p 压缩。车辆控制项额外吸收加速度峰值、加速度波动和角速度峰值；轨迹和 IMU 覆盖率只作为证据质量，不把缺失填成正常驾驶。", sty["body"]),
              paragraph("证据置信度 = 0.60×是否存在事件源 + 0.30×轨迹覆盖率 + 0.10×IMU覆盖率。置信度低时输出 U 级并暂缓奖惩。", sty["callout"]),
              PageBreak()]

    points = metrics["max_deduction_points"]
    dimension_rows = [["维度", "最大扣分", "输入证据", "管理含义"],
                      ["V12未来风险", f"{points['model']:.0f}", "六状态时序表征输出的未来40天概率", "综合排序主轴"],
                      ["危险驾驶行为", f"{points['upstream']:.0f}", "超速、疲劳、分心、看手机等", "长期行为倾向"],
                      ["车辆控制", f"{points['control']:.0f}", "车道偏移、压线和IMU异常", "操作稳定性"],
                      ["近端险情", f"{points['proximal']:.0f}", "前碰撞预警、车距过近", "事故前近端信号"],
                      ["事故与未遂历史", f"{points['history']:.0f}", "评分日前事故、未遂事故", "已经显现的严重风险"],
                      ["管理配合度", f"{points['quality']:.0f}", "摄像头遮挡、角度扭转", "证据质量与设备管理"]]
    story += [paragraph("3　权重与计分公式", sty["h1"]),
              paragraph("对每种事件 e，将该事件的输入清零并重新推理。V12 概率变化的平均绝对值作为事件灵敏度。状态内最终权重由 50% 均匀权重与 50% 灵敏度权重组成，防止稀有事件在小样本中偶然占据过高权重。", sty["body"]),
              paragraph("事件权重：wₑ = 0.5/|G(e)| + 0.5·sₑ/Σsⱼ", sty["callout"]),
              table(dimension_rows, [35*mm, 19*mm, 73*mm, 48*mm], sty, {1: "CENTER"}),
              Spacer(1, 4*mm),
              paragraph("每项原始风险先转换为开发车辆第 60 天经验百分位 r。最终公式为：", sty["body"]),
              paragraph("安全分 = 100 − Σ 最大扣分ᵈ × (风险百分位ᵈ)²", sty["callout"]),
              paragraph("平方项让中低风险小幅扣分，把管理注意力集中到同类车辆中的高风险尾部。六项最大扣分相加为 100，分项扣分之和能逐车精确还原总分。", sty["body"]),
              paragraph("4　模型学到的事件重要性", sty["h1"])]
    top = (events.sort_values(["state", "learned_weight_within_state"], ascending=[True, False])
           .groupby("state", observed=False).head(2))
    state_cn = {"upstream": "危险行为", "control": "车辆控制", "proximal": "近端险情",
                "history": "事故历史", "quality": "管理配合", "exposure": "暴露环境"}
    event_rows = [["状态", "事件", "活跃车辆", "状态内权重", "屏蔽后平均|Δ概率|"]]
    for row in top.itertuples(index=False):
        event_rows.append([state_cn[str(row.state)], f"{int(row.event_code)} {row.event_name}",
                           str(int(row.active_vehicles)), f"{row.learned_weight_within_state:.3f}",
                           f"{row.mean_abs_probability_delta:.4f}"])
    story += [table(event_rows, [28*mm, 54*mm, 25*mm, 29*mm, 39*mm], sty,
                    {2: "CENTER", 3: "RIGHT", 4: "RIGHT"}), PageBreak()]

    grade_rows = [["等级", "规则", "风险解释", "运营动作"],
                  ["A", "≥85", "低风险", "常规月度复盘，可纳入安全激励"],
                  ["B", "75–84.99", "较低风险", "每周反馈主要扣分项"],
                  ["C", "65–74.99", "中风险", "针对性培训，两周后复评"],
                  ["D", "50–64.99", "高风险", "24小时内人工复核、专项培训和重点监控"],
                  ["E", "<50", "极高风险", "立即人工核查，暂停高风险任务并制定整改计划"],
                  ["U", "证据置信度低", "证据不足", "先修复数据链路，补齐前暂缓奖惩"]]
    story += [paragraph("5　等级、动态更新与运营闭环", sty["h1"]),
              table(grade_rows, [17*mm, 30*mm, 29*mm, 99*mm], sty), Spacer(1, 5*mm),
              paragraph("模型从第 14 天开始每日更新，共输出每车 47 个时间点。周报取每周最后一天，月报取月末。每辆车还输出最大行为扣分维度和贡献最大的三个具体事件，使培训内容直接对应行为原因。", sty["body"]),
              paragraph("建议运营顺序：先处理 U 级设备问题；再对 E/D 级人工复核并制定整改；C 级开展针对性培训；A/B 级采用正向反馈。任何纪律或奖惩决定均保留人工复核。", sty["body"]),
              paragraph("6　最终车队分布", sty["h1"]), grade_chart(fleet)]
    fleet_rows = [["等级", "车辆数", "占比", "平均分", "平均V12风险"]]
    for row in fleet.itertuples(index=False):
        fleet_rows.append([row.grade, str(int(row.vehicles)), f"{row.share:.1%}",
                           f"{row.mean_score:.2f}", f"{row.mean_v12_risk:.3f}"])
    story += [table(fleet_rows, [22*mm, 28*mm, 28*mm, 40*mm, 47*mm], sty,
                    {1: "CENTER", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT"}), PageBreak()]

    outcome_rows = [["历史等级", "车辆数", "后40天事件数", "后40天事件率", "平均安全分"]]
    for row in outcomes.itertuples(index=False):
        outcome_rows.append([row.grade, str(int(row.vehicles)), str(int(row.future_events)),
                             f"{row.future_event_rate:.1%}", f"{row.mean_safety_score:.2f}"])
    story += [paragraph("7　风险区分能力与验证边界", sty["h1"]),
              paragraph(f"使用第20天的严格 OOF V12 概率和截至第20天的行为，预测随后40天事故/未遂。共 {anchor['vehicles']} 辆开发车、{anchor['positive']} 个正例。100−安全分的 ROC-AUC 为 {anchor['safety_score_risk_auc']:.4f}，原 V12 概率 AUC 为 {anchor['v12_probability_auc']:.4f}，风险排名与结局秩相关为 {anchor['rank_correlation_with_future_event']:.4f}。", sty["body"]),
              table(outcome_rows, [27*mm, 29*mm, 36*mm, 39*mm, 39*mm], sty,
                    {1: "CENTER", 2: "CENTER", 3: "RIGHT", 4: "RIGHT"}),
              Spacer(1, 5*mm),
              paragraph("A→E 的历史事件率总体单调上升，说明分数保留了风险排序能力。正式评分日期为 2026-07-30，此后40天没有本地结果，因此不能把上述历史 OOF 指标称为正式预测准确率。96辆任务一锁定车标签未开启。", sty["body"]),
              paragraph("主要限制", sty["h2"]),
              paragraph("① 样本只有约500辆，事件权重仍可能受稀有事件影响；② 经验百分位反映当前车队的相对风险，换车队后需要重新校准参考分布；③ 模型给出统计风险和管理优先级，不替代事故调查或人工纪律决定；④ U级75分只是中性占位，必须与正常B级分开。", sty["body"]),
              PageBreak()]

    files = [
        ["文件", "用途"],
        ["task2_driver_safety_scores.csv", "500辆车编号与安全分；最简提交表"],
        ["task2_driver_scorecards.csv", "六项分位、扣分、等级、原因和管理动作"],
        ["task2_daily_score_history.csv", "第14–60天每日动态分数，共23,500行"],
        ["task2_event_weights.csv", "24类事件的学习灵敏度和状态内权重"],
        ["task2_fleet_summary.csv", "等级分布和平均风险"],
        ["task2_v12_safety_model_report.pdf", "模型、公式、验证与管理建议说明"],
    ]
    story += [paragraph("8　复现与交付清单", sty["h1"]),
              paragraph("评分脚本读取已冻结的 V12 State-MAE 权重和任务一预处理日表，在数据库中创建全新的版本目录。源码快照、配置、模型哈希、参考分布、诊断结果和交付表保存在同一运行目录。", sty["body"]),
              table(files, [72*mm, 103*mm], sty), Spacer(1, 6*mm),
              paragraph("复现命令", sty["h2"]),
              paragraph("python score_task2_v12.py --config configs/experiments/task2_v12_scorecard.json", sty["callout"]),
              paragraph("python generate_task2_report.py --run-dir &lt;任务二运行目录&gt;", sty["callout"]),
              paragraph(f"模型哈希：{manifest['source_model_sha256']}<br/>配置哈希：{manifest['config_sha256']}<br/>报告生成时间：{datetime.now(timezone.utc).isoformat()}", sty["small"]),
              Spacer(1, 8*mm),
              HRFlowable(width="100%", thickness=1, color=MID, spaceAfter=8),
              paragraph("结论：V12 安全评价模型在不读取锁定标签的前提下，把未来风险概率转换成透明、动态、可运营的安全分。历史 OOF 结果支持其风险排序能力；正式泛化能力以赛方评测为准。", sty["callout"])]

    doc.build(story)

    source_dir = run_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), source_dir / Path(__file__).name)
    design = REPO / "docs" / "任务二V12安全评价模型设计.md"
    if design.exists():
        destination = source_dir / "docs" / design.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(design, destination)

    manifest.update({
        "status": "completed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "report_sha256": sha256(output),
        "output_files": sorted(str(path.relative_to(run_dir)) for path in deliverables.glob("*")),
    })
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--database-root", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = resolve_run(args.run_dir, args.database_root)
    output = (args.output.resolve() if args.output else
              run_dir / "deliverables" / "task2_v12_safety_model_report.pdf")
    build_report(run_dir, output)
    print(output)


if __name__ == "__main__":
    main()
