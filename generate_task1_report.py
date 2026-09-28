"""Create the formal task-one V12 model explanation PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, HRFlowable, PageBreak, PageTemplate, Paragraph, Spacer

from generate_task2_report import (
    AMBER, BLUE, LIGHT, MID, NAVY, ORANGE, RED, TEAL, TEXT, FONT, FONT_BOLD,
    metric_cards, paragraph, register_fonts, styles, table,
)


class TaskOneDoc(BaseDocTemplate):
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
        canvas.drawString(doc.leftMargin, A4[1] - 11.5 * mm, "IE亮剑 · 任务一事故预测")
        canvas.drawRightString(A4[0] - doc.rightMargin, A4[1] - 11.5 * mm, "V12 State-MAE")
        canvas.line(doc.leftMargin, 14 * mm, A4[0] - doc.rightMargin, 14 * mm)
        canvas.drawString(doc.leftMargin, 9.5 * mm, "开发集历史 OOF 验证；不代表官方测试成绩")
        canvas.drawRightString(A4[0] - doc.rightMargin, 9.5 * mm, f"第 {doc.page} 页")
        canvas.restoreState()


def model_chart() -> Drawing:
    values = [("V3线性", .7875, colors.HexColor("#8D99AE")),
              ("V11 Masked", .7741, TEAL), ("V12 State-MAE", .7752, BLUE),
              ("V12 Dynamics", .7761, AMBER), ("V12 HRC", .7709, ORANGE)]
    width, height = 475, 150
    d = Drawing(width, height)
    x0, y0, bw, gap = 34, 25, 62, 27
    for idx, (name, value, color) in enumerate(values):
        h = (value - .70) / .10 * 95
        x = x0 + idx * (bw + gap)
        d.add(Rect(x, y0, bw, h, fillColor=color, strokeColor=None))
        d.add(String(x + bw/2, y0 - 13, name, textAnchor="middle", fontName=FONT, fontSize=7, fillColor=TEXT))
        d.add(String(x + bw/2, y0 + h + 4, f"{value:.4f}", textAnchor="middle",
                     fontName=FONT_BOLD, fontSize=8, fillColor=NAVY))
    d.add(String(2, 132, "20→40 OOF ROC-AUC", fontName=FONT, fontSize=7, fillColor=TEXT))
    return d


def build_report(run_dir: Path, output: Path) -> None:
    register_fonts()
    run_dir, output = run_dir.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    audit = json.loads((run_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    score = metrics["development_oof"]
    sty = styles()
    doc = TaskOneDoc(str(output), pagesize=A4, leftMargin=17*mm, rightMargin=17*mm,
                     topMargin=21*mm, bottomMargin=19*mm,
                     title="任务一事故风险预测模型说明", author="IE-Alpaca")
    story: list[object] = [Spacer(1, 8*mm), paragraph("任务一事故风险预测模型说明", sty["title"]),
        paragraph("V12 State-MAE · 人类先验引导的驾驶状态自监督表征", sty["subtitle"]),
        HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=13),
        metric_cards([("500", "正式预测车辆"), (f"{score['roc_auc']:.4f}", "20→40 OOF AUC"),
                      (f"{score['pr_auc']:.4f}", "OOF PR-AUC"), (f"{score['brier']:.4f}", "OOF Brier")], sty),
        Spacer(1, 6*mm), paragraph("执行摘要", sty["h1"]),
        paragraph("模型使用每辆车2026-06-01至07-30共60天的风险事件、轨迹和IMU记录，输出未来40天至少发生一次事故或未遂事故的概率，并以0.5固定阈值给出二分类。事故口径为11803或11804。", sty["body"]),
        paragraph("V12先把逐日记录学习为暴露机会、上游危险行为、控制稳定性、近端险情、严重历史和证据质量六个状态，再以State-MAE遮蔽恢复进行自监督预训练，最后用右删失首次事故损失学习每日风险率。正式Day60概率覆盖全部500辆车。", sty["body"]),
        paragraph("本地没有第61—100天真值。报告中的0.7752 AUC来自382辆开发车在第20天预测随后40天的严格车辆级OOF回测；96辆锁定车标签没有用于拟合、选择或指标。", sty["body"]),
        paragraph(f"运行编号：{manifest['run_id']}　训练设备：{manifest['device']}　随机种子：{config['seed']}", sty["small"]),
        PageBreak(),
        paragraph("1　任务定义与数据边界", sty["h1"]),
        paragraph("最终目标是 P(第61—100天至少出现一次11803/11804 | 第1—60天历史)。网络先输出近似恒定的每日首次事故风险率 q，再用 p₄₀ = 1−(1−q)⁴⁰ 转成40天概率。", sty["body"]),
    ]
    bounds = [["锚点", "历史长度", "可观察未来", "训练处理"],
              ["14", "14天", "40天", "完整40天监督"], ["20", "20天", "40天", "主要回测锚点"],
              ["30", "30天", "30天", "未观察10天不计损失"], ["40", "40天", "20天", "右删失"],
              ["50", "50天", "10天", "右删失"], ["53", "53天", "7天", "右删失"]]
    story += [table(bounds, [27*mm, 38*mm, 43*mm, 67*mm], sty), Spacer(1, 5*mm),
              paragraph("同一车辆的所有日期和锚点始终位于同一侧。478辆有可靠事件监督，其中382辆为开发池、96辆为锁定组；另22辆没有可靠行为源，在最终预测中使用开发先验并标记为低证据。", sty["body"]),
              paragraph("泄漏审计结果", sty["h2"]),
              paragraph(f"五折验证车辆均只出现一次；训练/验证重叠为0；参考输入截至{audit['reference_input_end']}，标签从{audit['reference_label_start']}开始；未来日期不可被锚点读取；锁定标签未用于拟合或指标。", sty["body"]),
              paragraph("2　特征工程", sty["h1"]),
              paragraph("原始数十GB数据只清洗一次，汇总成500×60车辆日表。事件同时保留触发次数、报警段、每百公里率、每小时率、是否发生和暴露可见性；轨迹提取里程、时长、夜间驾驶、速度与行程；IMU提取旋转不变的加速度峰值、波动和角速度峰值；每个缺失通道另设掩码。", sty["body"])]
    states = [["状态", "主要信息", "作用"], ["O 暴露", "里程、时长、夜间、盲区环境", "区分驾驶机会与单位风险"],
              ["U 上游行为", "超速、疲劳、分心、看手机", "长期危险倾向"],
              ["C 控制", "偏移、压线、IMU动态", "操作稳定性"],
              ["P 近端险情", "前碰撞预警、车距过近", "事故前信号"],
              ["H 严重历史", "事故、未遂事故", "已显现风险"],
              ["Q 证据质量", "摄像头遮挡、角度扭转、缺失", "防止沉默即安全"]]
    story += [table(states, [30*mm, 85*mm, 60*mm], sty), PageBreak(),
              paragraph("3　V12 State-MAE模型架构", sty["h1"])]
    architecture = [["阶段", "结构", "输出"],
                    ["事件编码", "每类事件8项输入→共享MLP；加入事件编号、角色和阶段向量", "24个32维事件表示"],
                    ["六状态聚合", "24×6可学习软归属；人工分组只提供初始化和弱约束", "每天6个32维状态"],
                    ["时间编码", "4层因果TCN，卷积核3，膨胀1/2/4/8，残差和LayerNorm", "仅使用锚点及以前历史"],
                    ["State-MAE", "遮事件包、连续状态块和上下文；EMA教师提供隐藏目标", "无标签驾驶状态表征"],
                    ["风险头", "长期均值、近14天均值、近7天变化→64维隐藏层", "每日风险率q与40天概率"]]
    story += [table(architecture, [28*mm, 105*mm, 42*mm], sty), Spacer(1, 5*mm),
              paragraph("人工判断规定事件大致属于哪种状态，但不固定事件权重。软归属矩阵、事件门控、状态表示和最终风险贡献均由训练数据学习。最终State-MAE有效风险路径约47,871个参数，避免在数百辆监督车辆上使用数千万参数模型。", sty["body"]),
              paragraph("4　自监督目标与事故损失", sty["h1"]),
              paragraph("预训练随机遮住完整事件包、连续2—5天状态块和上下文通道，恢复事件是否发生、正值强度、上下文和EMA隐藏表示。原本缺失的位置不计算误差；事故/未遂不会作为未来SSL目标，防止自监督任务直接偷看监督标签。", sty["body"]),
              paragraph("Lmask = L事件存在 + 0.50L事件强度 + 0.50L上下文 + 0.25LEMA状态 + 0.01L语义弱约束", sty["callout"]),
              paragraph("若锚点后第j天首次出险，L = −(j−1)log(1−q)−log(q)；若观察c天仍未出险，L = −c·log(1−q)。第c+1天以后没有观测，不填成负例。每车多个锚点先平均，再跨车辆平均。", sty["body"]),
              PageBreak(), paragraph("5　训练参数与模型选择", sty["h1"])]
    s = config["model"]
    params = [["类别", "参数", "设置"], ["状态编码", "宽度 / TCN层数", f"{s['width']} / 4"],
              ["正则", "Dropout / weight decay", f"{s['dropout']} / {s['weight_decay']}"],
              ["自监督", "步数 / 批量 / 学习率", f"{s['ssl_steps']} / {s['ssl_batch_size']} / {s['ssl_learning_rate']}"],
              ["微调", "批量 / 编码器学习率 / 风险头学习率", f"{s['finetune_batch_size']} / {s['encoder_learning_rate']} / {s['head_learning_rate']}"],
              ["选轮", "内部早停 / 最终轮数", f"耐心{s['finetune_patience']}轮 / {config['selected_finetune_epochs']}轮"],
              ["稳定性", "随机种子 / 梯度裁剪", f"{config['seed']} / 1.0"]]
    story += [table(params, [30*mm, 82*mm, 63*mm], sty), Spacer(1, 5*mm),
              paragraph("同一架构先比较State-MAE、未来动态预测、无先验图、人类风险链和随机链。Dynamics相对State-MAE只提高0.0009且配对区间跨0；人类风险链未超过普通动态或无先验图。因此按预注册停止规则保留更简单的State-MAE。", sty["body"]),
              model_chart(),
              paragraph("V3线性模型的开发AUC为0.7875，高于V12约0.0123，但配对区间跨0。最终提交V12，是因为它提供统一的深度驾驶状态表征并直接支撑任务二安全评价；报告保留这一差异，不把V12描述成开发集绝对最高分模型。", sty["body"]),
              paragraph("6　历史回测结果", sty["h1"])]
    result_rows = [["指标", "数值", "说明"], ["ROC-AUC", f"{score['roc_auc']:.4f}", "382辆开发车，20→40 OOF"],
                   ["PR-AUC", f"{score['pr_auc']:.4f}", "正例119辆"], ["Brier", f"{score['brier']:.4f}", "概率误差"],
                   ["Accuracy@0.5", f"{score['accuracy_at_0_5']:.4f}", "固定阈值"],
                   ["AUC 95%区间", f"[{score['auc_95pct_bootstrap']['lower']:.4f}, {score['auc_95pct_bootstrap']['upper']:.4f}]", "车辆级bootstrap"]]
    story += [table(result_rows, [45*mm, 52*mm, 78*mm], sty), PageBreak(),
              paragraph("7　关键发现、泛化措施与限制", sty["h1"])]
    findings = [["主题", "结论"], ["自监督", "V11已证明事件感知遮蔽重建稳定优于同架构Scratch；V12沿用这一有效方向"],
                ["语义状态", "合理六状态比随机分组方向更好，但不夸大为确定因果关系"],
                ["风险链", "U→C→P人工方向图未带来稳定增益，最终模型不使用该模块"],
                ["小样本", "小网络、车辆级分折、折内预处理、内层选轮和效应量门槛共同抑制过拟合"],
                ["缺失数据", "事件源、轨迹和IMU缺失显式建模；22辆低证据车使用开发先验"],
                ["时间边界", "任何锚点只读取当天及之前数据；未观察未来按右删失处理"]]
    story += [table(findings, [34*mm, 141*mm], sty), Spacer(1, 5*mm),
              paragraph("主要限制", sty["h2"]),
              paragraph("① 本地没有第61—100天标签，完整60→40概率存在时间外推；② 最终概率校准依赖日风险率近似恒定假设；③ 可监督车辆不足500，稀有事故造成较宽置信区间；④ 0.5阈值用于统一交付，AUC评测本身只关心概率排序；⑤ 22辆低证据车的概率是先验占位。", sty["body"]),
              paragraph("8　输出与复现", sty["h1"])]
    outputs = [["交付", "内容"], ["任务一结果CSV", "gpsno、prediction、probability，共500行"],
               ["模型权重", "development_state_mae.pt，含缩放器、结构配置和状态字典"],
               ["训练入口", "train_v12.py 与 finalize_v12.py"], ["直接推理", "generate_submission_outputs_v12.py"],
               ["验证", "500辆覆盖、概率范围、阈值一致、模型哈希和PDF检查"]]
    story += [table(outputs, [55*mm, 120*mm], sty), Spacer(1, 5*mm),
              paragraph("正式提交表只包含结果字段，不包含训练标签、验证折或内部路由。代码包提供依赖、预处理、训练、直接推理和任务二评分入口；打包过程已从所附模型权重重新生成两项结果并逐车比对。", sty["body"]),
              paragraph(f"模型文件：development_state_mae.pt<br/>源模型哈希：{manifest['source_sha256']['src/ie_alpaca/models/state_encoder_v12.py']}<br/>数据指纹：{manifest['data_fingerprint']}", sty["small"])]
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.run_dir, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
