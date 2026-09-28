"""Assemble, reproduce and validate the final V12 submission folder for both tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from generate_submission_outputs_v12 import generate as reproduce_outputs
from generate_task1_report import build_report as build_task1_report
from generate_task2_report import FONT, FONT_BOLD, NAVY, BLUE, register_fonts


REPO = Path(__file__).resolve().parent
DEFAULT_DB = Path(r"E:\Databases\2026 清华IE亮剑-算法赛道赛题")
TASK1_RUN_ID = "V12FINAL_20260928_005652_c09b822a"
TASK2_RUN_ID = "T2V12_20260928_013606_f66a4b6f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def draw_cover(path: Path, task1_pages: int, task2_pages: int) -> None:
    register_fonts()
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    c.setFillColor(colors.HexColor("#F4F8FA"))
    c.rect(0, 0, width, height, stroke=0, fill=1)
    c.setFillColor(NAVY)
    c.rect(0, height - 84*mm, width, 84*mm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont(FONT_BOLD, 25)
    c.drawString(25*mm, height - 39*mm, "V12 算法与模型说明")
    c.setFont(FONT, 13)
    c.drawString(25*mm, height - 52*mm, "任务一事故预测 · 任务二司机安全评价")
    c.setFillColor(BLUE)
    c.rect(25*mm, height - 95*mm, 42*mm, 2.2*mm, stroke=0, fill=1)
    c.setFillColor(NAVY)
    c.setFont(FONT_BOLD, 14)
    c.drawString(25*mm, height - 116*mm, "提交内容")
    c.setFont(FONT, 10.5)
    lines = [
        f"第一部分　任务一事故风险预测模型说明（{task1_pages}页）",
        f"第二部分　任务二司机安全评价模型说明（{task2_pages}页）",
        "模型版本　V12 State-MAE",
        "结果范围　500辆车，60天驾驶记录",
        "正式输出　任务一二分类与概率；任务二0–100安全分",
    ]
    y = height - 132*mm
    for line in lines:
        c.drawString(28*mm, y, line)
        y -= 12*mm
    c.setFillColor(colors.HexColor("#607480"))
    c.setFont(FONT, 8.5)
    c.drawString(25*mm, 30*mm, "说明：本地指标来自开发集车辆级OOF历史回测，正式成绩以赛方评测为准。")
    c.drawRightString(width - 25*mm, 20*mm, "2026-09-28")
    c.save()


def merge_reports(cover: Path, task1: Path, task2: Path, output: Path) -> int:
    writer = PdfWriter()
    pages = 0
    for source in (cover, task1, task2):
        reader = PdfReader(str(source))
        pages += len(reader.pages)
        for page in reader.pages:
            writer.add_page(page)
    with output.open("xb") as stream:
        writer.write(stream)
    return pages


def submission_readme(task1_pages: int, task2_pages: int, combined_pages: int) -> str:
    return f"""# V12 最终提交说明

本文件夹按照赛题第6节提交要求整理，包含任务一结果、任务二结果、PDF算法说明和可复现源码。未包含任何原始比赛数据。

## 正式提交文件

1. `01_任务一预测结果/task1_predictions.csv`：500辆车，列为`gpsno,prediction,probability`。
2. `02_任务二安全评价/task2_safety_scores.csv`：500辆车，列为`gpsno,safety_score`。
3. `03_算法与模型说明/V12算法与模型说明_任务一与任务二.pdf`：统一正式说明，共{combined_pages}页。
4. `04_可复现代码/`：预处理、V12训练、直接推理、任务二评分、依赖和运行说明。

任务一单独说明为{task1_pages}页，任务二单独说明为{task2_pages}页，也保存在`03_算法与模型说明`中，便于分开审阅。

## 任务二补充文件

`02_任务二安全评价/支持材料`保存详细计分卡、日度动态得分、24类事件权重和车队分档汇总。这些文件用于证明可解释性、动态性和可运营性；正式最简评分结果仍是`task2_safety_scores.csv`。

## 校验结果

- 两个正式CSV都覆盖相同的500个车辆ID，且每车恰好一行；
- 任务一概率均位于[0,1]，二分类严格等于`probability>=0.5`；
- 任务二安全分均位于[0,100]，详细分项扣分可以还原总分；
- 已使用随包提供的V12权重重新生成任务一和任务二输出，并与正式CSV逐车比对；
- 三份PDF均完成页数、文本和视觉检查；
- `SHA256SUMS.txt`记录提交包内文件哈希。

正式提交前如平台提供专用模板，只需按模板重命名列，不应改变车辆顺序、概率或安全分。
"""


def code_readme() -> str:
    return """# V12 代码运行说明

## 1. 环境

推荐 Python 3.11+。在本目录执行：

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
```

## 2. 从原始比赛数据预处理

```powershell
python preprocess.py --input '<比赛数据库根目录>' --stage all --threads 4 --memory-limit 6GB
python validate_preprocessing.py --input '<比赛数据库根目录>\\任务一预处理结果'
```

## 3. 使用随包模型直接复现两项结果

```powershell
python generate_submission_outputs_v12.py `
  --input '<比赛数据库根目录>\\任务一预处理结果' `
  --checkpoint 'artifacts\\task1_run\\models\\development_state_mae.pt' `
  --splits 'artifacts\\task1_run\\splits.json' `
  --task2-config 'configs\\experiments\\task2_v12_scorecard.json' `
  --output 'reproduced_outputs' `
  --device auto
```

生成的`task1_predictions.csv`和`task2_safety_scores.csv`应分别与提交包中的正式结果一致。`--device auto`优先使用CUDA，没有CUDA时回退CPU。

## 4. 重新训练V12

```powershell
python train_v12.py --config configs\\experiments\\v12_risk_chain_ssl.json
```

该命令完成车辆级五折实验和模型消融。若要重新执行最终重训，将打印出的父运行ID填入`configs/experiments/v12_finalize_state_mae.json`的`parent_run_id`，再运行：

```powershell
python finalize_v12.py --config configs\\experiments\\v12_finalize_state_mae.json
```

## 5. 关键入口

- `preprocess.py`：四类原始数据清洗与日级汇总；
- `train_v12.py`：State-MAE、动态预测和风险链消融；
- `finalize_v12.py`：按停止规则重训最终State-MAE；
- `generate_submission_outputs_v12.py`：从所附权重直接生成两个任务结果；
- `score_task2_v12.py`：任务二版本化评分流水线；
- `validate_task2_outputs.py`：任务二交付校验。

所有训练与评分仅使用赛方提供数据，没有使用外部数据。
"""


def compare_reproduced(reproduced: Path, official_task1: Path, official_task2: Path,
                       official_cards: Path, official_history: Path) -> dict[str, object]:
    t1_new = pd.read_csv(reproduced / "task1_predictions.csv", encoding="utf-8-sig", dtype={"gpsno": str})
    t1_old = pd.read_csv(official_task1, encoding="utf-8-sig", dtype={"gpsno": str})
    t1 = t1_old.merge(t1_new, on="gpsno", suffixes=("_official", "_reproduced"), validate="one_to_one")
    if len(t1) != 500 or not np.array_equal(t1.prediction_official, t1.prediction_reproduced):
        raise ValueError("reproduced task-one classes differ")
    t1_error = float(np.max(np.abs(t1.probability_official - t1.probability_reproduced)))
    if t1_error > 1e-12:
        raise ValueError(f"reproduced task-one probabilities differ: {t1_error}")

    t2_new = pd.read_csv(reproduced / "task2_safety_scores.csv", encoding="utf-8-sig", dtype={"gpsno": str})
    t2_old = pd.read_csv(official_task2, encoding="utf-8-sig", dtype={"gpsno": str})
    t2 = t2_old.merge(t2_new, on="gpsno", suffixes=("_official", "_reproduced"), validate="one_to_one")
    t2_error = float(np.max(np.abs(t2.safety_score_official - t2.safety_score_reproduced)))
    if len(t2) != 500 or t2_error > 5e-5:
        raise ValueError(f"reproduced task-two scores differ: {t2_error}")

    cards_new = pd.read_csv(reproduced / "task2_scorecards.csv", encoding="utf-8-sig", dtype={"gpsno": str})
    cards_old = pd.read_csv(official_cards, encoding="utf-8-sig", dtype={"gpsno": str})
    cards = cards_old[["gpsno", "grade", "total_deduction"]].merge(
        cards_new[["gpsno", "grade", "total_deduction"]], on="gpsno",
        suffixes=("_official", "_reproduced"), validate="one_to_one")
    if not (cards.grade_official == cards.grade_reproduced).all():
        raise ValueError("reproduced task-two grades differ")
    card_error = float(np.max(np.abs(cards.total_deduction_official - cards.total_deduction_reproduced)))
    if card_error > 1e-10:
        raise ValueError(f"reproduced task-two deductions differ: {card_error}")

    history_new = pd.read_csv(reproduced / "task2_daily_score_history.csv", encoding="utf-8-sig",
                              dtype={"gpsno": str})
    history_old = pd.read_csv(official_history, encoding="utf-8-sig", dtype={"gpsno": str})
    if len(history_new) != 23500 or len(history_old) != 23500:
        raise ValueError("daily score history row count differs")
    key = ["gpsno", "anchor_day"]
    h = history_old[key + ["safety_score"]].merge(
        history_new[key + ["safety_score"]], on=key,
        suffixes=("_official", "_reproduced"), validate="one_to_one")
    history_error = float(np.max(np.abs(h.safety_score_official - h.safety_score_reproduced)))
    if history_error > 1e-10:
        raise ValueError(f"reproduced task-two history differs: {history_error}")
    return {
        "status": "passed", "vehicles": 500,
        "task1_max_probability_error": t1_error,
        "task2_max_score_error": t2_error,
        "task2_max_deduction_error": card_error,
        "task2_history_max_score_error": history_error,
    }


def assemble(database_root: Path, output_root: Path | None = None) -> Path:
    database_root = database_root.resolve()
    output_root = (output_root or database_root).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = output_root / f"V12最终提交_{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    task1_run = database_root / "任务一实验结果" / "runs" / TASK1_RUN_ID
    task2_run = database_root / "任务二安全评价结果" / "runs" / TASK2_RUN_ID
    t1_manifest = json.loads((task1_run / "manifest.json").read_text(encoding="utf-8"))
    t2_manifest = json.loads((task2_run / "manifest.json").read_text(encoding="utf-8"))
    if t1_manifest.get("status") != "completed" or t1_manifest.get("primary_method") != "state_mae":
        raise ValueError("task-one V12 final run is not complete")
    if t2_manifest.get("status") != "completed" or t2_manifest.get("validation_status") != "passed":
        raise ValueError("task-two V12 run is not complete and validated")

    task1_csv = task1_run / "deliverables" / "task1_v12_predictions.csv"
    task2_csv = task2_run / "deliverables" / "task2_driver_safety_scores.csv"
    copy_file(task1_csv, target / "01_任务一预测结果" / "task1_predictions.csv")
    copy_file(task2_csv, target / "02_任务二安全评价" / "task2_safety_scores.csv")
    support = target / "02_任务二安全评价" / "支持材料"
    for source_name, destination_name in (
        ("task2_driver_scorecards.csv", "task2_scorecards.csv"),
        ("task2_daily_score_history.csv", "task2_daily_score_history.csv"),
        ("task2_event_weights.csv", "task2_event_weights.csv"),
        ("task2_fleet_summary.csv", "task2_fleet_summary.csv"),
    ):
        copy_file(task2_run / "deliverables" / source_name, support / destination_name)

    docs = target / "03_算法与模型说明"
    task1_pdf = docs / "任务一事故风险预测模型说明.pdf"
    task2_pdf = docs / "任务二司机安全评价模型说明.pdf"
    combined_pdf = docs / "V12算法与模型说明_任务一与任务二.pdf"
    docs.mkdir(parents=True)
    build_task1_report(task1_run, task1_pdf)
    copy_file(task2_run / "deliverables" / "task2_v12_safety_model_report.pdf", task2_pdf)
    task1_pages = len(PdfReader(str(task1_pdf)).pages)
    task2_pages = len(PdfReader(str(task2_pdf)).pages)
    with tempfile.TemporaryDirectory(prefix="v12_cover_", dir=database_root) as temp:
        cover = Path(temp) / "cover.pdf"
        draw_cover(cover, task1_pages, task2_pages)
        combined_pages = merge_reports(cover, task1_pdf, task2_pdf, combined_pdf)

    code = target / "04_可复现代码"
    code.mkdir(parents=True)
    script_names = [
        "preprocess.py", "validate_preprocessing.py", "train_v12.py", "finalize_v12.py",
        "export_task1_v12.py", "score_task2_v12.py", "generate_task1_report.py",
        "generate_task2_report.py", "generate_submission_outputs_v12.py",
        "validate_task2_outputs.py", "prepare_v12_submission.py",
    ]
    for name in script_names:
        copy_file(REPO / name, code / name)
    copy_file(REPO / "requirements.txt", code / "requirements.txt")
    copy_file(REPO / "README.md", code / "PROJECT_README.md")
    shutil.copytree(REPO / "src" / "ie_alpaca", code / "src" / "ie_alpaca",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("v12_risk_chain_ssl.json", "v12_finalize_state_mae.json", "task2_v12_scorecard.json"):
        copy_file(REPO / "configs" / "experiments" / name,
                  code / "configs" / "experiments" / name)
    for name in ("test_train_v12.py", "test_task2_v12.py"):
        copy_file(REPO / "tests" / name, code / "tests" / name)
    for name in ("任务一算法与模型说明文档.md", "V12人类先验引导的驾驶状态表征设计.md",
                 "任务二V12安全评价模型设计.md"):
        copy_file(REPO / "docs" / name, code / "docs" / name)

    artifacts = code / "artifacts" / "task1_run"
    for name in ("manifest.json", "metrics.json", "leakage_audit.json", "splits.json", "config.resolved.json"):
        copy_file(task1_run / name, artifacts / name)
    copy_file(task1_run / "models" / "development_state_mae.pt",
              artifacts / "models" / "development_state_mae.pt")
    copy_file(task2_run / "metrics.json", code / "artifacts" / "task2_metrics.json")
    copy_file(task2_run / "validation_report.json", code / "artifacts" / "task2_validation_report.json")
    (code / "README.md").write_text(code_readme(), encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="v12_reproduce_", dir=database_root) as temp:
        reproduced = Path(temp) / "outputs"
        reproduce_outputs(
            database_root / "任务一预处理结果",
            artifacts / "models" / "development_state_mae.pt",
            artifacts / "splits.json",
            code / "configs" / "experiments" / "task2_v12_scorecard.json",
            reproduced,
            "auto",
        )
        reproduction = compare_reproduced(
            reproduced, task1_csv, task2_csv,
            task2_run / "deliverables" / "task2_driver_scorecards.csv",
            task2_run / "deliverables" / "task2_daily_score_history.csv",
        )

    (target / "提交说明.md").write_text(
        submission_readme(task1_pages, task2_pages, combined_pages), encoding="utf-8")
    pdf_info = {}
    for path in (task1_pdf, task2_pdf, combined_pdf):
        reader = PdfReader(str(path))
        text_chars = sum(len(page.extract_text() or "") for page in reader.pages)
        if len(reader.pages) < 5 or text_chars < 1000:
            raise ValueError(f"PDF validation failed: {path}")
        pdf_info[path.name] = {"pages": len(reader.pages), "text_chars": text_chars}

    official1 = pd.read_csv(target / "01_任务一预测结果" / "task1_predictions.csv",
                            encoding="utf-8-sig", dtype={"gpsno": str})
    official2 = pd.read_csv(target / "02_任务二安全评价" / "task2_safety_scores.csv",
                            encoding="utf-8-sig", dtype={"gpsno": str})
    if len(official1) != 500 or official1.gpsno.nunique() != 500 or set(official1.gpsno) != set(official2.gpsno):
        raise ValueError("official result vehicle coverage mismatch")
    if not official1.probability.between(0, 1).all() or not np.array_equal(
            official1.prediction.to_numpy(int), (official1.probability.to_numpy(float) >= .5).astype(int)):
        raise ValueError("task-one probability or threshold validation failed")
    if not official2.safety_score.between(0, 100).all():
        raise ValueError("task-two score validation failed")

    forbidden = [path for path in target.rglob("*") if path.is_file() and path.suffix.lower() in {".parquet", ".xlsx"}]
    if forbidden:
        raise ValueError(f"unexpected data artifacts in submission: {forbidden}")
    all_files = sorted(path for path in target.rglob("*") if path.is_file())
    hash_lines = [f"{sha256(path)}  {path.relative_to(target).as_posix()}" for path in all_files]
    (target / "SHA256SUMS.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    validation = {
        "status": "passed",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "task1_run_id": TASK1_RUN_ID,
        "task2_run_id": TASK2_RUN_ID,
        "official_files": {
            "task1": "01_任务一预测结果/task1_predictions.csv",
            "task2": "02_任务二安全评价/task2_safety_scores.csv",
            "model_report": "03_算法与模型说明/V12算法与模型说明_任务一与任务二.pdf",
            "source": "04_可复现代码",
        },
        "vehicles": 500,
        "task1_probability_range": [float(official1.probability.min()), float(official1.probability.max())],
        "task1_positive_predictions": int(official1.prediction.sum()),
        "task2_score_range": [float(official2.safety_score.min()), float(official2.safety_score.max())],
        "reproduction": reproduction,
        "pdfs": pdf_info,
        "raw_competition_data_included": False,
        "external_data_used": False,
        "checks": [
            "two official CSV files contain the same 500 unique vehicle IDs",
            "task-one probabilities are in [0,1] and classes equal probability >= 0.5",
            "task-two safety scores are in [0,100]",
            "provided V12 checkpoint reproduced both tasks vehicle by vehicle",
            "combined and separate PDFs contain extractable text and valid page counts",
            "source bundle includes preprocessing, training, inference, scoring and run instructions",
            "no raw parquet or spreadsheet data was copied into the submission folder",
        ],
    }
    (target / "submission_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_files = sorted(path for path in target.rglob("*") if path.is_file())
    manifest = {
        "package": target.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_submission",
        "file_count": len(manifest_files),
        "total_bytes": sum(path.stat().st_size for path in manifest_files),
        "files": [{"path": path.relative_to(target).as_posix(), "bytes": path.stat().st_size,
                   "sha256": sha256(path)} for path in manifest_files],
    }
    (target / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-root", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(assemble(args.database_root, args.output_root))


if __name__ == "__main__":
    main()
