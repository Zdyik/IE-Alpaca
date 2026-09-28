"""Build and validate the final V12 task-one and task-two submission folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent
DEFAULT_DATABASE = Path(r"E:\Databases\2026 清华IE亮剑-算法赛道赛题")
TASK1_RUN_ID = "V12FINAL_20260928_005652_c09b822a"
TASK2_RUN_ID = "T2SIMPLE_20260928_094215_652e8ce1"


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


def copy_python_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if (not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc"
                or path.name == ".gitkeep"):
            continue
        copy_file(path, destination / path.relative_to(source))


def copy_portable_resolved_config(source: Path, destination: Path) -> None:
    """Copy the resolved experiment settings without machine-specific paths."""
    config = json.loads(source.read_text(encoding="utf-8"))
    config["input_dir"] = "data/task1_preprocessed"
    config["results_root"] = "runs"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_portable_files(root: Path) -> None:
    """Reject PDFs, Python caches, review markers and absolute drive paths."""
    text_suffixes = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml", ".toml"}
    drive_path = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]+")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".pdf" or "__pycache__" in path.parts or path.suffix == ".pyc":
            raise ValueError(f"unsupported submission artifact: {relative}")
        if "待审" in path.name:
            raise ValueError(f"review draft included in submission: {relative}")
        if path.suffix.lower() in text_suffixes:
            content = path.read_text(encoding="utf-8")
            if drive_path.search(content):
                raise ValueError(f"machine-specific absolute path found in: {relative}")


def validate_outputs(task1_path: Path, task2_dir: Path) -> dict:
    task1 = pd.read_csv(task1_path, dtype={"gpsno": str})
    scores = pd.read_csv(task2_dir / "task2_safety_scores.csv", dtype={"gpsno": str})
    support = task2_dir / "支持材料"
    cards = pd.read_csv(support / "task2_driver_scorecards.csv", dtype={"gpsno": str})
    weights = pd.read_csv(support / "task2_event_weights.csv")

    if list(task1.columns) != ["gpsno", "prediction", "probability"]:
        raise ValueError("task-one output columns do not match the required three-column table")
    if list(scores.columns) != ["gpsno", "safety_score"]:
        raise ValueError("task-two compact output must contain gpsno and safety_score")
    for name, frame in (("task1", task1), ("task2", scores), ("scorecards", cards)):
        if len(frame) != 500 or frame.gpsno.nunique() != 500:
            raise ValueError(f"{name} must contain 500 unique vehicle IDs")
        if frame.isna().any().any():
            raise ValueError(f"{name} contains missing values")
    if set(task1.gpsno) != set(scores.gpsno) or set(scores.gpsno) != set(cards.gpsno):
        raise ValueError("task-one and task-two vehicle IDs differ")
    probability = task1.probability.to_numpy(float)
    prediction = task1.prediction.to_numpy(int)
    if not np.logical_and(probability >= 0, probability <= 1).all():
        raise ValueError("task-one probability is outside [0,1]")
    if not np.array_equal(prediction, (probability >= .5).astype(int)):
        raise ValueError("task-one class and probability threshold disagree")
    safety = scores.safety_score.to_numpy(float)
    if not np.logical_and(safety >= 0, safety <= 100).all():
        raise ValueError("task-two safety score is outside [0,100]")
    deduction_columns = [column for column in cards if column.startswith("deduction_")]
    if len(deduction_columns) != 6:
        raise ValueError("task-two scorecard must contain six deductions")
    reconstruction = 100 - cards[deduction_columns].sum(axis=1).to_numpy(float)
    reconciliation_error = float(np.max(np.abs(reconstruction - cards.safety_score.to_numpy(float))))
    if reconciliation_error > 1e-8:
        raise ValueError("task-two deductions do not reconcile to safety score")
    if (weights.event_weight_within_dimension < 0).any():
        raise ValueError("task-two event weights must be nonnegative")
    weight_error = float((weights.groupby("dimension").event_weight_within_dimension.sum() - 1).abs().max())
    if weight_error > 1e-8:
        raise ValueError("task-two event weights do not sum to one within each dimension")
    return {
        "status": "passed",
        "vehicles": 500,
        "task1_probability_range": [float(probability.min()), float(probability.max())],
        "task1_positive_predictions": int(prediction.sum()),
        "task2_score_range": [float(safety.min()), float(safety.max())],
        "task2_grade_counts": {
            str(key): int(value) for key, value in cards.grade.value_counts().sort_index().items()
        },
        "task2_low_evidence_vehicles": int(cards.confidence_level.eq("低").sum()),
        "task2_max_score_reconciliation_error": reconciliation_error,
        "task2_max_event_weight_sum_error": weight_error,
        "raw_competition_data_included": False,
        "external_data_used": False,
        "documents": "Markdown only; no PDF included",
    }


def write_submission_readme(path: Path) -> None:
    text = """# V12 最终提交说明

本文件夹包含任务一事故风险预测和任务二司机安全评价的正式结果、Markdown模型说明以及复现代码。

## 正式结果

- `01_任务一预测结果/task1_predictions.csv`：500辆车的二分类结果和风险概率。
- `02_任务二安全评价/task2_safety_scores.csv`：500辆车的0–100分安全分。
- `02_任务二安全评价/任务二司机安全评价模型说明.md`：评分维度、权重、公式、V12分层风险链和车队运营建议。
- `02_任务二安全评价/支持材料/task2_driver_scorecards.csv`：逐车六项扣分、等级、主要风险和管理建议。
- `02_任务二安全评价/支持材料/task2_event_weights.csv`：各维度内具体事件的学习权重。

## 文档

任务一和任务二说明均使用Markdown格式。本提交包不包含PDF。

## 复现

先完成数据预处理和V12任务一训练，再运行：

```powershell
python score_task2_simple_v12.py --config configs/experiments/task2_simple_score_v12.json
```

任务二评分采用六个维度。六项扣分之和能够逐项还原安全总分。证据不足车辆标记为U级，补齐设备和数据证据前不用于奖惩。
"""
    path.write_text(text, encoding="utf-8")


def build(database: Path, output: Path | None) -> Path:
    database = database.resolve()
    task1_run = database / "任务一实验结果" / "runs" / TASK1_RUN_ID
    task2_run = database / "任务二安全评价结果" / "runs" / TASK2_RUN_ID
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = database / f"V12正式提交_{stamp}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    task1_csv = task1_run / "deliverables" / "task1_v12_predictions.csv"
    task2_deliverables = task2_run / "deliverables"
    copy_file(task1_csv, output / "01_任务一预测结果" / "task1_predictions.csv")
    copy_file(task2_deliverables / "task2_safety_scores.csv",
              output / "02_任务二安全评价" / "task2_safety_scores.csv")
    copy_file(REPO / "docs" / "提交版_任务二司机安全评价模型说明.md",
              output / "02_任务二安全评价" / "任务二司机安全评价模型说明.md")
    for name in ("task2_driver_scorecards.csv", "task2_event_weights.csv", "task2_grade_summary.csv"):
        copy_file(task2_deliverables / name, output / "02_任务二安全评价" / "支持材料" / name)

    copy_file(REPO / "docs" / "提交版_任务一算法与模型说明_V12.md",
              output / "03_算法与模型说明" / "任务一事故风险预测算法与模型说明.md")
    copy_file(REPO / "docs" / "提交版_任务二司机安全评价模型说明.md",
              output / "03_算法与模型说明" / "任务二司机安全评价模型说明.md")

    code = output / "04_可复现代码"
    root_files = (
        "preprocess.py", "validate_preprocessing.py", "train_v12.py", "finalize_v12.py",
        "export_task1_v12.py", "score_task2_simple_v12.py", "requirements.txt",
    )
    for name in root_files:
        copy_file(REPO / name, code / name)
    copy_python_tree(REPO / "src", code / "src")
    for name in ("test_train_v12.py", "test_task2_simple_v12.py"):
        copy_file(REPO / "tests" / name, code / "tests" / name)
    for name in ("v12_risk_chain_ssl.json", "v12_finalize_state_mae.json", "task2_simple_score_v12.json"):
        copy_file(REPO / "configs" / "experiments" / name,
                  code / "configs" / "experiments" / name)
    copy_file(REPO / "docs" / "V12人类先验引导的驾驶状态表征设计.md",
              code / "docs" / "V12人类先验引导的驾驶状态表征设计.md")
    copy_file(REPO / "docs" / "提交版_任务一算法与模型说明_V12.md",
              code / "docs" / "任务一算法与模型说明.md")
    copy_file(REPO / "docs" / "提交版_任务二司机安全评价模型说明.md",
              code / "docs" / "任务二司机安全评价模型说明.md")
    copy_portable_resolved_config(
        task1_run / "config.resolved.json",
        code / "artifacts" / "task1_run" / "config.resolved.json",
    )
    for name in ("leakage_audit.json", "manifest.json", "metrics.json", "splits.json"):
        copy_file(task1_run / name, code / "artifacts" / "task1_run" / name)
    copy_file(task1_run / "models" / "development_state_mae.pt",
              code / "artifacts" / "task1_run" / "models" / "development_state_mae.pt")
    for name in ("manifest.json", "metrics.json"):
        copy_file(task2_run / name, code / "artifacts" / "task2_run" / name)
    write_submission_readme(output / "提交说明.md")
    write_submission_readme(code / "README.md")
    validate_portable_files(output)

    validation = validate_outputs(
        output / "01_任务一预测结果" / "task1_predictions.csv",
        output / "02_任务二安全评价",
    )
    validation.update({
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "task1_run_id": TASK1_RUN_ID,
        "task2_run_id": TASK2_RUN_ID,
    })
    (output / "submission_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in {"submission_manifest.json", "SHA256SUMS.txt"}:
            continue
        files.append({"path": path.relative_to(output).as_posix(),
                      "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "package": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_submission",
        "task1_run_id": TASK1_RUN_ID,
        "task2_run_id": TASK2_RUN_ID,
        "file_count": len(files),
        "total_bytes": int(sum(item["bytes"] for item in files)),
        "files": files,
    }
    (output / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    hash_lines = [f"{sha256(path)}  {path.relative_to(output).as_posix()}"
                  for path in sorted(output.rglob("*"))
                  if path.is_file() and path.name != "SHA256SUMS.txt"]
    (output / "SHA256SUMS.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-root", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(build(args.database_root, args.output))


if __name__ == "__main__":
    main()
