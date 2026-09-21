"""Create a self-contained, immutable experiment folder beside predictions."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(input_dir: Path) -> dict:
    return {
        name: {
            "size": (input_dir / name).stat().st_size,
            "sha256": sha256_file(input_dir / name),
        }
        for name in ("daily_features.parquet", "bag_index.parquet", "profile.parquet")
    }


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True,
            check=True, timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def create_run(
    repo: Path, results_root: Path, config_path: Path, config: dict,
    *, version: str = "V1", entrypoint: str = "train_v1.py",
) -> tuple[Path, dict]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{version}_{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir = results_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("source", "predictions", "models", "logs"):
        (run_dir / name).mkdir()
    files = [repo / entrypoint, repo / "preprocess.py", repo / "requirements.txt", config_path]
    files.extend(sorted((repo / "src" / "ie_alpaca").rglob("*.py")))
    source_hashes = {}
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(repo)
        target = run_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[str(relative).replace("\\", "/")] = sha256_file(target)
    (run_dir / "config.resolved.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(repo, "rev-parse", "HEAD"),
        "git_status": _git(repo, "status", "--short"),
        "parent_run_id": config.get("parent_run_id"),
        "source_sha256": source_hashes,
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_dir, manifest


def verify_source(repo: Path, manifest: dict) -> None:
    changed = [
        name for name, expected in manifest["source_sha256"].items()
        if not (repo / name).is_file() or sha256_file(repo / name) != expected
    ]
    if changed:
        raise RuntimeError(f"训练期间源代码已改变，请新建实验版本：{changed}")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_parquet(path: Path, data: pd.DataFrame) -> None:
    con = duckdb.connect()
    try:
        con.register("result_frame", data)
        con.execute("COPY result_frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
    finally:
        con.close()


def update_leaderboard(results_root: Path) -> None:
    """Rebuild the small index; version folders remain the source of truth."""
    rows = []
    for manifest_path in sorted((results_root / "runs").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics_path = manifest_path.parent / "metrics.json"
        if manifest.get("status") != "completed" or not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        primary = metrics["development_oof"]
        rows.append({
            "run_id": manifest["run_id"],
            "created_utc": manifest["created_utc"],
            "model": manifest.get("model", "catboost_v1"),
            "split_version": manifest["split_version"],
            "data_fingerprint": manifest["data_fingerprint"],
            "evaluation_version": manifest["evaluation_version"],
            "roc_auc": primary["roc_auc"],
            "accuracy_at_0_5": primary["accuracy_at_0_5"],
            "pr_auc": primary["pr_auc"],
            "brier": primary["brier"],
            "run_dir": str(manifest_path.parent),
        })
    path = results_root / "leaderboard.csv"
    temp = results_root / "leaderboard.csv.partial"
    pd.DataFrame(rows, columns=[
        "run_id", "created_utc", "model", "split_version", "data_fingerprint", "evaluation_version",
        "roc_auc", "accuracy_at_0_5", "pr_auc", "brier", "run_dir",
    ]).sort_values(
        ["data_fingerprint", "split_version", "evaluation_version", "roc_auc"],
        ascending=[True, True, True, False], na_position="last",
    ).to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)
