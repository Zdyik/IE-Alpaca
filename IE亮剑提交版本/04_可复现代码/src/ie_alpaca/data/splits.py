"""Persist one vehicle-level development/holdout split for all iterations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def _fingerprint(reference: pd.DataFrame) -> str:
    pairs = reference[["gpsno", "label"]].sort_values("gpsno").astype(str)
    payload = "\n".join(pairs.gpsno + ":" + pairs.label).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_or_create_split(
    reference: pd.DataFrame, path: Path, *, seed: int, folds: int, holdout_fraction: float
) -> dict:
    if folds < 2 or not 0 < holdout_fraction < 0.5:
        raise ValueError("folds 至少为 2，holdout_fraction 必须在 (0, 0.5) 内")
    fingerprint = _fingerprint(reference)
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
        expected = (fingerprint, seed, folds, holdout_fraction)
        actual = (split.get("label_fingerprint"), split.get("seed"), split.get("folds"), split.get("holdout_fraction"))
        if actual != expected:
            raise ValueError(f"已冻结的划分与当前标签或配置不一致：{path}；请显式新建划分版本")
    else:
        ids = reference.gpsno.to_numpy()
        labels = reference.label.to_numpy()
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=holdout_fraction, random_state=seed)
        dev_pos, holdout_pos = next(splitter.split(ids, labels))
        dev = reference.iloc[dev_pos].sort_values("gpsno").reset_index(drop=True)
        holdout = reference.iloc[holdout_pos].sort_values("gpsno").reset_index(drop=True)
        if dev.label.value_counts().min() < folds:
            raise ValueError("开发集各类别车辆数必须不少于折数")
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        fold_map = {}
        for fold, (_, val_pos) in enumerate(skf.split(dev.gpsno, dev.label)):
            fold_map[str(fold)] = sorted(dev.iloc[val_pos].gpsno.tolist())
        split = {
            "version": path.stem,
            "label_fingerprint": fingerprint,
            "seed": seed,
            "folds": folds,
            "holdout_fraction": holdout_fraction,
            "development": sorted(dev.gpsno.tolist()),
            "holdout": sorted(holdout.gpsno.tolist()),
            "validation_folds": fold_map,
            "label_policy": "2026-06-20 20d history -> 40d future; labeled bag_index only",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as output:
            json.dump(split, output, ensure_ascii=False, indent=2)
    all_ids = set(reference.gpsno)
    dev_ids, holdout_ids = set(split["development"]), set(split["holdout"])
    fold_ids = [set(values) for values in split["validation_folds"].values()]
    if dev_ids & holdout_ids or dev_ids | holdout_ids != all_ids:
        raise ValueError("划分文件的开发/锁定组不互斥或未覆盖全部可监督车辆")
    if len(fold_ids) != folds or set.union(*fold_ids) != dev_ids or sum(map(len, fold_ids)) != len(dev_ids):
        raise ValueError("划分文件的 OOF 折不互斥或未覆盖开发集")
    return split


def audit_split(reference: pd.DataFrame, split: dict) -> dict:
    """Fail closed if any development, validation, or locked vehicle overlaps."""
    labels = reference.set_index("gpsno").label.astype(int)
    development = set(split["development"])
    holdout = set(split["holdout"])
    if development & holdout or development | holdout != set(labels.index):
        raise ValueError("开发集与锁定组交叉或缺少车辆")
    folds = []
    validation_union: set[str] = set()
    for fold, ids in sorted(split["validation_folds"].items(), key=lambda item: int(item[0])):
        validation = set(ids)
        training = development - validation
        if not validation or validation & holdout or training & holdout or training & validation:
            raise ValueError(f"折 {fold} 存在车辆交叉或空验证集")
        if validation_union & validation:
            raise ValueError("同一车辆出现在多个验证折")
        if set(labels.loc[list(validation)]) != {0, 1} or set(labels.loc[list(training)]) != {0, 1}:
            raise ValueError(f"折 {fold} 的训练或验证集缺少正负类别")
        validation_union |= validation
        folds.append({"fold": int(fold), "train_vehicles": len(training), "validation_vehicles": len(validation), "overlap": 0})
    if validation_union != development:
        raise ValueError("验证折未完整覆盖开发集")
    return {
        "status": "passed",
        "unit": "gpsno",
        "development_vehicles": len(development),
        "locked_vehicles": len(holdout),
        "locked_labels_used_in_fit_or_metrics": False,
        "validation_appears_once": True,
        "folds": folds,
        "reference_input_start": "2026-06-01",
        "reference_input_end": "2026-06-20",
        "reference_label_start": "2026-06-21",
        "reference_label_end": "2026-07-30",
    }
