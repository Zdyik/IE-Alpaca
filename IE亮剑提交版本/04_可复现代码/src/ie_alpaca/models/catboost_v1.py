"""Small fixed CatBoost models; hyperparameters are configuration, not OOF tuned."""

from __future__ import annotations

from catboost import CatBoostClassifier


def create_model(parameters: dict, seed: int) -> CatBoostClassifier:
    device = str(parameters.get("device", "cpu")).lower()
    if device not in {"cpu", "gpu"}:
        raise ValueError("model.device must be cpu or gpu")
    options = {
        "iterations": int(parameters["iterations"]),
        "depth": int(parameters["depth"]),
        "learning_rate": float(parameters["learning_rate"]),
        "l2_leaf_reg": float(parameters["l2_leaf_reg"]),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": int(seed),
        "task_type": device.upper(),
        "thread_count": int(parameters.get("thread_count", 4)),
        "allow_writing_files": False,
        "verbose": False,
    }
    if device == "gpu":
        options["devices"] = str(parameters.get("gpu_devices", "0"))
    return CatBoostClassifier(**options)
