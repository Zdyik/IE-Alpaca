"""Fold-local fitting and inference for the 60-day sequence model."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.nn import functional as F

from ie_alpaca.features.sequence_v2 import HORIZONS, SequenceData, fit_scaler, labels_for, transform
from ie_alpaca.models.temporal_v2 import TemporalRiskNet


def resolve_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device 只能是 auto、cpu 或 cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch 无法使用 CUDA；请安装匹配的 CUDA 构建")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_model(data: SequenceData, train_ids: set[str], config: dict, seed: int, device: torch.device):
    set_seed(seed)
    scaler = fit_scaler(data, train_ids)
    values_np, observed_np = transform(data, scaler)
    values = torch.as_tensor(values_np, device=device)
    observed = torch.as_tensor(observed_np, device=device)
    net = TemporalRiskNet(data.groups, len(data.features), hidden=int(config["hidden"]),
                          dropout=float(config["dropout"])).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    generator = np.random.default_rng(seed)
    train_positions = data.positions(train_ids)
    if len(train_positions) != len(train_ids):
        raise ValueError("训练车辆缺失日特征")
    batch_size = int(config["batch_size"])
    event_indicator = torch.zeros(len(data.features), dtype=torch.bool, device=device)
    event_indicator[data.groups["event"]] = True
    log = []

    # All 60 days of training vehicles are used here. Validation/test vehicles
    # never enter self-supervised fitting, including normalization statistics.
    for epoch in range(int(config["pretrain_epochs"])):
        net.train()
        losses = []
        for chunk in np.array_split(generator.permutation(train_positions),
                                    max(1, int(np.ceil(len(train_positions) / batch_size)))):
            index = torch.as_tensor(chunk, device=device)
            original = values[index]
            mask = observed[index]
            hidden_days = torch.as_tensor(generator.random((len(chunk), 60)) < 0.2, device=device)
            # At least one masked day in each batch.
            hidden_days[:, 0] = True
            visible = (~hidden_days).unsqueeze(-1)
            _, reconstruction = net(original * visible, mask * visible,
                                    torch.full((len(chunk),), 60, dtype=torch.long, device=device))
            weight = mask * hidden_days.unsqueeze(-1)
            loss = (F.smooth_l1_loss(reconstruction, original, reduction="none") * weight).sum() / weight.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        log.append({"phase": "masked_60day_pretrain", "epoch": epoch + 1, "loss": float(np.mean(losses))})

    rows, labels, available = labels_for(data, train_positions)
    if not len(rows):
        raise ValueError("没有可监督的训练样本")
    targets = torch.as_tensor(labels, device=device)
    valid_targets = torch.as_tensor(available, device=device)
    horizon_weight = torch.as_tensor([1.0, 1.0, 1.0, 1.0, 2.0], device=device)
    for epoch in range(int(config["train_epochs"])):
        net.train()
        losses = []
        for chunk in np.array_split(generator.permutation(len(rows)),
                                    max(1, int(np.ceil(len(rows) / batch_size)))):
            vehicle = torch.as_tensor(rows[chunk, 0], dtype=torch.long, device=device)
            length = torch.as_tensor(rows[chunk, 1], dtype=torch.long, device=device)
            batch_values = values[vehicle].clone()
            batch_observed = observed[vehicle].clone()
            # Simulate a missing event modality for part of the training cars.
            # The missingness mask lets the model use exposure/IMU when the feed is absent.
            dropout_rows = torch.as_tensor(generator.random(len(chunk)) < float(config["event_dropout"]), device=device)
            if bool(dropout_rows.any()):
                missing_event = dropout_rows[:, None, None] & event_indicator[None, None, :]
                batch_values = batch_values.masked_fill(missing_event, 0)
                batch_observed = batch_observed.masked_fill(missing_event, 0)
            probability, _ = net(batch_values, batch_observed, length)
            weight = valid_targets[chunk] * horizon_weight
            loss = (F.binary_cross_entropy(probability, targets[chunk], reduction="none") * weight).sum() / weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        log.append({"phase": "supervised", "epoch": epoch + 1, "loss": float(np.mean(losses))})
    return net, scaler, log


@torch.inference_mode()
def predict(net: TemporalRiskNet, data: SequenceData, scaler: dict,
            vehicle_indices: np.ndarray, history_days: int, device: torch.device) -> np.ndarray:
    if not 1 <= history_days <= 60:
        raise ValueError("历史天数必须介于 1 和 60")
    values, observed = transform(data, scaler)
    net.eval()
    outputs = []
    for chunk in np.array_split(vehicle_indices, max(1, int(np.ceil(len(vehicle_indices) / 128)))):
        v = torch.as_tensor(values[chunk], device=device)
        m = torch.as_tensor(observed[chunk], device=device)
        lengths = torch.full((len(chunk),), history_days, dtype=torch.long, device=device)
        p, _ = net(v, m, lengths)
        outputs.append(p.cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float64)
