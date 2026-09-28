"""Frozen probe, right-censored risk fine-tuning and inference for V12."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch.nn import functional as F

from ie_alpaca.features.landmark_v3 import ANCHORS
from ie_alpaca.models.state_encoder_v12 import RiskNetV12
from ie_alpaca.training.event_v4 import next_event_loss, set_seed


def _tensors(pack: dict, device: torch.device):
    return {name: torch.as_tensor(pack[name], device=device)
            for name in ("events", "context", "aux_targets", "event_day", "horizon")}


def _base_hazard(pack: dict) -> float:
    event, horizon = pack["event_day"], pack["horizon"]
    positive = (event > 0).sum()
    observed = np.where(event > 0, event, horizon).sum()
    return float(np.clip(positive / max(observed, 1), 1e-4, .1))


def _positive_weight(targets: torch.Tensor) -> torch.Tensor:
    values = targets[:, :46]
    positive = values.sum(dim=(0, 1))
    total = values.shape[0] * values.shape[1]
    return ((total - positive) / positive.clamp_min(1)).clamp(1, 20)


def _loss(net, data, positions, config, positive_weight):
    logits, auxiliary, _ = net(data["events"][positions], data["context"][positions], ANCHORS)
    survival = next_event_loss(logits, data["event_day"][positions], data["horizon"][positions])
    auxiliary_loss = F.binary_cross_entropy_with_logits(
        auxiliary[:, :46], data["aux_targets"][positions, :46], pos_weight=positive_weight)
    return survival + float(config["auxiliary_weight"]) * auxiliary_loss, survival, auxiliary_loss


@torch.inference_mode()
def _validation_nll(net, data, batch_size: int) -> float:
    net.eval()
    weighted = []
    for start in range(0, len(data["events"]), batch_size):
        stop = start + batch_size
        logits, _, _ = net(data["events"][start:stop], data["context"][start:stop], ANCHORS)
        loss = next_event_loss(logits, data["event_day"][start:stop], data["horizon"][start:stop])
        weighted.append(float(loss.cpu()) * len(data["events"][start:stop]))
    return float(sum(weighted) / len(data["events"]))


def fit_risk_v12(pack: dict, config: dict, method: str, backbone_state: dict,
                 seed: int, device: torch.device, epochs: int | None = None,
                 validation_pack: dict | None = None):
    set_seed(seed)
    net = RiskNetV12(pack["context"].shape[-1], method, int(config["width"]),
                     float(config["dropout"]), _base_hazard(pack)).to(device)
    net.backbone.load_state_dict(backbone_state)
    train = _tensors(pack, device)
    validation = None if validation_pack is None else _tensors(validation_pack, device)
    batch_size = int(config["finetune_batch_size"])
    rng = np.random.default_rng(seed)
    positive_weight = _positive_weight(train["aux_targets"])
    history: list[dict] = []

    for parameter in net.backbone.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([parameter for parameter in net.parameters() if parameter.requires_grad],
                                  lr=float(config["head_learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    for epoch in range(1, int(config["probe_epochs"]) + 1):
        net.train(); losses = []
        chunks = np.array_split(rng.permutation(len(train["events"])),
                                max(1, int(np.ceil(len(train["events"]) / batch_size))))
        for chunk in chunks:
            positions = torch.as_tensor(chunk, device=device)
            loss, _, _ = _loss(net, train, positions, config, positive_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        entry = {"phase": "probe", "epoch": epoch, "train_objective": float(np.mean(losses))}
        if validation is not None:
            entry["inner_val_nll"] = _validation_nll(net, validation, batch_size)
        history.append(entry)
    probe_state = copy.deepcopy(net.state_dict())

    for parameter in net.backbone.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {"params": net.backbone.parameters(), "lr": float(config["encoder_learning_rate"])},
        {"params": [parameter for name, parameter in net.named_parameters()
                    if not name.startswith("backbone.")], "lr": float(config["head_learning_rate"])},
    ], weight_decay=float(config["weight_decay"]))
    limit = int(config["max_finetune_epochs"] if epochs is None else epochs)
    best_epoch, best_nll, best_state = 0, float("inf"), None
    for epoch in range(1, limit + 1):
        net.train(); losses, survival_losses, auxiliary_losses = [], [], []
        chunks = np.array_split(rng.permutation(len(train["events"])),
                                max(1, int(np.ceil(len(train["events"]) / batch_size))))
        for chunk in chunks:
            positions = torch.as_tensor(chunk, device=device)
            loss, survival, auxiliary = _loss(net, train, positions, config, positive_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
            survival_losses.append(float(survival.detach().cpu()))
            auxiliary_losses.append(float(auxiliary.detach().cpu()))
        entry = {"phase": "finetune", "epoch": epoch,
                 "train_objective": float(np.mean(losses)),
                 "train_survival_nll": float(np.mean(survival_losses)),
                 "train_auxiliary_loss": float(np.mean(auxiliary_losses))}
        if validation is not None:
            value = _validation_nll(net, validation, batch_size)
            entry["inner_val_nll"] = value
            if epoch >= int(config["min_finetune_epochs"]) and value < best_nll - float(config["min_delta"]):
                best_epoch, best_nll, best_state = epoch, value, copy.deepcopy(net.state_dict())
            if (epoch >= int(config["min_finetune_epochs"]) and best_epoch and
                    epoch - best_epoch >= int(config["finetune_patience"])):
                history.append(entry)
                break
        history.append(entry)
    if validation_pack is not None:
        if best_state is None:
            best_epoch, best_state = limit, copy.deepcopy(net.state_dict())
        net.load_state_dict(best_state)
    return net, history, (best_epoch if validation_pack is not None else limit), probe_state


@torch.inference_mode()
def predict_anchors_v12(net: RiskNetV12, pack: dict, device: torch.device, batch_size: int) -> np.ndarray:
    net.eval(); values = []
    for start in range(0, len(pack["events"]), batch_size):
        event = torch.as_tensor(pack["events"][start:start + batch_size], device=device)
        context = torch.as_tensor(pack["context"][start:start + batch_size], device=device)
        values.append(torch.sigmoid(net(event, context, ANCHORS)[0]).cpu().numpy())
    return np.concatenate(values).astype(float)


@torch.inference_mode()
def predict_day60_v12(net: RiskNetV12, events: np.ndarray, context: np.ndarray,
                      device: torch.device, batch_size: int) -> np.ndarray:
    net.eval(); values = []
    for start in range(0, len(events), batch_size):
        event = torch.as_tensor(events[start:start + batch_size], device=device)
        ctx = torch.as_tensor(context[start:start + batch_size], device=device)
        values.append(torch.sigmoid(net(event, ctx, (60,))[0][:, 0]).cpu().numpy())
    return np.concatenate(values).astype(float)

