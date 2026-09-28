"""Frozen probe, risk fine-tuning and inference for V11."""

from __future__ import annotations

import copy
import numpy as np
import torch
from torch.nn import functional as F

from ie_alpaca.features.daily_v10 import AUX_GROUPS
from ie_alpaca.features.landmark_v3 import ANCHORS
from ie_alpaca.models.representation_v11 import RiskNetV11
from ie_alpaca.training.event_v4 import next_event_loss, set_seed


def _tensors(pack: dict, device: torch.device):
    return {k: torch.as_tensor(pack[k], device=device)
            for k in ("events", "context", "aux_targets", "event_day", "horizon")}


def _base_hazard(pack: dict) -> float:
    event = pack["event_day"]; horizon = pack["horizon"]
    positive = (event > 0).sum(); observed = np.where(event > 0, event, horizon).sum()
    return float(np.clip(positive / max(observed, 1), 1e-4, .1))


def _positive_weight(targets: torch.Tensor):
    values = targets[:, :46]; positive = values.sum(dim=(0, 1)); total = values.shape[0] * values.shape[1]
    return ((total - positive) / positive.clamp_min(1)).clamp(1, 20)


def _loss(net, data, index, config, pos_weight):
    logits, auxiliary, _ = net(data["events"][index], data["context"][index], ANCHORS)
    main = next_event_loss(logits, data["event_day"][index], data["horizon"][index])
    aux = F.binary_cross_entropy_with_logits(auxiliary[:, :46], data["aux_targets"][index, :46],
                                             pos_weight=pos_weight)
    return main + float(config["auxiliary_weight"]) * aux, main, aux


def _validation_nll(net, data, batch_size):
    net.eval(); total = 0.
    with torch.inference_mode():
        for start in range(0, len(data["events"]), batch_size):
            stop = start + batch_size
            logits, _, _ = net(data["events"][start:stop], data["context"][start:stop], ANCHORS)
            total += float(next_event_loss(logits, data["event_day"][start:stop], data["horizon"][start:stop]).cpu()) * len(data["events"][start:stop])
    return total / len(data["events"])


def fit_risk(pack: dict, config: dict, encoder_state: dict, seed: int, device: torch.device,
             epochs: int | None = None, validation_pack: dict | None = None):
    """Run the fixed frozen probe then full fine-tune; optional validation selects epochs."""
    set_seed(seed)
    net = RiskNetV11(pack["context"].shape[-1], int(config["width"]), float(config["dropout"]), _base_hazard(pack)).to(device)
    net.encoder.load_state_dict(encoder_state)
    train = _tensors(pack, device); val = None if validation_pack is None else _tensors(validation_pack, device)
    batch_size = int(config["finetune_batch_size"]); rng = np.random.default_rng(seed)
    pos_weight = _positive_weight(train["aux_targets"])
    for parameter in net.encoder.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                                  lr=float(config["head_learning_rate"]), weight_decay=float(config["weight_decay"]))
    history = []
    for epoch in range(1, int(config["probe_epochs"]) + 1):
        net.train(); losses = []
        for positions in np.array_split(rng.permutation(len(train["events"])), max(1, int(np.ceil(len(train["events"]) / batch_size)))):
            index = torch.as_tensor(positions, device=device); loss, _, _ = _loss(net, train, index, config, pos_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        entry = {"phase": "probe", "epoch": epoch, "train_objective": float(np.mean(losses))}
        if val is not None: entry["inner_val_nll"] = _validation_nll(net, val, batch_size)
        history.append(entry)
    probe_state = copy.deepcopy(net.state_dict())
    for parameter in net.encoder.parameters(): parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {"params": net.encoder.parameters(), "lr": float(config["encoder_learning_rate"])},
        {"params": [p for name, p in net.named_parameters() if not name.startswith("encoder.")],
         "lr": float(config["head_learning_rate"])},
    ], weight_decay=float(config["weight_decay"]))
    limit = int(config["max_finetune_epochs"] if epochs is None else epochs)
    best_epoch, best_nll, best_state = 0, float("inf"), None
    for epoch in range(1, limit + 1):
        net.train(); losses, mains, auxes = [], [], []
        for positions in np.array_split(rng.permutation(len(train["events"])), max(1, int(np.ceil(len(train["events"]) / batch_size)))):
            index = torch.as_tensor(positions, device=device); loss, main, aux = _loss(net, train, index, config, pos_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.); optimizer.step()
            losses.append(float(loss.detach().cpu())); mains.append(float(main.detach().cpu())); auxes.append(float(aux.detach().cpu()))
        entry = {"phase": "finetune", "epoch": epoch, "train_objective": float(np.mean(losses)),
                 "train_survival_nll": float(np.mean(mains)), "train_auxiliary_loss": float(np.mean(auxes))}
        if val is not None:
            value = _validation_nll(net, val, batch_size); entry["inner_val_nll"] = value
            if epoch >= int(config["min_finetune_epochs"]) and value < best_nll - float(config["min_delta"]):
                best_epoch, best_nll, best_state = epoch, value, copy.deepcopy(net.state_dict())
            if epoch >= int(config["min_finetune_epochs"]) and best_epoch and epoch - best_epoch >= int(config["finetune_patience"]):
                history.append(entry); break
        history.append(entry)
    if validation_pack is not None:
        if best_state is None:
            best_epoch, best_state = limit, copy.deepcopy(net.state_dict())
        net.load_state_dict(best_state)
    return net, history, (best_epoch if validation_pack is not None else limit), probe_state


@torch.inference_mode()
def predict_anchors(net: RiskNetV11, pack: dict, device: torch.device, batch_size: int):
    net.eval(); values = []
    for start in range(0, len(pack["events"]), batch_size):
        event = torch.as_tensor(pack["events"][start:start + batch_size], device=device)
        context = torch.as_tensor(pack["context"][start:start + batch_size], device=device)
        values.append(torch.sigmoid(net(event, context, ANCHORS)[0]).cpu().numpy())
    return np.concatenate(values).astype(float)


@torch.inference_mode()
def predict_day60(net: RiskNetV11, events: np.ndarray, context: np.ndarray, device: torch.device, batch_size: int):
    net.eval(); values = []
    for start in range(0, len(events), batch_size):
        event = torch.as_tensor(events[start:start + batch_size], device=device)
        ctx = torch.as_tensor(context[start:start + batch_size], device=device)
        values.append(torch.sigmoid(net(event, ctx, (60,))[0][:, 0]).cpu().numpy())
    return np.concatenate(values).astype(float)

