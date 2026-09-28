"""Vehicle-normalized V10 training and inner-fold epoch selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ie_alpaca.features.daily_v10 import AUX_GROUPS, NodeDayScaler, future_group_targets
from ie_alpaca.features.landmark_v3 import ANCHORS
from ie_alpaca.models.risk_chain_v10 import RiskChainNetV10
from ie_alpaca.training.event_v4 import next_event_loss, set_seed


def vehicle_pack(day_table: pd.DataFrame, outcomes: pd.DataFrame, ids: set[str],
                 scaler: NodeDayScaler) -> dict:
    vehicles = sorted(ids)
    rows = day_table[day_table.gpsno.isin(ids)].sort_values(["gpsno", "day_index"])
    if not vehicles or len(rows) != len(vehicles) * 60 or rows.gpsno.unique().tolist() != vehicles:
        raise ValueError("V10 every vehicle must have a complete 60-day sequence")
    labels = outcomes[outcomes.gpsno.isin(ids)].sort_values(["gpsno", "anchor_day"])
    if len(labels) != len(vehicles) * len(ANCHORS) or labels.anchor_day.tolist() != list(ANCHORS) * len(vehicles):
        raise ValueError("V10 every vehicle must have six ordered outcomes")
    event, context = scaler.transform(rows)
    shape = (len(vehicles), len(ANCHORS))
    return {"gpsno": vehicles,
            "events": event.reshape(len(vehicles), 60, event.shape[-2], event.shape[-1]),
            "context": context.reshape(len(vehicles), 60, context.shape[-1]),
            "aux_targets": future_group_targets(day_table, vehicles),
            "event_day": labels.first_event_day.fillna(0).to_numpy(np.float32).reshape(shape),
            "horizon": labels.horizon_days.to_numpy(np.float32).reshape(shape),
            "label": labels.label.to_numpy(np.int8).reshape(shape)}


def _tensors(pack: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(pack[name], device=device)
            for name in ("events", "context", "aux_targets", "event_day", "horizon")}


def _new_network(pack: dict, config: dict, mode: str, device: torch.device) -> RiskChainNetV10:
    event_day = torch.as_tensor(pack["event_day"], device=device)
    horizon = torch.as_tensor(pack["horizon"], device=device)
    positive = (event_day > 0).float().sum()
    observed = torch.where(event_day > 0, event_day, horizon).sum()
    base_q = float((positive / observed).clamp(1e-4, .1).cpu())
    return RiskChainNetV10(
        context_width=pack["context"].shape[-1], mode=mode,
        width=int(config["width"]), event_embedding=int(config["event_embedding"]),
        role_embedding=int(config["role_embedding"]), stage_embedding=int(config["stage_embedding"]),
        dropout=float(config["dropout"]), modality_dropout=float(config["modality_dropout"]),
        base_daily_hazard=base_q,
    ).to(device)


def _auxiliary_loss(logits: torch.Tensor, targets: torch.Tensor,
                    positive_weight: torch.Tensor) -> torch.Tensor:
    # Days 1..46 have a complete seven-day target within the day-53 training prefix.
    logits, targets = logits[:, :46], targets[:, :46]
    element = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=positive_weight)
    return element.mean(dim=(1, 2)).mean()


def _positive_weight(targets: torch.Tensor) -> torch.Tensor:
    values = targets[:, :46]
    positive = values.sum(dim=(0, 1))
    total = values.shape[0] * values.shape[1]
    return ((total - positive) / positive.clamp_min(1)).clamp(1, 20)


def _fit_epochs(net: RiskChainNetV10, train: dict[str, torch.Tensor], config: dict,
                seed: int, max_epochs: int,
                val: dict[str, torch.Tensor] | None = None) -> tuple[list[dict], int]:
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    generator = np.random.default_rng(seed)
    batch_size, history = int(config["batch_size"]), []
    best_epoch, best_nll = 0, float("inf")
    pos_weight = _positive_weight(train["aux_targets"])
    for epoch in range(1, max_epochs + 1):
        net.train(); losses, main_losses, aux_losses = [], [], []
        batches = np.array_split(generator.permutation(len(train["events"])),
                                 max(1, int(np.ceil(len(train["events"]) / batch_size))))
        for positions in batches:
            index = torch.as_tensor(positions, device=train["events"].device)
            logits, auxiliary, _ = net(train["events"][index], train["context"][index], ANCHORS)
            main = next_event_loss(logits, train["event_day"][index], train["horizon"][index])
            aux = (torch.zeros((), device=main.device) if auxiliary is None else
                   _auxiliary_loss(auxiliary, train["aux_targets"][index], pos_weight))
            loss = main + float(config["auxiliary_weight"]) * aux
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu())); main_losses.append(float(main.detach().cpu()))
            aux_losses.append(float(aux.detach().cpu()))
        entry = {"epoch": epoch, "train_objective": float(np.mean(losses)),
                 "train_survival_nll": float(np.mean(main_losses)),
                 "train_auxiliary_loss": float(np.mean(aux_losses))}
        if val is not None:
            net.eval(); weighted = []
            with torch.inference_mode():
                for start in range(0, len(val["events"]), batch_size):
                    stop = start + batch_size
                    logits, _, _ = net(val["events"][start:stop], val["context"][start:stop], ANCHORS)
                    nll = next_event_loss(logits, val["event_day"][start:stop], val["horizon"][start:stop])
                    weighted.append(float(nll.cpu()) * len(val["events"][start:stop]))
            val_nll = float(sum(weighted) / len(val["events"]))
            entry["inner_val_nll"] = val_nll
            # The warm-up floor prevents a lucky first few epochs from being
            # selected simply because the network still predicts the base rate.
            if epoch >= int(config["min_epochs"]) and val_nll < best_nll - float(config["min_delta"]):
                best_epoch, best_nll = epoch, val_nll
            if epoch >= int(config["min_epochs"]) and epoch - best_epoch >= int(config["patience"]):
                history.append(entry); break
        history.append(entry)
    return history, best_epoch if val is not None else max_epochs


def select_epochs(train_pack: dict, val_pack: dict, config: dict, *, mode: str,
                  seed: int, device: torch.device) -> tuple[int, list[dict]]:
    if set(train_pack["gpsno"]) & set(val_pack["gpsno"]):
        raise ValueError("V10 inner fit and validation vehicles overlap")
    set_seed(seed)
    net = _new_network(train_pack, config, mode, device)
    history, selected = _fit_epochs(net, _tensors(train_pack, device), config, seed,
                                    int(config["max_epochs"]), _tensors(val_pack, device))
    if selected < 1:
        raise RuntimeError("V10 did not select a valid epoch")
    return selected, history


def fit_network(pack: dict, config: dict, *, mode: str, epochs: int,
                seed: int, device: torch.device) -> tuple[RiskChainNetV10, list[dict]]:
    set_seed(seed)
    net = _new_network(pack, config, mode, device)
    history, _ = _fit_epochs(net, _tensors(pack, device), config, seed, epochs)
    return net, history


@torch.inference_mode()
def predict_anchors(net: RiskChainNetV10, pack: dict, device: torch.device,
                    batch_vehicles: int) -> tuple[np.ndarray, np.ndarray]:
    net.eval(); hazards, gates = [], []
    for start in range(0, len(pack["gpsno"]), batch_vehicles):
        events = torch.as_tensor(pack["events"][start:start + batch_vehicles], device=device)
        context = torch.as_tensor(pack["context"][start:start + batch_vehicles], device=device)
        logits, _, details = net(events, context, ANCHORS)
        hazards.append(torch.sigmoid(logits).cpu().numpy())
        gates.append(details["event_gates"].mean(dim=1).cpu().numpy())
    return np.concatenate(hazards).astype(float), np.concatenate(gates).astype(float)


@torch.inference_mode()
def predict_day60(net: RiskChainNetV10, events: np.ndarray, context: np.ndarray,
                  device: torch.device, batch_vehicles: int) -> tuple[np.ndarray, np.ndarray]:
    net.eval(); hazards, gates = [], []
    for start in range(0, len(events), batch_vehicles):
        event = torch.as_tensor(events[start:start + batch_vehicles], device=device)
        ctx = torch.as_tensor(context[start:start + batch_vehicles], device=device)
        logits, _, details = net(event, ctx, (60,))
        hazards.append(torch.sigmoid(logits[:, 0]).cpu().numpy())
        gates.append(details["event_gates"].mean(dim=1).cpu().numpy())
    return np.concatenate(hazards).astype(float), np.concatenate(gates).astype(float)
