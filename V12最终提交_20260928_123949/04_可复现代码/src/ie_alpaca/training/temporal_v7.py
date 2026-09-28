"""Vehicle-normalized V7 survival training with inner vehicle epoch selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ie_alpaca.features.daily_v7 import DayScaler, TOKEN_COLUMNS
from ie_alpaca.features.landmark_v3 import ANCHORS
from ie_alpaca.models.temporal_transformer_v7 import TemporalHazardV7
from ie_alpaca.training.event_v4 import next_event_loss, set_seed


def vehicle_pack(day_table: pd.DataFrame, outcomes: pd.DataFrame,
                 ids: set[str], scaler: DayScaler) -> dict:
    vehicles = sorted(ids)
    rows = day_table[day_table.gpsno.isin(ids)].sort_values(["gpsno", "day_index"])
    if not vehicles or len(rows) != len(vehicles) * 60 or rows.gpsno.unique().tolist() != vehicles:
        raise ValueError("V7 每台车必须有完整的 60 天序列")
    if rows.day_index.tolist() != list(range(1, 61)) * len(vehicles):
        raise ValueError("V7 日序列顺序错误")
    labels = outcomes[outcomes.gpsno.isin(ids)].sort_values(["gpsno", "anchor_day"])
    if len(labels) != len(vehicles) * len(ANCHORS) or labels.gpsno.unique().tolist() != vehicles:
        raise ValueError("V7 每台车必须有完整的六个结局锚点")
    if labels.anchor_day.tolist() != list(ANCHORS) * len(vehicles):
        raise ValueError("V7 结局锚点顺序错误")
    shape = (len(vehicles), len(ANCHORS))
    return {"gpsno": vehicles,
            "x": scaler.transform(rows).reshape(len(vehicles), 60, len(TOKEN_COLUMNS)),
            "event_day": labels.first_event_day.fillna(0).to_numpy(dtype=np.float32).reshape(shape),
            "horizon": labels.horizon_days.to_numpy(dtype=np.float32).reshape(shape),
            "label": labels.label.to_numpy(dtype=np.int8).reshape(shape)}


def _tensors(pack: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(pack[key], device=device) for key in ("x", "event_day", "horizon")}


def _anchor_logits(net: TemporalHazardV7, x: torch.Tensor) -> torch.Tensor:
    vehicles = len(x)
    days = x.shape[1]
    expanded = x[:, None].expand(-1, len(ANCHORS), -1, -1).reshape(-1, days, x.shape[-1])
    mask = (torch.arange(days, device=x.device)[None, :] <
            torch.as_tensor(ANCHORS, device=x.device)[:, None])
    valid = mask[None].expand(vehicles, -1, -1).reshape(-1, days)
    return net(expanded, valid).reshape(vehicles, len(ANCHORS))


def _new_network(pack: dict, config: dict, mode: str, device: torch.device) -> TemporalHazardV7:
    net = TemporalHazardV7(
        len(TOKEN_COLUMNS), mode=mode, d_model=int(config["d_model"]),
        heads=int(config["heads"]), ff_dim=int(config["ff_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    event_day = torch.as_tensor(pack["event_day"], device=device)
    horizon = torch.as_tensor(pack["horizon"], device=device)
    positive = (event_day > 0).float().sum()
    observed = torch.where(event_day > 0, event_day, horizon).sum()
    base_q = (positive / observed).clamp(1e-4, .1)
    with torch.no_grad():
        net.hazard_head.bias.fill_(torch.logit(base_q).item())
    return net


def _fit_epochs(net: TemporalHazardV7, train: dict[str, torch.Tensor], config: dict,
                seed: int, *, max_epochs: int,
                val: dict[str, torch.Tensor] | None = None) -> tuple[list[dict], int]:
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    generator = np.random.default_rng(seed)
    n = len(train["x"])
    batch_size = int(config["batch_size"])
    history: list[dict] = []
    best_epoch, best_nll = 0, float("inf")
    for epoch in range(1, max_epochs + 1):
        net.train()
        losses = []
        for positions in np.array_split(generator.permutation(n), max(1, int(np.ceil(n / batch_size)))):
            ix = torch.as_tensor(positions, device=train["x"].device)
            logit = _anchor_logits(net, train["x"][ix])
            loss = next_event_loss(logit, train["event_day"][ix], train["horizon"][ix])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        entry = {"epoch": epoch, "train_objective": float(np.mean(losses))}
        if val is not None:
            net.eval()
            with torch.inference_mode():
                val_losses = []
                for start in range(0, len(val["x"]), batch_size):
                    stop = start + batch_size
                    logits = _anchor_logits(net, val["x"][start:stop])
                    val_losses.append(next_event_loss(
                        logits, val["event_day"][start:stop], val["horizon"][start:stop],
                    ).item() * len(val["x"][start:stop]))
                val_nll = float(sum(val_losses) / len(val["x"]))
            if not np.isfinite(val_nll):
                raise FloatingPointError("V7 inner validation loss is not finite")
            entry["inner_val_nll"] = val_nll
            if val_nll < best_nll - float(config["min_delta"]):
                best_epoch, best_nll = epoch, val_nll
            if epoch >= int(config["min_epochs"]) and epoch - best_epoch >= int(config["patience"]):
                history.append(entry)
                break
        history.append(entry)
    return history, best_epoch if val is not None else max_epochs


def select_epochs(train_pack: dict, val_pack: dict, config: dict, *,
                  mode: str, seed: int, device: torch.device) -> tuple[int, list[dict]]:
    if set(train_pack["gpsno"]) & set(val_pack["gpsno"]):
        raise ValueError("V7 inner fit and validation vehicles overlap")
    if not (1 <= int(config["min_epochs"]) <= int(config["max_epochs"]) and
            int(config["patience"]) >= 1 and float(config["min_delta"]) >= 0):
        raise ValueError("V7 early stopping configuration is invalid")
    set_seed(seed)
    net = _new_network(train_pack, config, mode, device)
    history, selected = _fit_epochs(net, _tensors(train_pack, device), config, seed,
                                    max_epochs=int(config["max_epochs"]),
                                    val=_tensors(val_pack, device))
    return selected, history


def fit_network(pack: dict, config: dict, *, mode: str, epochs: int,
                seed: int, device: torch.device) -> tuple[TemporalHazardV7, list[dict]]:
    if epochs < 1:
        raise ValueError("training epochs must be positive")
    set_seed(seed)
    net = _new_network(pack, config, mode, device)
    history, _ = _fit_epochs(net, _tensors(pack, device), config, seed, max_epochs=epochs)
    return net, history


@torch.inference_mode()
def predict_anchors(net: TemporalHazardV7, pack: dict,
                    device: torch.device, batch_vehicles: int) -> np.ndarray:
    net.eval()
    result = []
    for start in range(0, len(pack["gpsno"]), batch_vehicles):
        x = torch.as_tensor(pack["x"][start:start + batch_vehicles], device=device)
        result.append(torch.sigmoid(_anchor_logits(net, x)).cpu().numpy())
    return np.concatenate(result, axis=0).astype(np.float64)


@torch.inference_mode()
def predict_day60(net: TemporalHazardV7, x: np.ndarray,
                  device: torch.device, batch_vehicles: int) -> np.ndarray:
    net.eval()
    result = []
    for start in range(0, len(x), batch_vehicles):
        batch = torch.as_tensor(x[start:start + batch_vehicles], device=device)
        valid = torch.ones(batch.shape[:2], dtype=torch.bool, device=device)
        result.append(torch.sigmoid(net(batch, valid)).cpu().numpy())
    return np.concatenate(result).astype(np.float64)
