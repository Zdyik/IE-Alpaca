"""Fold-local V4 preprocessing and vehicle-normalized landmark likelihood."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ie_alpaca.features.landmark_v3 import ANCHORS
from ie_alpaca.features.landmark_v4 import EVENT_CODES, EVENT_METRICS, RISK_CODES, context_columns, event_columns
from ie_alpaca.models.event_hazard_v4 import EventHazardNet


class FoldScaler:
    def fit(self, frame: pd.DataFrame) -> "FoldScaler":
        self.context_names = context_columns()
        self.event_names = event_columns()
        raw = self._context_raw(frame)
        with np.errstate(all="ignore"):
            self.low = np.nanpercentile(raw, 1, axis=0)
            self.high = np.nanpercentile(raw, 99, axis=0)
        self.low = np.nan_to_num(self.low, nan=0.0)
        self.high = np.nan_to_num(self.high, nan=0.0)
        clipped = np.clip(raw, self.low, self.high)
        self.median = np.nan_to_num(np.nanmedian(clipped, axis=0), nan=0.0)
        filled = np.where(np.isfinite(clipped), clipped, self.median)
        self.scale = np.where(filled.std(axis=0) > 1e-6, filled.std(axis=0), 1.0)
        events = self._events_raw(frame)
        present = (events[..., 0] > 0) | (events[..., 5] > 0)
        if not present.any():
            raise ValueError("训练折没有任何历史事件")
        # Per-type scales keep a frequent alarm from shrinking rare alarm channels.
        self.event_scale = np.stack([
            np.maximum(np.percentile(events[present[:, i], i, :], 99, axis=0), 0.05)
            if present[:, i].any() else np.ones(len(EVENT_METRICS))
            for i in range(len(EVENT_CODES))
        ])
        return self

    def _context_raw(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame[self.context_names].to_numpy(dtype=float)
        raw[~np.isfinite(raw)] = np.nan
        return np.sign(raw) * np.log1p(np.abs(raw))

    def _events_raw(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame[self.event_names].to_numpy(dtype=float).reshape(-1, len(EVENT_CODES), len(EVENT_METRICS))
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError("事件输入含负数或非有限值")
        return np.log1p(raw)

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = np.clip(self._context_raw(frame), self.low, self.high)
        observed = np.isfinite(raw)
        filled = np.where(observed, raw, self.median)
        context = np.concatenate(((filled - self.median) / self.scale, (~observed).astype(float)), axis=1)
        events = self._events_raw(frame)
        present = ((events[..., 0] > 0) | (events[..., 5] > 0)).astype(np.float32)
        events = np.minimum(events, self.event_scale) / self.event_scale
        return events.astype(np.float32), present, context.astype(np.float32)

    def describe(self) -> dict:
        return {"context_columns": self.context_names, "event_columns": self.event_names,
                "context_low": self.low.tolist(), "context_high": self.high.tolist(),
                "context_median": self.median.tolist(), "context_scale": self.scale.tolist(),
                "event_scale": self.event_scale.tolist(),
                "fit_policy": "training vehicles and observed anchors only"}


def vehicle_batch(features: pd.DataFrame, outcomes: pd.DataFrame, ids: set[str], scaler: FoldScaler):
    rows = features[features.gpsno.isin(ids) & features.anchor_day.isin(ANCHORS)].merge(
        outcomes[["gpsno", "anchor_day", "first_event_day", "horizon_days", "label"]],
        on=["gpsno", "anchor_day"], validate="one_to_one",
    ).sort_values(["gpsno", "anchor_day"]).reset_index(drop=True)
    vehicles = sorted(ids)
    if len(rows) != len(vehicles) * len(ANCHORS) or rows.gpsno.unique().tolist() != vehicles:
        raise ValueError("每台训练车必须有完整的六个 landmark")
    if not np.tile(np.asarray(ANCHORS), len(vehicles)).tolist() == rows.anchor_day.tolist():
        raise ValueError("训练 landmark 顺序错误")
    event, present, context = scaler.transform(rows)
    shape = (len(vehicles), len(ANCHORS))
    return {
        "gpsno": vehicles,
        "events": event.reshape(*shape, len(EVENT_CODES), len(EVENT_METRICS)),
        "present": present.reshape(*shape, len(EVENT_CODES)),
        "context": context.reshape(*shape, -1),
        "event_day": rows.first_event_day.fillna(0).to_numpy(dtype=np.float32).reshape(shape),
        "horizon": rows.horizon_days.to_numpy(dtype=np.float32).reshape(shape),
        "label": rows.label.to_numpy(dtype=np.int8).reshape(shape),
    }


def next_event_loss(logit: torch.Tensor, event_day: torch.Tensor,
                    observed_days: torch.Tensor) -> torch.Tensor:
    """Each vehicle averages anchors; each anchor sums its observed risk days."""
    hit = event_day > 0
    negative_days = torch.where(hit, event_day - 1, observed_days)
    loss = negative_days * F.softplus(logit) + hit.float() * F.softplus(-logit)
    return loss.mean(dim=1).mean()


def resolve_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device 只能是 auto、cpu 或 cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 不支持 CUDA")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_network(pack: dict, config: dict, *, learned_weights: bool,
                seed: int, device: torch.device,
                network_factory=EventHazardNet) -> tuple[EventHazardNet, list[dict]]:
    set_seed(seed)
    events = torch.as_tensor(pack["events"], device=device)
    present = torch.as_tensor(pack["present"], device=device)
    context = torch.as_tensor(pack["context"], device=device)
    event_day = torch.as_tensor(pack["event_day"], device=device)
    horizon = torch.as_tensor(pack["horizon"], device=device)
    if not len(pack["gpsno"]) or np.sum(pack["label"][:, 1]) == 0:
        raise ValueError("训练车辆为空或 20→40 没有正例")
    risk_positions = [EVENT_CODES.index(code) for code in RISK_CODES]
    support = present[:, -1, risk_positions].sum(dim=0)
    rarity = 1 + 20 / (1 + support)
    weighted_positive = (event_day > 0).float().sum()
    weighted_days = torch.where(event_day > 0, event_day, horizon).sum()
    base_q = float((weighted_positive / weighted_days).clamp(1e-4, .1).cpu())
    net = network_factory(context.shape[-1], learned_weights=learned_weights,
                          dropout=float(config["dropout"]), base_daily_hazard=base_q).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    generator = np.random.default_rng(seed)
    history = []
    for epoch in range(int(config["epochs"])):
        net.train()
        losses = []
        for positions in np.array_split(generator.permutation(len(pack["gpsno"])),
                                        max(1, int(np.ceil(len(pack["gpsno"]) / config["batch_size"])))):
            ix = torch.as_tensor(positions, device=device)
            logit = net(events[ix], present[ix], context[ix])
            loss = next_event_loss(logit, event_day[ix], horizon[ix])
            loss = loss + net.type_penalty(rarity, float(config["type_penalty"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "train_objective": float(np.mean(losses))})
    return net, history


@torch.inference_mode()
def predict_daily(net: EventHazardNet, scaler: FoldScaler,
                  features: pd.DataFrame, device: torch.device) -> np.ndarray:
    net.eval()
    event, present, context = scaler.transform(features)
    outputs = []
    for start in range(0, len(features), 1024):
        stop = start + 1024
        logit = net(torch.as_tensor(event[start:stop], device=device),
                    torch.as_tensor(present[start:stop], device=device),
                    torch.as_tensor(context[start:stop], device=device))
        outputs.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)
