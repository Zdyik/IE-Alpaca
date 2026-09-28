"""Fold-only feature scaling and vehicle-normalized V9 exposure training."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ie_alpaca.features.landmark_v3 import ANCHORS
from ie_alpaca.models.exposure_v9 import ExposureRiskNetV9


class LandmarkScalerV9:
    def __init__(self, columns: list[str]):
        self.columns = columns

    def _raw(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.columns].to_numpy(dtype=float)
        values[~np.isfinite(values)] = np.nan
        return np.sign(values) * np.log1p(np.abs(values))

    def fit(self, frame: pd.DataFrame) -> "LandmarkScalerV9":
        raw = self._raw(frame)
        with np.errstate(all="ignore"):
            self.low = np.nan_to_num(np.nanpercentile(raw, 1, axis=0), nan=0.0)
            self.high = np.nan_to_num(np.nanpercentile(raw, 99, axis=0), nan=0.0)
        clipped = np.clip(raw, self.low, self.high)
        self.median = np.nan_to_num(np.nanmedian(clipped, axis=0), nan=0.0)
        filled = np.where(np.isfinite(clipped), clipped, self.median)
        self.scale = np.where(filled.std(axis=0) > 1e-6, filled.std(axis=0), 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = np.clip(self._raw(frame), self.low, self.high)
        observed = np.isfinite(raw)
        filled = np.where(observed, raw, self.median)
        return np.concatenate(((filled - self.median) / self.scale,
                               (~observed).astype(float)), axis=1).astype(np.float32)

    def describe(self) -> dict:
        return {"columns": self.columns, "low": self.low.tolist(), "high": self.high.tolist(),
                "median": self.median.tolist(), "scale": self.scale.tolist(),
                "fit_policy": "outer-training vehicles only"}


def add_future_exposure(daily: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(key): part.sort_values("day") for key, part in daily.groupby("gpsno", sort=False)}
    values = []
    for gpsno, anchor, horizon in outcomes[["gpsno", "anchor_day", "horizon_days"]].itertuples(index=False, name=None):
        future = lookup[str(gpsno)].iloc[int(anchor):int(anchor + horizon)]
        distance = pd.to_numeric(future.distance_km, errors="coerce").fillna(0).clip(lower=0).sum()
        values.append(float(distance) / int(horizon))
    result = outcomes.copy()
    result["future_km_per_day"] = values
    return result


def vehicle_pack(features: pd.DataFrame, outcomes: pd.DataFrame, ids: set[str],
                 scaler: LandmarkScalerV9) -> dict:
    rows = features[features.gpsno.isin(ids) & features.anchor_day.isin(ANCHORS)].merge(
        outcomes[["gpsno", "anchor_day", "first_event_day", "horizon_days", "label", "future_km_per_day"]],
        on=["gpsno", "anchor_day"], validate="one_to_one").sort_values(["gpsno", "anchor_day"])
    vehicles = sorted(ids)
    if len(rows) != len(vehicles) * len(ANCHORS):
        raise ValueError("每辆车必须有六个 exposure landmark")
    shape = (len(vehicles), len(ANCHORS))
    return {"gpsno": vehicles,
            "inputs": scaler.transform(rows).reshape(*shape, -1),
            "event_day": rows.first_event_day.fillna(0).to_numpy(np.float32).reshape(shape),
            "horizon": rows.horizon_days.to_numpy(np.float32).reshape(shape),
            "label": rows.label.to_numpy(np.int8).reshape(shape),
            "log_exposure": np.log1p(rows.future_km_per_day.to_numpy(np.float32)).reshape(shape)}


def survival_loss(q: torch.Tensor, event_day: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
    hit = event_day > 0
    negative_days = torch.where(hit, event_day - 1, horizon)
    loss = -negative_days * torch.log1p(-q) - hit.float() * torch.log(q)
    return loss.mean(dim=1).mean()


def objective(net: ExposureRiskNetV9, inputs: torch.Tensor, event_day: torch.Tensor,
              horizon: torch.Tensor, log_exposure: torch.Tensor, exposure_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    q, predicted = net(inputs)
    survival = survival_loss(q, event_day, horizon)
    if net.variant == "e4a_direct":
        return survival, survival
    auxiliary = F.smooth_l1_loss(predicted, log_exposure)
    return survival + exposure_weight * auxiliary, survival


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_exposure(pack: dict, config: dict, *, variant: str, epochs: int,
                 seed: int, device: torch.device) -> tuple[ExposureRiskNetV9, list[dict]]:
    _seed(seed)
    tensors = {key: torch.as_tensor(pack[key], device=device)
               for key in ("inputs", "event_day", "horizon", "log_exposure")}
    positive = (tensors["event_day"] > 0).float().sum()
    days = torch.where(tensors["event_day"] > 0, tensors["event_day"], tensors["horizon"]).sum()
    base_q = float((positive / days).clamp(1e-4, .1).cpu())
    base_exposure = float(tensors["log_exposure"].mean().cpu())
    net = ExposureRiskNetV9(tensors["inputs"].shape[-1], variant=variant,
                            dropout=config["dropout"], base_daily_hazard=base_q,
                            base_log_exposure=base_exposure).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    generator, history = np.random.default_rng(seed), []
    for epoch in range(1, epochs + 1):
        net.train(); losses = []
        batches = np.array_split(generator.permutation(len(pack["gpsno"])),
                                 max(1, int(np.ceil(len(pack["gpsno"]) / config["batch_size"]))))
        for batch in batches:
            index = torch.as_tensor(batch, device=device)
            total, survival = objective(net, *(tensors[key][index] for key in
                ("inputs", "event_day", "horizon", "log_exposure")), config["exposure_weight"])
            optimizer.zero_grad(set_to_none=True); total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
            losses.append((float(total.detach().cpu()), float(survival.detach().cpu())))
        history.append({"epoch": epoch, "train_objective": float(np.mean([x[0] for x in losses])),
                        "train_survival_nll": float(np.mean([x[1] for x in losses]))})
    return net, history


def select_exposure_epochs(train: dict, validation: dict, config: dict, *, variant: str,
                           seed: int, device: torch.device) -> tuple[int, list[dict]]:
    # Refit one epoch at a time here so selection behavior exactly matches final Adam updates.
    _seed(seed)
    tensors = lambda pack: {key: torch.as_tensor(pack[key], device=device)
                            for key in ("inputs", "event_day", "horizon", "log_exposure")}
    tr, va = tensors(train), tensors(validation)
    positive = (tr["event_day"] > 0).float().sum()
    days = torch.where(tr["event_day"] > 0, tr["event_day"], tr["horizon"]).sum()
    net = ExposureRiskNetV9(tr["inputs"].shape[-1], variant=variant, dropout=config["dropout"],
                            base_daily_hazard=float((positive / days).clamp(1e-4, .1).cpu()),
                            base_log_exposure=float(tr["log_exposure"].mean().cpu())).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    generator = np.random.default_rng(seed)
    best_epoch, best_nll, history = 0, float("inf"), []
    for epoch in range(1, config["max_epochs"] + 1):
        net.train(); losses = []
        batches = np.array_split(generator.permutation(len(train["gpsno"])),
                                 max(1, int(np.ceil(len(train["gpsno"]) / config["batch_size"]))))
        for batch in batches:
            ix = torch.as_tensor(batch, device=device)
            total, _ = objective(net, *(tr[key][ix] for key in
                ("inputs", "event_day", "horizon", "log_exposure")), config["exposure_weight"])
            optimizer.zero_grad(set_to_none=True); total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
            losses.append(float(total.detach().cpu()))
        net.eval()
        with torch.inference_mode():
            _, val_survival = objective(net, *(va[key] for key in
                ("inputs", "event_day", "horizon", "log_exposure")), config["exposure_weight"])
        nll = float(val_survival.cpu())
        history.append({"epoch": epoch, "inner_train_objective": float(np.mean(losses)),
                        "inner_val_survival_nll": nll})
        if nll < best_nll - config["min_delta"]:
            best_epoch, best_nll = epoch, nll
        if epoch >= config["min_epochs"] and epoch - best_epoch >= config["patience"]:
            break
    return best_epoch, history


@torch.inference_mode()
def predict_exposure(net: ExposureRiskNetV9, scaler: LandmarkScalerV9,
                     features: pd.DataFrame, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    net.eval(); inputs = scaler.transform(features); hazards, exposures = [], []
    for start in range(0, len(inputs), 1024):
        q, log_exp = net(torch.as_tensor(inputs[start:start + 1024], device=device))
        hazards.append(q.cpu().numpy()); exposures.append(torch.expm1(log_exp).cpu().numpy())
    return np.concatenate(hazards).astype(float), np.concatenate(exposures).astype(float)

