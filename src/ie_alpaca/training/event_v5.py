"""Vehicle-isolated inner epoch selection for the unchanged V4 event network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from ie_alpaca.features.landmark_v4 import EVENT_CODES, RISK_CODES
from ie_alpaca.models.event_hazard_v4 import EventHazardNet
from ie_alpaca.training.event_v4 import next_event_loss, set_seed


def inner_vehicle_split(ids: set[str], labels: pd.Series, *, seed: int,
                        validation_fraction: float) -> tuple[set[str], set[str]]:
    """Use only outer-training vehicles and their day-20 labels."""
    if not 0 < validation_fraction < .5:
        raise ValueError("inner validation fraction must be in (0, 0.5)")
    vehicles = np.asarray(sorted(ids))
    y = labels.loc[vehicles].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("inner split requires both classes")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction,
                                      random_state=seed)
    fit_ix, val_ix = next(splitter.split(vehicles, y))
    fit_ids, val_ids = set(vehicles[fit_ix]), set(vehicles[val_ix])
    if fit_ids & val_ids or fit_ids | val_ids != ids:
        raise RuntimeError("inner vehicle split overlaps or misses vehicles")
    return fit_ids, val_ids


def select_epochs(train_pack: dict, val_pack: dict, config: dict, *,
                  learned_weights: bool, seed: int, device: torch.device,
                  network_factory=EventHazardNet) -> tuple[int, list[dict]]:
    """Select the minimum observed inner validation NLL; outer fold is untouched."""
    if set(train_pack["gpsno"]) & set(val_pack["gpsno"]):
        raise ValueError("inner training and validation vehicles overlap")
    max_epochs = int(config["max_epochs"])
    min_epochs = int(config["min_epochs"])
    patience = int(config["patience"])
    min_delta = float(config["min_delta"])
    if not (1 <= min_epochs <= max_epochs and patience >= 1 and min_delta >= 0):
        raise ValueError("invalid early stopping configuration")
    set_seed(seed)

    def tensors(pack: dict) -> dict[str, torch.Tensor]:
        return {name: torch.as_tensor(pack[name], device=device)
                for name in ("events", "present", "context", "event_day", "horizon")}

    train, val = tensors(train_pack), tensors(val_pack)
    if not len(train_pack["gpsno"]) or not len(val_pack["gpsno"]):
        raise ValueError("empty inner train or validation set")
    if np.sum(train_pack["label"][:, 1]) == 0:
        raise ValueError("inner training has no day-20 positive vehicle")
    positions = [EVENT_CODES.index(code) for code in RISK_CODES]
    support = train["present"][:, -1, positions].sum(dim=0)
    rarity = 1 + 20 / (1 + support)
    positive = (train["event_day"] > 0).float().sum()
    observed = torch.where(train["event_day"] > 0, train["event_day"], train["horizon"]).sum()
    base_q = float((positive / observed).clamp(1e-4, .1).cpu())
    net = network_factory(train["context"].shape[-1], learned_weights=learned_weights,
                          dropout=float(config["dropout"]), base_daily_hazard=base_q).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    generator = np.random.default_rng(seed)
    history: list[dict] = []
    best_epoch, best_nll = 0, float("inf")
    for epoch in range(1, max_epochs + 1):
        net.train()
        losses = []
        batches = np.array_split(generator.permutation(len(train_pack["gpsno"])),
                                 max(1, int(np.ceil(len(train_pack["gpsno"]) / config["batch_size"]))))
        for batch in batches:
            ix = torch.as_tensor(batch, device=device)
            logits = net(train["events"][ix], train["present"][ix], train["context"][ix])
            objective = next_event_loss(logits, train["event_day"][ix], train["horizon"][ix])
            objective = objective + net.type_penalty(rarity, float(config["type_penalty"]))
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            losses.append(float(objective.detach().cpu()))
        net.eval()
        with torch.inference_mode():
            logits = net(val["events"], val["present"], val["context"])
            val_nll = float(next_event_loss(logits, val["event_day"], val["horizon"]).cpu())
        if not np.isfinite(val_nll):
            raise FloatingPointError("nonfinite inner validation loss")
        history.append({"epoch": epoch, "inner_train_objective": float(np.mean(losses)),
                        "inner_val_nll": val_nll})
        if val_nll < best_nll - min_delta:
            best_epoch, best_nll = epoch, val_nll
        if epoch >= min_epochs and epoch - best_epoch >= patience:
            break
    if best_epoch < 1:
        raise RuntimeError("no epoch selected")
    return best_epoch, history
