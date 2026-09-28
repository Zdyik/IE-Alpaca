"""Load the selected V12 model and estimate per-event prediction sensitivity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ie_alpaca.features.daily_v10 import CONTEXT_COLUMNS, NodeDayScaler
from ie_alpaca.features.daily_v12 import STATE_EVENT_CODES
from ie_alpaca.features.landmark_v4 import EVENT_CODES, EVENT_NAMES
from ie_alpaca.models.state_encoder_v12 import RiskNetV12
from ie_alpaca.training.landmark_v3 import horizon_probability


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[RiskNetV12, NodeDayScaler, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    settings = checkpoint["model_config"]
    method = checkpoint["method"]
    if method != "state_mae":
        raise ValueError("task-two scorecard requires the selected V12 State-MAE checkpoint")
    net = RiskNetV12(len(CONTEXT_COLUMNS), method, int(settings["width"]),
                     float(settings["dropout"])).to(device)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net, NodeDayScaler.from_description(checkpoint["scaler"]), checkpoint


@torch.inference_mode()
def predict_probabilities(net: RiskNetV12, events: np.ndarray, context: np.ndarray,
                          anchors: tuple[int, ...], device: torch.device,
                          batch_size: int) -> np.ndarray:
    values = []
    for start in range(0, len(events), batch_size):
        stop = start + batch_size
        event = torch.as_tensor(events[start:stop], device=device)
        ctx = torch.as_tensor(context[start:stop], device=device)
        hazards = torch.sigmoid(net(event, ctx, anchors)[0]).cpu().numpy()
        values.append(horizon_probability(hazards, 40))
    return np.concatenate(values).astype(float)


def event_sensitivity(net: RiskNetV12, events: np.ndarray, context: np.ndarray,
                      base_probability: np.ndarray, raw_counts: np.ndarray,
                      device: torch.device, batch_size: int) -> pd.DataFrame:
    state_lookup = {code: state for state, codes in STATE_EVENT_CODES.items() for code in codes}
    rows = []
    for position, code in enumerate(EVENT_CODES):
        ablated = events.copy()
        ablated[:, :, position, :5] = 0.0
        prediction = predict_probabilities(net, ablated, context, (60,), device, batch_size)[:, 0]
        delta = base_probability - prediction
        active = raw_counts[:, :, position].sum(axis=1) > 0
        selected = delta[active] if active.any() else delta
        rows.append({
            "event_code": int(code),
            "event_name": EVENT_NAMES[code],
            "state": state_lookup[code],
            "active_vehicles": int(active.sum()),
            "mean_signed_probability_delta": float(selected.mean()) if len(selected) else 0.0,
            "mean_abs_probability_delta": float(np.abs(selected).mean()) if len(selected) else 0.0,
        })
    return pd.DataFrame(rows)
