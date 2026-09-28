"""Leakage-safe masked-state and risk-chain self-supervision for V12."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from ie_alpaca.features.daily_v11 import apply_masks, context_block_mask, event_bundle_mask
from ie_alpaca.features.daily_v12 import RISK_GROUP_POSITIONS, STATE_NAMES, state_block_mask, state_masks_to_inputs
from ie_alpaca.features.daily_v7 import NUMERIC_COLUMNS, QUALITY_COLUMNS
from ie_alpaca.models.state_encoder_v12 import TransitionPredictorV12
from ie_alpaca.models.state_mae_v12 import SSLModelV12, clone_backbone_state
from ie_alpaca.training.event_v4 import set_seed


def _effective_rank(values: torch.Tensor) -> float:
    flat = values.float().reshape(-1, values.shape[-1])
    flat = flat - flat.mean(0)
    singular = torch.linalg.svdvals(flat).clamp_min(1e-12)
    probability = singular / singular.sum()
    return float(torch.exp(-(probability * probability.log()).sum()).cpu())


def _batch(pack: dict, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    index = rng.integers(0, len(pack["events"]), size=size)
    return pack["events"][index], pack["context"][index]


def _future_targets(events: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Future U/C/P occurrence and intensity; severe events are intentionally absent."""
    if horizon < 1 or events.shape[1] <= horizon:
        raise ValueError("invalid V12 future horizon")
    daily_occurrence, daily_magnitude = [], []
    for positions in RISK_GROUP_POSITIONS:
        daily_occurrence.append(events[..., list(positions), 4].amax(dim=-1))
        daily_magnitude.append(events[..., list(positions), 0].amax(dim=-1))
    occurrence = torch.stack(daily_occurrence, dim=-1)
    magnitude = torch.stack(daily_magnitude, dim=-1)
    future_occurrence = F.max_pool1d(occurrence[:, 1:].transpose(1, 2), horizon, stride=1).transpose(1, 2)
    future_magnitude = F.max_pool1d(magnitude[:, 1:].transpose(1, 2), horizon, stride=1).transpose(1, 2)
    return future_occurrence, future_magnitude


def _positive_weights(events: np.ndarray, horizons: tuple[int, ...]) -> torch.Tensor:
    tensor = torch.as_tensor(events)
    rows = []
    for horizon in horizons:
        occurrence, _ = _future_targets(tensor, horizon)
        positive = occurrence.sum(dim=(0, 1))
        total = occurrence.shape[0] * occurrence.shape[1]
        rows.append(((total - positive) / positive.clamp_min(1)).clamp(1, 20))
    return torch.stack(rows)


def _make_masks(events: np.ndarray, context: np.ndarray, rng: np.random.Generator):
    state_mask = state_block_mask((events.shape[0], events.shape[1], len(STATE_NAMES)), rng)
    state_event, state_context = state_masks_to_inputs(state_mask)
    random_event = event_bundle_mask(events.shape[:3], rng, day_fraction=.10, group_fraction=.08)
    random_context = context_block_mask(context.shape, rng, fraction=.08)
    return state_mask, state_event | random_event, state_context | random_context


def _mask_losses(model: SSLModelV12, events: np.ndarray, context: np.ndarray,
                 rng: np.random.Generator, device: torch.device):
    state_mask, event_mask, context_mask = _make_masks(events, context, rng)
    masked_event, masked_context = apply_masks(events, context, event_mask, context_mask)
    event = torch.as_tensor(events, device=device)
    ctx = torch.as_tensor(context, device=device)
    em = torch.as_tensor(event_mask, device=device)
    cm = torch.as_tensor(context_mask, device=device)
    sm = torch.as_tensor(state_mask, device=device)
    encoded = model.online(torch.as_tensor(masked_event, device=device),
                           torch.as_tensor(masked_context, device=device), em, cm)
    with torch.no_grad():
        target = model.target(event, ctx)["states"]
    event_prediction, context_prediction = model.decoder(encoded["states"], encoded["assignment"])
    occurrence = F.binary_cross_entropy_with_logits(event_prediction[..., 4][em], event[..., 4][em])
    positive = em[..., None] & (event[..., 4:5] > 0)
    positive = positive.expand_as(event[..., :4])
    magnitude = (F.smooth_l1_loss(event_prediction[..., :4][positive], event[..., :4][positive])
                 if positive.any() else occurrence * 0)
    context_valid = cm.clone()
    numeric_count, quality_count = len(NUMERIC_COLUMNS), len(QUALITY_COLUMNS)
    original_missing = ctx[..., numeric_count + quality_count:numeric_count + quality_count + numeric_count] > .5
    context_valid[..., :numeric_count] &= ~original_missing
    context_loss = F.smooth_l1_loss(context_prediction[context_valid], ctx[context_valid])
    student_state = F.layer_norm(encoded["states"][sm], (encoded["states"].shape[-1],))
    target_state = F.layer_norm(target[sm], (target.shape[-1],))
    state_loss = F.smooth_l1_loss(student_state, target_state)
    assignment = model.online.encoder.assignment_loss()
    mask_loss = occurrence + .5 * magnitude + .5 * context_loss + .25 * state_loss + .01 * assignment
    return mask_loss, encoded["states"], target, {
        "event_occurrence": occurrence, "event_magnitude": magnitude,
        "context": context_loss, "state_recovery": state_loss, "assignment": assignment,
    }


def _dynamics_losses(model: SSLModelV12, student: torch.Tensor, target: torch.Tensor,
                     events: torch.Tensor, positive_weights: torch.Tensor,
                     progress: float):
    zero = student.sum() * 0
    if not model.online.transition.uses_dynamics:
        return zero, zero, zero, zero, {"graph_sparsity": zero, "graph_prior": zero}
    transition_losses, event_losses, predictions = [], [], {}
    for horizon_index, horizon in enumerate(TransitionPredictorV12.HORIZONS):
        source = student[:, :-horizon]
        prediction = model.online.transition(source, horizon_index)
        target_delta = (target[:, horizon:, list(model.online.encoder.risk_indices)] -
                        target[:, :-horizon, list(model.online.encoder.risk_indices)]).detach()
        transition_losses.append(F.smooth_l1_loss(prediction, target_delta))
        future_state = source[:, :, list(model.online.encoder.risk_indices)] + prediction
        future_prediction = model.decode_future(future_state)
        occurrence, magnitude = _future_targets(events, horizon)
        occurrence_loss = F.binary_cross_entropy_with_logits(
            future_prediction[..., 0], occurrence,
            pos_weight=positive_weights[horizon_index].to(events.device))
        positive = occurrence > 0
        magnitude_loss = (F.smooth_l1_loss(future_prediction[..., 1][positive], magnitude[positive])
                          if positive.any() else occurrence_loss * 0)
        event_losses.append(occurrence_loss + .5 * magnitude_loss)
        predictions[horizon] = prediction
    transition_loss = torch.stack(transition_losses).mean()
    future_event_loss = torch.stack(event_losses).mean()
    common = min(value.shape[1] for value in predictions.values())
    velocity1 = predictions[1][:, :common]
    velocity3 = predictions[3][:, :common] / 3.0
    velocity7 = predictions[7][:, :common] / 7.0
    path = .5 * (F.smooth_l1_loss(velocity1, velocity3) + F.smooth_l1_loss(velocity3, velocity7))
    graph, sparsity, prior = model.online.transition.graph_loss(progress)
    return transition_loss, future_event_loss, path, graph, {
        "graph_sparsity": sparsity, "graph_prior": prior,
    }


@torch.inference_mode()
def _diagnostics(model: SSLModelV12, pack: dict, device: torch.device) -> dict:
    model.eval()
    event = torch.as_tensor(pack["events"][:128], device=device)
    context = torch.as_tensor(pack["context"][:128], device=device)
    encoded = model.online(event, context)
    states = encoded["states"]
    per_state = {}
    for index, name in enumerate(STATE_NAMES):
        values = states[:, :, index]
        per_state[name] = {
            "mean_dimension_std": float(values.reshape(-1, values.shape[-1]).std(0).mean().cpu()),
            "effective_rank": _effective_rank(values),
        }
    assignment = encoded["assignment"].cpu().numpy()
    return {
        "per_state": per_state,
        "assignment_argmax": assignment.argmax(axis=1).tolist(),
        "assignment_entropy_mean": float((-(encoded["assignment"] * encoded["assignment"].clamp_min(1e-9).log()).sum(-1)).mean().cpu()),
        "edge_weights": model.online.transition.edge_weights().detach().cpu().tolist(),
        "transition_beta": float(torch.tanh(model.online.transition.beta).cpu()),
    }


def pretrain_v12(pack: dict, config: dict, method: str, steps: int, seed: int,
                 device: torch.device) -> tuple[dict[str, torch.Tensor], list[dict], dict]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = SSLModelV12(pack["context"].shape[-1], method, int(config["width"]),
                       float(config["dropout"])).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config["ssl_learning_rate"]),
                                  weight_decay=float(config["ssl_weight_decay"]))
    horizons = TransitionPredictorV12.HORIZONS
    positive_weights = _positive_weights(pack["events"], horizons).to(device)
    history: list[dict] = []
    for step in range(1, steps + 1):
        model.train()
        event_np, context_np = _batch(pack, rng, int(config["ssl_batch_size"]))
        mask_loss, student, target, mask_parts = _mask_losses(model, event_np, context_np, rng, device)
        event = torch.as_tensor(event_np, device=device)
        transition, future_event, path, graph, graph_parts = _dynamics_losses(
            model, student, target, event, positive_weights, step / max(1, steps))
        loss = mask_loss + .5 * transition + .5 * future_event + .05 * path + graph
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).detach().cpu())
        optimizer.step()
        momentum = .99 + (.999 - .99) * step / max(1, steps)
        model.update_target(momentum)
        if step == 1 or step % int(config["ssl_check_every"]) == 0 or step == steps:
            parts = {**mask_parts, **graph_parts, "transition": transition,
                     "future_event": future_event, "path": path, "graph": graph}
            history.append({
                "step": step, "train_ssl_loss": float(loss.detach().cpu()),
                "gradient_norm": gradient, "ema_momentum": momentum,
                **{name: float(value.detach().cpu()) for name, value in parts.items()},
            })
    diagnostics = _diagnostics(model, pack, device)
    diagnostics.update({"method": method, "steps": steps,
                        "future_positive_weights": positive_weights.cpu().tolist()})
    return clone_backbone_state(model), history, diagnostics

