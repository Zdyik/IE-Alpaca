"""Leakage-safe self-supervised pretraining for V11."""

from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ie_alpaca.features.daily_v11 import apply_masks, event_bundle_mask, context_block_mask, light_view_masks
from ie_alpaca.models.representation_v11 import RepresentationEncoderV11
from ie_alpaca.models.ssl_heads_v11 import ContrastiveHeadV11, JEPAV11, MaskedHeadV11
from ie_alpaca.training.event_v4 import set_seed


def _effective_rank(z: torch.Tensor) -> float:
    z = z.float().reshape(-1, z.shape[-1]); z = z - z.mean(0)
    values = torch.linalg.svdvals(z).clamp_min(1e-12)
    probabilities = values / values.sum()
    return float(torch.exp(-(probabilities * probabilities.log()).sum()).cpu())


def _diagnostics(encoder: RepresentationEncoderV11, events: np.ndarray, context: np.ndarray,
                 device: torch.device) -> dict:
    encoder.eval()
    with torch.inference_mode():
        encoded = encoder(torch.as_tensor(events[:128], device=device), torch.as_tensor(context[:128], device=device))["public"]
    flat = encoded.reshape(-1, encoded.shape[-1])
    return {"mean_dimension_std": float(flat.std(0).mean().cpu()), "minimum_dimension_std": float(flat.std(0).min().cpu()),
            "effective_rank": _effective_rank(flat)}


def _infonce(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(len(a), device=a.device)
    return .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _variance_covariance(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = z.reshape(-1, z.shape[-1]); centered = z - z.mean(0)
    std = torch.sqrt(centered.var(0) + 1e-4)
    variance = F.relu(1 - std).mean()
    covariance = centered.T @ centered / max(1, len(z) - 1)
    covariance.fill_diagonal_(0)
    return variance, covariance.square().sum() / z.shape[-1]


def _batch(pack: dict, rng: np.random.Generator, size: int, days: int = 53):
    index = rng.integers(0, len(pack["events"]), size=size)
    return pack["events"][index, :days], pack["context"][index, :days]


def _masked_step(encoder, head, events, context, rng, device):
    em = event_bundle_mask(events.shape[:3], rng); cm = context_block_mask(context.shape, rng)
    me, mc = apply_masks(events, context, em, cm)
    output = encoder(torch.as_tensor(me, device=device), torch.as_tensor(mc, device=device),
                     torch.as_tensor(em, device=device), torch.as_tensor(cm, device=device))["public"]
    event_pred, context_pred = head(output)
    event_true = torch.as_tensor(events, device=device); context_true = torch.as_tensor(context, device=device)
    event_mask = torch.as_tensor(em, device=device); context_mask = torch.as_tensor(cm, device=device)
    occurrence = F.binary_cross_entropy_with_logits(event_pred[..., 4][event_mask], event_true[..., 4][event_mask])
    positive = event_mask[..., None] & (event_true[..., 4:5] > 0) & torch.ones_like(event_true[..., :4], dtype=torch.bool)
    magnitude = F.smooth_l1_loss(event_pred[..., :4][positive], event_true[..., :4][positive]) if positive.any() else occurrence * 0
    ctx = F.smooth_l1_loss(context_pred[context_mask], context_true[context_mask])
    return occurrence + .5 * magnitude + .5 * ctx, {"occurrence": occurrence, "magnitude": magnitude, "context": ctx}


def _contrast_step(encoder, head, events, context, rng, device, temperature):
    masks1 = light_view_masks(events.shape[:3], context.shape, rng)
    masks2 = light_view_masks(events.shape[:3], context.shape, rng)
    e1, c1 = apply_masks(events, context, *masks1); e2, c2 = apply_masks(events, context, *masks2)
    z1 = encoder(torch.as_tensor(e1, device=device), torch.as_tensor(c1, device=device),
                 torch.as_tensor(masks1[0], device=device), torch.as_tensor(masks1[1], device=device))["public"]
    z2 = encoder(torch.as_tensor(e2, device=device), torch.as_tensor(c2, device=device),
                 torch.as_tensor(masks2[0], device=device), torch.as_tensor(masks2[1], device=device))["public"]
    stable1, acute1 = head(z1); stable2, acute2 = head(z2)
    stable, acute = _infonce(stable1, stable2, temperature), _infonce(acute1, acute2, temperature)
    return .5 * (stable + acute), {"stable": stable, "acute": acute}


def _jepa_step(model, events, context, rng, device, horizons, step, total_steps):
    h_index = int(rng.integers(0, len(horizons))); horizon = int(horizons[h_index])
    cut = int(rng.integers(14, 54 - horizon))
    event = torch.as_tensor(events, device=device); ctx = torch.as_tensor(context, device=device)
    online = model.online(event[:, :cut], ctx[:, :cut])["public"][:, -1]
    with torch.no_grad():
        target_all = model.target(event[:, :cut + horizon], ctx[:, :cut + horizon])
        future = target_all["branches"][:, cut:cut + horizon]
        before = target_all["branches"][:, cut - 1]
        target = torch.cat((future.mean(1), future[:, -1], future[:, -1] - before), dim=-1).flatten(1)
    h = model.horizon(torch.full((len(event),), h_index, device=device))
    prediction = model.predictor(torch.cat((online, h), dim=-1))
    regression = F.smooth_l1_loss(F.layer_norm(prediction, prediction.shape[1:]),
                                  F.layer_norm(target, target.shape[1:]))
    variance, covariance = _variance_covariance(model.online(event[:, :cut], ctx[:, :cut])["public"])
    loss = regression + .10 * variance + .01 * covariance
    momentum = .99 + (.999 - .99) * min(1., step / max(1, total_steps))
    return loss, {"regression": regression, "variance": variance, "covariance": covariance, "momentum": momentum}


def pretrain(pack: dict, config: dict, method: str, steps: int, seed: int, device: torch.device,
             validation_pack: dict | None = None) -> tuple[RepresentationEncoderV11, list[dict], dict]:
    """Train one encoder. Validation is SSL-only and contains inner training vehicles."""
    set_seed(seed); rng = np.random.default_rng(seed)
    width = int(config["width"])
    encoder = RepresentationEncoderV11(pack["context"].shape[-1], width, float(config["dropout"])).to(device)
    if method == "scratch":
        return encoder, [], _diagnostics(encoder, pack["events"][:, :53], pack["context"][:, :53], device)
    if method == "masked":
        head, module = MaskedHeadV11(2 * width).to(device), encoder
        parameters = list(encoder.parameters()) + list(head.parameters())
    elif method == "contrastive":
        head, module = ContrastiveHeadV11(2 * width).to(device), encoder
        parameters = list(encoder.parameters()) + list(head.parameters())
    elif method == "jepa":
        model = JEPAV11(encoder, width).to(device); module = model
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        raise ValueError(f"unsupported V11 method: {method}")
    optimizer = torch.optim.AdamW(parameters, lr=float(config["ssl_learning_rate"]), weight_decay=float(config["ssl_weight_decay"]))
    batch_size = int(config["ssl_batch_size"]); history = []
    best_loss, best_step, best_state = float("inf"), steps, None
    for step in range(1, steps + 1):
        module.train(); events, context = _batch(pack, rng, batch_size)
        if method == "masked":
            loss, parts = _masked_step(encoder, head, events, context, rng, device)
        elif method == "contrastive":
            loss, parts = _contrast_step(encoder, head, events, context, rng, device, float(config["temperature"]))
        else:
            loss, parts = _jepa_step(model, events, context, rng, device, tuple(config["jepa_horizons"]), step, steps)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).detach().cpu()); optimizer.step()
        if method == "jepa":
            model.update_target(float(parts["momentum"]))
        if step == 1 or step % int(config["ssl_check_every"]) == 0 or step == steps:
            entry = {"step": step, "train_ssl_loss": float(loss.detach().cpu()), "gradient_norm": grad}
            entry.update({k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in parts.items()})
            if validation_pack is not None:
                module.eval()
                # Reuse an identical validation view at every checkpoint so
                # step selection reflects learning instead of mask difficulty.
                validation_rng = np.random.default_rng(seed + 100000)
                ve, vc = _batch(validation_pack, validation_rng, min(batch_size, max(8, len(validation_pack["events"]))))
                with torch.no_grad():
                    if method == "masked":
                        validation_loss, _ = _masked_step(encoder, head, ve, vc, validation_rng, device)
                    elif method == "contrastive":
                        validation_loss, _ = _contrast_step(encoder, head, ve, vc, validation_rng, device,
                                                             float(config["temperature"]))
                    else:
                        validation_loss, _ = _jepa_step(model, ve, vc, validation_rng, device,
                                                        tuple(config["jepa_horizons"]), step, steps)
                value = float(validation_loss.cpu()); entry["inner_val_ssl_loss"] = value
                if step >= int(config["ssl_min_steps"]) and value < best_loss:
                    best_loss, best_step = value, step
                    best_state = copy.deepcopy(encoder.state_dict())
            history.append(entry)
    if best_state is not None:
        encoder.load_state_dict(best_state)
    diagnostics = _diagnostics(encoder, pack["events"][:, :53], pack["context"][:, :53], device)
    diagnostics.update({"method": method, "steps": steps, "selected_steps": best_step,
                        "best_inner_val_ssl_loss": None if best_state is None else best_loss})
    return encoder, history, diagnostics
