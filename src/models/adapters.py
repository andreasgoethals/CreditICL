"""Opt-in Experiment 3 adapters; released models and Experiment 1 are unchanged.

DiagScale follows the upstream TTC insertion points: final scaled queries in
TabICL attention and TabPFN-v3 SoftmaxScalingMLP. LoRA changes Q/V only, including
TabICL's packed QKV; K is never updated. Native weights and resumable adapter
state are stored separately, so evaluation cannot silently discard an adapter.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import parametrize


class QueryScale(nn.Module):
    def __init__(self, base, heads: int, dim: int, *, layout: str):
        super().__init__()
        self.base = base
        self.layout = layout
        self.delta = nn.Parameter(torch.zeros(heads, dim))

    def forward(self, q, n):
        q = self.base(q, n) if self.base is not None else q
        scale = (1 + self.delta).to(q.dtype)
        if self.layout == "BHSD":
            scale = scale[:, None, :]
        return q * scale


class QVLowRank(nn.Module):
    def __init__(self, rows: int, columns: int, rank: int, alpha: float, *, packed=False):
        super().__init__()
        self.packed = packed
        self.scale = alpha / rank
        outputs = rows // 3 if packed else rows
        self.a = nn.Parameter(torch.empty(2 if packed else 1, rank, columns))
        self.b = nn.Parameter(torch.zeros(2 if packed else 1, outputs, rank))
        for a in self.a:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))

    def forward(self, weight):
        # Form the effective FP32 weight identically during autocast training
        # and native checkpoint export; otherwise merging changes predictions.
        with torch.autocast(device_type=weight.device.type, enabled=False):
            delta = (self.b.float() @ self.a.float()) * self.scale
        if self.packed:
            delta = torch.cat((delta[0], torch.zeros_like(delta[0]), delta[1]), dim=0)
        else:
            delta = delta[0]
        return weight + delta.to(weight.dtype)


def inject(model: nn.Module, cfg: dict, *, seed: int = 0) -> dict:
    """Freeze the backbone and inject identity-initialised adapters exactly once."""
    method = cfg.get("adaptation", "full")
    if method == "full":
        return {"method": method}
    if method not in {"lora", "diagscale"}:
        raise ValueError(f"Unknown adaptation: {method}")
    if getattr(model, "_crediticl_adapter", None):
        raise ValueError("Adapters already installed")
    if float(cfg.get("l2sp_alpha", 0)):
        raise ValueError("L2-SP is a full-weight treatment, not combined with adapters")
    rank, alpha = int(cfg.get("lora_rank", 16)), float(cfg.get("lora_alpha", 16))
    if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    model.requires_grad_(False)
    sites = []
    # Use a private RNG scope: changing the adapter must not change synthetic tasks.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
        torch.manual_seed(seed)
        for name, module in list(model.named_modules()):
            if method == "lora":
                packed = (isinstance(module, nn.MultiheadAttention)
                          and module.in_proj_weight is not None)
                linear = isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in {"q_projection", "v_projection"}
                if not (packed or linear):
                    continue
                attr = "in_proj_weight" if packed else "weight"
                weight = getattr(module, attr)
                adapter = QVLowRank(*weight.shape, rank, alpha, packed=packed).to(weight.device)
                parametrize.register_parametrization(module, attr, adapter)
                sites.append(name + "." + attr)
            elif hasattr(module, "ssmax_layer") and isinstance(module, nn.MultiheadAttention):
                module.ssmax_layer = QueryScale(module.ssmax_layer, module.num_heads,
                                                module.head_dim, layout="BHSD").to(next(module.parameters()).device)
                sites.append(name + ".ssmax_layer")
            elif type(module).__name__ == "SoftmaxScalingMLP":
                parent_name, _, attr = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                wrapped = QueryScale(module, module.num_heads, module.head_dim, layout="BSHD")
                setattr(parent, attr, wrapped.to(next(module.parameters()).device))
                sites.append(name)
    if not sites:
        raise ValueError(f"{method} matched no supported native attention modules")
    model._crediticl_adapter = dict(method=method, sites=sites, rank=rank, alpha=alpha, seed=seed)
    return {**model._crediticl_adapter,
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in model.parameters())}


def checkpoint_fields(model: nn.Module) -> dict:
    """Native inference weights plus exact adapter state for resuming training."""
    meta = getattr(model, "_crediticl_adapter", None)
    if not meta:
        return {"state_dict": model.state_dict()}
    training = model.state_dict()
    native = dict(training)
    if meta["method"] == "lora":
        for site in meta["sites"]:
            parent, _, attr = site.rpartition(".")
            module = model.get_submodule(parent) if parent else model
            prefix = (parent + "." if parent else "") + "parametrizations." + attr + "."
            native = {k: v for k, v in native.items() if not k.startswith(prefix)}
            native[site] = getattr(module, attr).detach()
    else:
        for site in meta["sites"]:
            native.pop(site + ".delta")
            for key in list(native):
                if key.startswith(site + ".base."):
                    native[site + key[len(site + ".base"):]] = native.pop(key)
    return {"state_dict": native, "crediticl_adapter": meta,
            "crediticl_training_state": training}


def measured_parameters(model: nn.Module):
    """For LoRA, measure the effective weights rather than arbitrary factor norms.

    DiagScale keeps its explicit scale deltas in this parameter-space diagnostic;
    their norms are not a displacement of a linear weight matrix.
    """
    meta = getattr(model, "_crediticl_adapter", None)
    if not meta or meta["method"] != "lora":
        return model.named_parameters()
    parameters = dict(model.named_parameters())
    for site in meta["sites"]:
        parent, _, attr = site.rpartition(".")
        prefix = (parent + "." if parent else "") + "parametrizations." + attr + "."
        parameters = {n: p for n, p in parameters.items() if not n.startswith(prefix)}
        module = model.get_submodule(parent) if parent else model
        parameters[site] = getattr(module, attr)
    return parameters.items()


def restore_inference(model: nn.Module, payload: dict) -> None:
    """LoRA is already merged. DiagScale must remain an explicit query operator."""
    meta = payload.get("crediticl_adapter")
    if not meta or meta["method"] == "lora":
        return
    if meta["method"] != "diagscale":
        raise ValueError("Unknown checkpoint adapter")
    if getattr(model, "_crediticl_adapter", None):
        return  # native estimators can initialise their cached model more than once
    inject(model, {"adaptation": "diagscale"}, seed=meta["seed"])
    if model._crediticl_adapter["sites"] != meta["sites"]:
        raise ValueError("Adapter insertion sites changed")
    model.load_state_dict(payload["crediticl_training_state"], strict=True)
    model.eval()


def inference_class(base):
    """Preserve native preprocessing while installing adapters before any KV cache."""
    class AdapterEstimator(base):
        def _load_model(self):
            super()._load_model()
            restore_inference(self.model_, torch.load(self.model_path_, map_location="cpu", weights_only=True))

        def _initialize_model_variables(self):
            result = super()._initialize_model_variables()
            if isinstance(self.model_path, (str, Path)):
                payload = torch.load(self.model_path, map_location="cpu", weights_only=False)
                for model in self.models_:
                    restore_inference(model, payload)
            return result

    return AdapterEstimator
