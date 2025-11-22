"""Standalone SoRA implementation for reuse in custom projects."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW


class SoRALinear(nn.Module):
    """Adds a sparse low-rank branch next to a frozen linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be nn.Linear")
        self.base = base_layer
        for param in self.base.parameters():
            param.requires_grad = False

        self.rank = r
        self.scaling = alpha / max(r, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else (lambda x: x)

        if r > 0:
            self.lora_A = nn.Parameter(base_layer.weight.new_zeros((r, base_layer.in_features)))
            self.lora_B = nn.Parameter(base_layer.weight.new_zeros((base_layer.out_features, r)))
            self.gate = nn.Parameter(torch.ones(1, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.register_parameter("gate", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if self.rank == 0:
            return result
        delta = (self.dropout(x) @ self.lora_A.T).mul(self.gate) @ self.lora_B.T
        return result + delta * self.scaling


def wrap_linears(
    module: nn.Module,
    target_names: Sequence[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> None:
    """Replace the listed nn.Linear modules with SoRALinear in-place."""

    name_to_module = dict(module.named_modules())
    for name in target_names:
        parent_name, _, child_name = name.rpartition(".")
        parent = name_to_module.get(parent_name, module)
        child = getattr(parent, child_name, None)
        if not isinstance(child, nn.Linear):
            raise ValueError(f"Target '{name}' is not nn.Linear")
        setattr(parent, child_name, SoRALinear(child, r=r, alpha=alpha, dropout=dropout))


class SoRAOptimizer(AdamW):
    """AdamW variant that applies proximal sparsification to gate parameters."""

    def __init__(self, params: Iterable[torch.Tensor], sparse_lambda: float = 0.1, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.sparse_lambda = sparse_lambda

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = super().step(closure)
        for group in self.param_groups:
            threshold = self.sparse_lambda * group["lr"]
            if threshold <= 0:
                continue
            for param in group["params"]:
                if param.grad is None:
                    continue
                positive_mask = param.data > threshold
                negative_mask = param.data < -threshold
                middle_mask = (~positive_mask) & (~negative_mask)
                param.data[positive_mask] -= threshold
                param.data[negative_mask] += threshold
                param.data[middle_mask] = 0.0
        return loss


def split_sora_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Return gate parameters and all remaining trainable parameters."""

    gate_params: List[nn.Parameter] = []
    regular_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("gate"):
            gate_params.append(param)
        else:
            regular_params.append(param)
    return gate_params, regular_params


def build_sora_optimizers(
    model: nn.Module,
    base_lr: float,
    sparse_lambda: float,
    gate_lr: float | None = None,
    **adamw_kwargs,
) -> Tuple[SoRAOptimizer, AdamW]:
    """Create the paired optimizers for gate and regular parameters."""

    gate_params, regular_params = split_sora_params(model)
    if gate_lr is None:
        gate_lr = base_lr
    gate_opt = SoRAOptimizer(gate_params, lr=gate_lr, sparse_lambda=sparse_lambda, **adamw_kwargs)
    regular_opt = AdamW(regular_params, lr=base_lr, **adamw_kwargs)
    return gate_opt, regular_opt
