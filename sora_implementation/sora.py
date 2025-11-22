"""Standalone SoRA implementation for reuse in custom projects."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

from lora_implementation import LoRA_Linear

class SoRA_Linear(LoRA_Linear):
    """Adds a sparse low-rank branch next to a frozen linear layer with gating mechanism."""

    def __init__(
        self,
        base_linear,
        r=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        merged: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            base_linear=base_linear,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            **kwargs
        )
        
        # Add gate parameter (SoRA-specific)
        if r > 0:
            self.gate = nn.Parameter(torch.ones(1, r))
        else:
            self.register_parameter("gate", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        if self.r == 0 or self.merged:
            return result
        
        # Apply dropout
        x_dropout = self.lora_dropout(x)
        
        if self.gate is not None:
            # SoRA: gated low-rank adaptation
            # (x @ A.T * gate) @ B.T
            delta = (x_dropout @ self.lora_A.weight.T).mul(self.gate) @ self.lora_B.weight.T
        else:
            # Pruned mode (standard LoRA)
            delta = (x_dropout @ self.lora_A.weight.T) @ self.lora_B.weight.T
            
        return result + delta * self.scaling

    @torch.no_grad
    def prune(self) -> None:
        """Prune zeroed-out ranks and merge gate into weights."""
        if self.gate is None or self.r == 0:
            return

        # Identify non-zero indices
        gate_data = self.gate.data.flatten()
        non_zero_indices = torch.nonzero(gate_data).flatten()
        new_rank = non_zero_indices.numel()

        if new_rank == 0:
            self.r = 0
            self.lora_A = None
            self.lora_B = None
            self.gate = None
            return

        # Select rows/cols from lora_A.weight and lora_B.weight
        # lora_A.weight: (r, in) -> (new_r, in)
        new_A_weight = self.lora_A.weight.data[non_zero_indices].clone()
        
        # lora_B.weight: (out, r) -> (out, new_r)
        # Merge gate: B_new = B_old * gate
        new_B_weight = self.lora_B.weight.data[:, non_zero_indices].clone()
        gate_values = gate_data[non_zero_indices]
        new_B_weight = new_B_weight * gate_values.view(1, -1)

        # Update parameters (create new nn.Linear modules with correct shapes)
        self.r = new_rank
        self.lora_A = nn.Linear(self.in_features, new_rank, bias=False)
        self.lora_B = nn.Linear(new_rank, self.out_features, bias=False)
        self.lora_A.weight.data = new_A_weight  # Shape already (new_r, in)
        self.lora_B.weight.data = new_B_weight  # Shape already (out, new_r)
        self.gate = None # Remove gate

    @torch.no_grad
    def merge(self) -> None:
        """Merge the LoRA branch into the base layer weights for maximum inference speed."""
        if self.r == 0 or self.lora_A is None or self.merged:
            return

        # Calculate delta: B @ A * scaling
        # If gate exists, include it: B @ diag(g) @ A
        
        if self.gate is not None:
            # B.weight: (out, r), gate: (1, r) -> B_scaled: (out, r)
            B_scaled = self.lora_B.weight.data * self.gate.data.view(1, -1)
        else:
            B_scaled = self.lora_B.weight.data
            
        # A.weight: (r, in)
        # delta = B_scaled @ A.weight
        weight_delta = (B_scaled @ self.lora_A.weight.data) * self.scaling
        
        # Add to base weight
        self.linear.weight.data += weight_delta.to(self.linear.weight.device)
        self.merged = True

    @torch.no_grad
    def unmerge(self) -> None:
        """Unmerge the LoRA branch from the base layer weights."""
        if self.r == 0 or self.lora_A is None or not self.merged:
            return

        # Calculate delta: B @ A * scaling
        if self.gate is not None:
            B_scaled = self.lora_B.weight.data * self.gate.data.view(1, -1)
        else:
            B_scaled = self.lora_B.weight.data
            
        weight_delta = (B_scaled @ self.lora_A.weight.data) * self.scaling
        self.linear.weight.data -= weight_delta.to(self.linear.weight.device)
        self.merged = False 



def wrap_linears(
    module: nn.Module,
    target_names: Sequence[str],
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
) -> None:
    """Replace the listed nn.Linear modules with SoRA_Linear in-place."""

    name_to_module = dict(module.named_modules())
    for name in target_names:
        parent_name, _, child_name = name.rpartition(".")
        parent = name_to_module.get(parent_name, module)
        child = getattr(parent, child_name, None)
        if not isinstance(child, nn.Linear):
            raise ValueError(f"Target '{name}' is not nn.Linear")
        setattr(parent, child_name, SoRA_Linear(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))


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
    
    # Ensure weight decay is 0 for gates to solve pure L1 problem (Lasso), not Elastic Net
    gate_kwargs = adamw_kwargs.copy()
    gate_kwargs["weight_decay"] = 0.0
    gate_opt = SoRAOptimizer(gate_params, lr=gate_lr, sparse_lambda=sparse_lambda, **gate_kwargs)
    
    regular_opt = AdamW(regular_params, lr=base_lr, **adamw_kwargs)
    return gate_opt, regular_opt


def prune_sora_model(model: nn.Module) -> None:
    """Convert all SoRA_Linear modules in the model to pruned LoRA form."""
    for module in model.modules():
        if isinstance(module, SoRA_Linear):
            module.prune()

def merge_sora_model(model: nn.Module) -> None:
    """Merge all SoRA_Linear modules in the model into their base weights."""
    for module in model.modules():
        if isinstance(module, SoRA_Linear):
            module.merge()
