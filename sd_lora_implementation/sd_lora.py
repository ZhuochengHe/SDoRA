from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F

from lora_implementation import LoRA_Linear
from sora_implementation import SoRA_Linear

class SDoRA_Linear(SoRA_Linear):
    def __init__(
            self, 
            base_linear,
            r=4, 
            lora_alpha=1.0,
            bias=True,
            lora_dropout=0.0,
            merged : bool = False,
            **kwargs,
            ):
        super().__init__(
            base_linear=base_linear,
            r=r,
            lora_alpha=lora_alpha,
            bias=(base_linear.bias is not None),
            lora_dropout=lora_dropout,
        )
        self.weight_m_wdecomp = nn.Linear(1,self.out_features,bias=False)
        if r > 0:
            with torch.no_grad():
                mag = torch.linalg.norm(self.linear.weight.detach(), dim=1, keepdim=True)
                self.weight_m_wdecomp.weight.copy_(mag)

    def forward(self, x):
        if self.r > 0 and not self.merged:
            device = self.linear.weight.device
            dtype = self.linear.weight.dtype

            new_weight_v = self.linear.weight + (self.lora_B.weight.mul(self.gate) @ self.lora_A.weight)* self.scaling
            # v_norm = torch.linalg.norm(new_weight_v,dim=1) + 1e-6
            eps = 1e-6
            v_norm = torch.linalg.norm(new_weight_v, dim=1).detach().clamp(min=eps)

            norm_scale = self.weight_m_wdecomp.weight.view(-1).to(device=device, dtype=dtype) / v_norm.detach()
            org_result = (F.linear(x, self.linear.weight)).to(dtype)
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale-1) * (F.linear(dropout_x, self.linear.weight.to(dtype)))
            if not self.linear.bias is None:
                result += self.linear.bias.view(1, -1).expand_as(result)

            delta = (dropout_x @ self.lora_A.weight.T).mul(self.gate) @ self.lora_B.weight.T
            result += ( norm_scale * delta ) * self.scaling
        else:
            result = F.linear(x, self.linear.weight.to(dtype), bias=self.linear.bias)
        
        return result
    
    @torch.no_grad
    def merge(self):
        if self.r > 0 and not self.merged:
            device = self.linear.weight.device
            dtype = self.linear.weight.dtype

            if self.gate is not None:
                new_weight_v = (self.linear.weight + (self.lora_B.weight.mul(self.gate) @ self.lora_A.weight) * self.scaling).to(device=device, dtype=dtype)
            else:
                new_weight_v = (self.linear.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling).to(device=device, dtype=dtype)
            self._v_norm = torch.linalg.norm(new_weight_v,dim=1).clone().detach()
            norm_scale = self.weight_m_wdecomp.weight.view(-1).to(device=device, dtype=dtype) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            self.linear.weight.data = new_weight_v * norm_scale.view(-1,1).to(dtype)
            self.merged = True
    
    @torch.no_grad
    def unmerge(self):
        if self.r > 0 and self.merged:
            if hasattr(self, "_v_norm"):
                device = self.linear.weight.device
                dtype = self.linear.weight.dtype
                w_m = self.weight_m_wdecomp.weight.view(-1, 1).to(device=device, dtype=dtype)
                v_norm = self._v_norm.view(-1, 1).to(device=device, dtype=dtype)

                new_weight_v = v_norm * self.linear.weight.data / (w_m + 1e-6)
                if self.gate is not None:
                    self.linear.weight.data = new_weight_v - (self.lora_B.weight.mul(self.gate) @ self.lora_A.weight) * self.scaling
                else:
                    self.linear.weight.data = new_weight_v - (self.lora_B.weight @ self.lora_A.weight) * self.scaling
                del self._v_norm
                self.merged = False

def wrap_linears_SDoRA(
    module: nn.Module,
    target_names: Sequence[str],
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
) -> None:
    """Replace the listed nn.Linear modules with SDoRA_Linear in-place."""

    name_to_module = dict(module.named_modules())
    for name in target_names:
        parent_name, _, child_name = name.rpartition(".")
        parent = name_to_module.get(parent_name, module)
        child = getattr(parent, child_name, None)
        if not isinstance(child, nn.Linear):
            raise ValueError(f"Target '{name}' is not nn.Linear")
        setattr(parent, child_name, SDoRA_Linear(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))


class SDoRAOptimizer(AdamW):

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


def split_sdora_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
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


def build_sdora_optimizers(
    model: nn.Module,
    base_lr: float,
    sparse_lambda: float,
    gate_lr: float | None = None,
    **adamw_kwargs,
) -> Tuple[SDoRAOptimizer, AdamW]:
    """Create the paired optimizers for gate and regular parameters."""

    gate_params, regular_params = split_sdora_params(model)
    if gate_lr is None:
        gate_lr = base_lr
    
    # Ensure weight decay is 0 for gates to solve pure L1 problem (Lasso), not Elastic Net
    gate_kwargs = adamw_kwargs.copy()
    gate_kwargs["weight_decay"] = 0.0
    gate_opt = SDoRAOptimizer(gate_params, lr=gate_lr, sparse_lambda=sparse_lambda, **gate_kwargs)
    
    regular_opt = AdamW(regular_params, lr=base_lr, **adamw_kwargs)
    return gate_opt, regular_opt


def prune_sdora_model(model: nn.Module) -> None:
    """Convert all SDoRA_Linear modules in the model to pruned LoRA form."""
    for module in model.modules():
        if isinstance(module, SDoRA_Linear):
            module.prune()

def merge_sdora_model(model: nn.Module) -> None:
    """Merge all SDoRA_Linear modules in the model into their base weights."""
    for module in model.modules():
        if isinstance(module, SDoRA_Linear):
            module.merge()
