import torch
import torch.nn as nn
from torch.nn import functional as F
from lora_implementation import LoRA_Linear
import math
class DoRA_Linear(LoRA_Linear):
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

    def forward(self, x):
        previous_dtype = x.dtype
        if self.r > 0 and not self.merged:
            new_weight_v = self.linear.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            org_result = (F.linear(x, self.linear.weight))
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale-1) * (F.linear(dropout_x, self.linear.weight))
            if not self.linear.bias is None:
                result += self.linear.bias.view(1, -1).expand_as(result)

            result += ( norm_scale * (self.lora_B(self.lora_A(dropout_x)))) * self.scaling
        else:
            result = F.linear(x, self.linear.weight, bias=self.linear.bias)
        
        return result
        
    @torch.no_grad
    def merge(self):
        if self.r > 0 and not self.merged:
            self._v_norm = torch.linalg.norm(self.linear.weight,dim=1).clone().detach()
            new_weight_v = self.linear.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            self.linear.weight.data = new_weight_v * norm_scale.view(-1,1)
            self.merged = True

    @torch.no_grad
    def unmerge(self):
        if self.r > 0 and self.merged:
            if hasattr(self, "_v_norm"):
                new_weight_v = self._v_norm.view(-1,1) * self.linear.weight.data / self.weight_m_wdecomp.weight.view(-1,1)
                self.linear.weight.data = new_weight_v - (self.lora_B.weight @ self.lora_A.weight) * self.scaling
                del self._v_norm
                self.merged = False

    


    
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


class SDoRA_Linear(LoRA_Linear):
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
        