import torch
import torch.nn as nn
from torch.nn import functional as F
from lora_implementation import LoRA_Linear
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

    


    
class SoRA_Linear(nn.Module):
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
        
        if self.gate is not None:
            delta = (self.dropout(x) @ self.lora_A.T).mul(self.gate) @ self.lora_B.T
        else:
            # Pruned mode (standard LoRA)
            delta = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
            
        return result + delta * self.scaling

    @torch.no_grad()
    def prune(self) -> None:
        """Prune zeroed-out ranks and merge gate into weights."""
        if self.gate is None:
            return

        # Identify non-zero indices
        gate_data = self.gate.data.flatten()
        non_zero_indices = torch.nonzero(gate_data).flatten()
        new_rank = non_zero_indices.numel()

        if new_rank == 0:
            self.rank = 0
            self.lora_A = None
            self.lora_B = None
            self.gate = None
            return

        # Select rows/cols
        # lora_A: (r, in) -> (new_r, in)
        new_A = self.lora_A.data[non_zero_indices].clone()
        
        # lora_B: (out, r) -> (out, new_r)
        # Merge gate: B_new = B_old * gate
        # B: (out, r), gate: (1, r)
        new_B = self.lora_B.data[:, non_zero_indices].clone()
        gate_values = gate_data[non_zero_indices]
        new_B = new_B * gate_values.view(1, -1)

        # Update parameters
        self.rank = new_rank
        self.lora_A = nn.Parameter(new_A)
        self.lora_B = nn.Parameter(new_B)
        self.gate = None # Remove gate

    @torch.no_grad()
    def merge(self) -> None:
        """Merge the LoRA branch into the base layer weights for maximum inference speed."""
        if self.rank == 0 or self.lora_A is None:
            return

        # Calculate delta: B @ A * scaling
        # If gate exists, include it: B @ diag(g) @ A
        
        if self.gate is not None:
            # B: (out, r), gate: (1, r) -> B_scaled: (out, r)
            B_scaled = self.lora_B.data * self.gate.data.view(1, -1)
        else:
            B_scaled = self.lora_B.data
            
        # A: (r, in)
        # delta = B_scaled @ A
        weight_delta = (B_scaled @ self.lora_A.data) * self.scaling
        
        # Add to base weight
        self.base.weight.data += weight_delta.to(self.base.weight.device)
        
        # Remove LoRA params completely
        self.rank = 0
        self.lora_A = None
        self.lora_B = None
        self.gate = None