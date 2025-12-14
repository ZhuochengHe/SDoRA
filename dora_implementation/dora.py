import torch.nn as nn
import torch
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
        device = base_linear.weight.device
        dtype = base_linear.weight.dtype

        self.weight_m_wdecomp = nn.Linear(1,self.out_features,bias=False)
        if r > 0:
            with torch.no_grad():
                mag = torch.linalg.norm(self.linear.weight.detach(), dim=1, keepdim=True).to(device=device, dtype=dtype)
                self.weight_m_wdecomp.weight.copy_(mag)

    def forward(self, x):
        previous_dtype = x.dtype
        device = self.linear.weight.device
        dtype = self.linear.weight.dtype

        if self.r > 0 and not self.merged:
            new_weight_v = (
                self.linear.weight
                + (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            ).to(device=device, dtype=dtype)

            # v_norm = torch.linalg.norm(new_weight_v,dim=1) + 1e-6
            eps = 1e-6
            v_norm = torch.linalg.norm(new_weight_v, dim=1).detach().clamp(min=eps)
            norm_scale = self.weight_m_wdecomp.weight.view(-1).to(device=device, dtype=dtype) / v_norm.detach()

            org_result = (F.linear(x, self.linear.weight.to(dtype)))
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale-1) * (F.linear(dropout_x, self.linear.weight.to(dtype)))
            if not self.linear.bias is None:
                result += self.linear.bias.view(1, -1).expand_as(result)

            result += ( norm_scale * (self.lora_B(self.lora_A(dropout_x)))) * self.scaling
        else:
            result = F.linear(x, self.linear.weight.to(dtype), bias=self.linear.bias)
        
        return result
        
    @torch.no_grad()
    def merge(self):
        if self.r > 0 and not self.merged:
            device = self.linear.weight.device
            dtype = self.linear.weight.dtype

            new_weight_v = (self.linear.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling).to(device=device, dtype=dtype)
            self._v_norm = torch.linalg.norm(new_weight_v,dim=1).clone().detach()
            norm_scale = self.weight_m_wdecomp.weight.view(-1).to(device=device, dtype=dtype) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            self.linear.weight.data = new_weight_v * norm_scale.view(-1,1).to(dtype)
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.r > 0 and self.merged:
            if hasattr(self, "_v_norm"):
                device = self.linear.weight.device
                dtype = self.linear.weight.dtype

                w_m = self.weight_m_wdecomp.weight.view(-1, 1).to(device=device, dtype=dtype)
                v_norm = self._v_norm.view(-1, 1).to(device=device, dtype=dtype)

                new_weight_v = v_norm * self.linear.weight.data / (w_m + 1e-6)
                delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
                self.linear.weight.data = (new_weight_v - delta).to(dtype)

                del self._v_norm
                self.merged = False
