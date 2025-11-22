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

    


    
