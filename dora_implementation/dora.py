import torch.nn as nn
import torch
from lora_implementation import LoRA_Linear
from torch.nn import functional as F
    
class Dora_Linear(LoRA_Linear):
    def __init__(
            self, 
            r=4, 
            lora_alpha=1.0,
            bias=True,
            lora_dropout=0.0,
            base_linear: nn.Linear = None,
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
        if base_linear is not None:
            self.linear.weight = base_linear.weight
            if bias and base_linear.bias is not None:
                self.linear.bias = base_linear.bias

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

            result += ( norm_scale * (self.lora_B(self.lora_A(dropout_x.to(self.lora_A.weight.dtype))))) * self.scaling
        else:
             result = F.linear(x, self.linear.weight, bias=self.linear.bias)
        
        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result
    
    def merge(self):
        if self.r > 0 and not self.merged:
            new_weight_v = self.linear.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            self.linear.weight.data = self.linear.weight.data * norm_scale.view(-1,1)
            self.linear.weight.data += (self.lora_B.weight @ self.lora_A.weight) * self.scaling * norm_scale.view(-1,1)
            self.merged = True

    def unmerge(self):
        if self.r > 0 and self.merged:
            new_weight_v = self.linear.weight - (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            self.linear.weight.data = (self.linear.weight.data - (self.lora_B.weight @ self.lora_A.weight) * self.scaling * norm_scale.view(-1,1)) / norm_scale.view(-1,1)
            self.merged = False


    


    
