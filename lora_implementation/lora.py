import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRA_Linear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.linear = linear # (out_features, in_features)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.linear_weight = linear.weight.clone().detach()
        self.bias = linear.bias.clone().detach() if linear.bias is not None else None
        self.linear_weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        for p in self.linear.parameters():
            p.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.merged = False
        self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            self._initialize_params()


    def _initialize_params(self):
        nn.init.normal_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def merge(self):
        if self.r > 0 and not self.merged:
            self.linear_weight += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True
    
    def unmerge(self):
        if self.r > 0 and self.merged:
            self.linear_weight -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    def eval(self):
        super().eval()
        if self.r > 0:
            self.lora_A.eval()
            self.lora_B.eval()
        
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.linear_weight, bias=self.bias)
            result += self.lora_dropout(x) @ (self.lora_B @ self.lora_A).T * self.scaling
        else:
            return F.linear(x, self.linear_weight, bias=self.bias)