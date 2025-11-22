import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRA_Linear(nn.Module):
    def __init__(self, base_linear, r=0, lora_alpha=1.0, lora_dropout=0.0, **kwargs):
        super().__init__()

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.linear = base_linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.merged = False

        if r > 0:
            self.lora_A = nn.Linear(self.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, self.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

            self.lora_dropout = nn.Dropout(lora_dropout) \
                if lora_dropout > 0 else (lambda x: x)

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return (
                self.linear(x)
                + self.lora_dropout(x) @ (self.lora_B.weight @ self.lora_A.weight).T * self.scaling
            )
        else:
            return self.linear(x)

    @torch.no_grad
    def merge(self):
        if self.r > 0 and not self.merged:
            self.linear.weight.data += (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.merged = True

    @torch.no_grad
    def unmerge(self):
        if self.r > 0 and self.merged:
            self.linear.weight.data -= (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.merged = False
