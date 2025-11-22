import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRA_Linear(nn.Module):
    def __init__(self, base_linear, r=0, lora_alpha=1, lora_dropout=0.0, **kwargs):
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
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            nn.init.normal_(self.lora_A)
            nn.init.zeros_(self.lora_B)

            self.lora_dropout = nn.Dropout(lora_dropout) \
                if lora_dropout > 0 else (lambda x: x)

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return (
                self.linear(x)
                + self.lora_dropout(x) @ (self.lora_B @ self.lora_A).T * self.scaling
            )
        else:
            return self.linear(x)

    def merge(self):
        if self.r > 0 and not self.merged:
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerge(self):
        if self.r > 0 and self.merged:
            self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
