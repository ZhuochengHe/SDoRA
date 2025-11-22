import torch
import torch.nn as nn
from lora_implementation import LoRA_Linear
from dora_implementation import DoRA_Linear
from sora_implementation import SoRA_Linear

def replace_linear_with_lora(model, target_modules, lora_type="lora", **kwargs):
    """
	Iterate through the model and change all the target modules into lora
    """
    lora_module_dict = {
        "lora": LoRA_Linear,
        "sora": SoRA_Linear,
        "dora": DoRA_Linear
    }
    lora_cls = lora_module_dict.get(lora_type.lower())
    if lora_cls is None:
        raise ValueError(f"Unknown lora_type {lora_type}, available: {list(lora_module_dict.keys())}")
    for name, module in model.named_modules():
        is_target = any(name.endswith(t) for t in target_modules)

        if is_target and isinstance(module, nn.Linear):
            parent = model.get_submodule(".".join(name.split(".")[:-1]))
            child_name = name.split(".")[-1]

            new_module = lora_cls(base_linear=module, **kwargs)

            new_module.linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_module.bias = module.bias
            setattr(parent, child_name, new_module)
