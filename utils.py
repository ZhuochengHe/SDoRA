import torch
import torch.nn as nn
import json
from typing import Optional
from lora_implementation import LoRA_Linear
from dora_implementation import DoRA_Linear
from sora_implementation import SoRA_Linear, SoRAOptimizer
from sd_lora_implementation import SDoRA_Linear
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

def replace_linear_with_lora(model, target_modules, adapter_name, r=8, lora_alpha=16.0, lora_dropout=0.05, init_as_merged: bool = False):
    """
	Iterate through the model and change all the target modules into lora
    """
    adapter_classes = {
        "lora": LoRA_Linear,
        "sora": SoRA_Linear,
        "dora": DoRA_Linear,
        "sdora": SDoRA_Linear
    }
    adapter_cls = adapter_classes.get(adapter_name.lower())
    if adapter_cls is None:
        raise ValueError(f"Unknown adapter: {adapter_name}")
    
    replaced_count = 0
    for name, module in model.named_modules():
        is_target = any(name.endswith(target) for target in target_modules)
        
        if is_target and isinstance(module, nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model

            if init_as_merged and adapter_name.lower() in ["sora", "sdora"]:
                eff_r = 0
            else:
                eff_r = r
            
            adapter_module = adapter_cls(
                base_linear=module,
                r=eff_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
            
            setattr(parent, child_name, adapter_module)
            replaced_count += 1
            
    print(f"Replaced {replaced_count} modules with {adapter_name.upper()}")
    return model

def get_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params, all_param

def get_optimizer(model, adapter_name, lr=1e-4, sparse_lambda=0.3, weight_decay=0.01, gate_lr_multiplier=10, gate_lr: Optional[float] = None):
    adapter_name = adapter_name.lower()
    
    if adapter_name in ["sora", "sdora"]:        
        gate_params = []
        regular_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if "gate" in name:
                gate_params.append(param)
            else:
                regular_params.append(param)
        
        print(f"Found {len(gate_params)} gate parameters, {len(regular_params)} regular parameters")
        
        # if gate_lr explicit is provided, use that (absolute), otherwise use lr * gate_lr_multiplier
        if gate_lr is None:
            gate_lr_val = lr * gate_lr_multiplier
        else:
            gate_lr_val = gate_lr
        gate_optimizer = SoRAOptimizer(gate_params, lr=gate_lr_val, sparse_lambda=sparse_lambda, weight_decay=0.0)
        base_optimizer = AdamW(regular_params, lr=lr, weight_decay=weight_decay, fused=True)
        
        return base_optimizer, gate_optimizer
    
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay
        )
        
        return optimizer, None


def merge_and_save(model, adapter_name, save_path):
    adapter_classes = {
        "lora": LoRA_Linear,
        "sora": SoRA_Linear,
        "dora": DoRA_Linear,
        "sdora": SDoRA_Linear
    }
    
    adapter_cls = adapter_classes.get(adapter_name.lower())
    
    r_details = {}
    # 1. prune（SoRA/SDoRA）
    if adapter_name.lower() in ["sora", "sdora"]:
        print("Pruning sparse gates...")
        pruned_count = 0
        total_ranks_before = 0
        total_ranks_after = 0
        
        for name, module in model.named_modules():
            if isinstance(module, adapter_cls) and hasattr(module, 'prune'):
                if hasattr(module, 'r') and hasattr(module, 'gate') and module.gate is not None:
                    total_ranks_before += module.r
                    module.prune()
                    total_ranks_after += module.r
                    pruned_count += 1
                    r_details[name] = module.r
        
        if total_ranks_before > 0:
            compression = 100 * (1 - total_ranks_after / total_ranks_before)
            print(f"Pruned {pruned_count} modules: {total_ranks_before} -> {total_ranks_after} ranks ({compression:.1f}% reduction)")

    if adapter_name.lower() in ["lora", "dora"]:
        for name, module in model.named_modules():
            if isinstance(module, adapter_cls) and hasattr(module, 'r'):
                r_details[name] = module.r
    r_json_path = save_path + "_r_details.json"
    with open(r_json_path, "w") as f:
        json.dump(r_details, f, indent=2)
    print(f"Saved r details to {r_json_path}")
    
    # 2. Merge to base weights
    print(f"Merging {adapter_name.upper()} weights...")
    merged_count = 0
    for module in model.modules():
        if isinstance(module, adapter_cls) and hasattr(module, 'merge'):
            module.merge()
            merged_count += 1
    
    print(f"Merged {merged_count} modules")
    
    # 3. Save
    torch.save(model.state_dict(), save_path)
    print(f"Saved to {save_path}")

def log_gate_stats(model):
    gate_vals = []
    for name, module in model.named_modules():
        if hasattr(module, 'gate') and module.gate is not None:
            g = module.gate.detach().float().flatten()
            gate_vals.append(g)
    if not gate_vals:
        print("No gates found.")
        return
    allg = torch.cat(gate_vals)
    print(f"Gate stats: mean={allg.mean():.6f}, std={allg.std():.6f}, min={allg.min():.6f}, max={allg.max():.6f}, exact_zero={int((allg==0).sum().item())}/{allg.numel()}")


def get_sora_delta_stats(model):
    """Return a quick approximation of per-SoRA-layer delta vs base weight scales.
    The estimate is max(|A|) * max(|B|) * max(|gate|) and a ratio vs base weight max as a quick diagnostic.
    """
    res = []
    for name, module in model.named_modules():
        if isinstance(module, SoRA_Linear):
            a_max = module.A.data.abs().max().item() if hasattr(module, 'A') and module.A is not None else 0.0
            b_max = module.B.data.abs().max().item() if hasattr(module, 'B') and module.B is not None else 0.0
            gate_max = module.gate.data.abs().max().item() if hasattr(module, 'gate') and module.gate is not None else 0.0
            base_w_max = module.weight.data.abs().max().item() if hasattr(module, 'weight') and module.weight is not None else 0.0
            est_delta_max = a_max * b_max * gate_max
            ratio = est_delta_max / (base_w_max + 1e-12)
            res.append({
                'name': name,
                'a_max': a_max,
                'b_max': b_max,
                'gate_max': gate_max,
                'base_w_max': base_w_max,
                'est_delta_max': est_delta_max,
                'ratio': ratio
            })
    return res