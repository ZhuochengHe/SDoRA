import os
import json
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
# Ensure your utils.py contains the correct SoRAOptimizer implementation!
from utils import replace_linear_with_lora, get_optimizer, merge_and_save, log_gate_stats, get_sora_delta_stats
from typing import Optional
from tqdm import tqdm
import argparse
from accelerate import Accelerator 
from sora_implementation.sora import prune_sora_model

class MetricsTracker:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.history = {
            'train_loss': [], 'val_loss': [],
            'epoch_time': [], 'sparsity': []
        }
    
    def update(self, metrics):
        for k, v in metrics.items():
            if k in self.history:
                self.history[k].append(v)
        
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def print_summary(self, epoch, metrics):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch} Summary:")
        print(f"{'='*70}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:20s}: {v:.4f}")
        print(f"{'='*70}\n")


def get_sparsity_stats(model, threshold: float = 1e-3):
    """
    Calculates the percentage of gates that are effectively zero (below threshold).
    Using 1e-3 is safer than 1e-6 for Proximal methods to account for float precision.
    """
    total_gates = 0
    zero_gates = 0
    
    for module in model.modules():
        if hasattr(module, 'gate') and module.gate is not None:
            gate = module.gate
            total_gates += gate.numel()
            zero_gates += (gate.abs() < threshold).sum().item()
    
    return zero_gates / total_gates if total_gates > 0 else 0.0


def generate_prompt(data_point):
    if data_point.get("input"):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        ### Instruction:
        {data_point["instruction"]}
        ### Input:
        {data_point["input"]}
        ### Response:
        {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction:
            {data_point["instruction"]}
            ### Response:
            {data_point["output"]}"""


def generate_and_tokenize_prompt(batch, tokenizer, cutoff_len=256):
    prompts = [
        generate_prompt({
            "instruction": instr,
            "input": inp,
            "output": out
        })
        for instr, inp, out in zip(
            batch["instruction"],
            batch.get("input", [None]*len(batch["instruction"])),
            batch["output"],
        )
    ]
    tokens = tokenizer(
        prompts,
        truncation=True,
        max_length=cutoff_len,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def collate_fn(batch):
    for i, x in enumerate(batch):
        if "input_ids" not in x:
            print(f"[collate] Missing input_ids in sample {i}: keys={list(x.keys())}")
            raise ValueError("Missing input_ids")
            
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = [x["input_ids"] + [0]*(max_len-len(x["input_ids"])) for x in batch]
    attention_mask = [x["attention_mask"] + [0]*(max_len-len(x["attention_mask"])) for x in batch]
    labels = [x["labels"] + [-100]*(max_len-len(x["labels"])) for x in batch]
    
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels)
    }


def train(
    base_model: str,
    data_path: str,
    output_dir: str,
    adapter_name: str = "lora",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    batch_size: int = 32,
    micro_batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    sparse_lambda: float = 0.3,
    gate_lr_multiplier: int = 10,
    gate_lr: Optional[float] = None,
    debug_print_steps: int = 0,
    use_prox_only: bool = True,  # Defaulted to True for safety in this revised version
    sparsity_threshold: float = 1e-3,
    log_gate_stats_steps: int = 1000,
    cutoff_len: int = 256,
    val_set_size: int = 120,
):
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]

    # --- SAFETY CHECK ---
    if adapter_name.lower() in ['sora', 'sdora']:
        print(f"[{adapter_name.upper()} MODE DETECTED]")
        print(f" -> Sparsity Lambda: {sparse_lambda}")
        print(f" -> Gate LR Multiplier: {gate_lr_multiplier}")
        
        # Check if lambda is strong enough for >0.5 sparsity
        if sparse_lambda < 0.1:
            print(f"⚠️  WARNING: sparse_lambda ({sparse_lambda}) is very low. Sparsity may not reach 50%.")
        
        # Enforce Proximal Logic intention
        if not use_prox_only:
             print("⚠️  WARNING: use_prox_only is set to False. Standard L1 loss will be used (Less effective for true zero sparsity).")

    gradient_accumulation_steps = batch_size // micro_batch_size

    # 1. Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    device = accelerator.device
    
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"Training {adapter_name.upper()}")
        print(f"{'='*70}")
        print(f"Model: {base_model}")
        print(f"Rank: {lora_r}, Alpha: {lora_alpha}, Epochs: {num_epochs}")
        print(f"Output: {output_dir}")
        print(f"{'='*70}\n")
        
    accelerator.wait_for_everyone()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Apply adapter
    model = replace_linear_with_lora(
        model, target_modules, adapter_name, 
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
    )

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)\n")

    # Load dataset
    if data_path.endswith(".json"):
        data = load_dataset("json", data_files=data_path)["train"]
    else:
        data = load_dataset(data_path)["train"]

    # Split train/val
    data = data.train_test_split(test_size=val_set_size, seed=42)
    train_data = data["train"]
    val_data = data["test"]

    # Tokenize
    train_data = train_data.map(
        lambda batch: generate_and_tokenize_prompt(batch, tokenizer, cutoff_len),
        remove_columns=train_data.column_names, batched=True, num_proc=4
    )
    val_data = val_data.map(
        lambda batch: generate_and_tokenize_prompt(batch, tokenizer, cutoff_len),
        remove_columns=val_data.column_names, batched=True, num_proc=2
    )

    # Dataloader
    train_loader = DataLoader(
        train_data, batch_size=micro_batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_data, batch_size=micro_batch_size, collate_fn=collate_fn, 
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    # Optimizer
    # NOTE: Ensure get_optimizer uses SoRAOptimizer when adapter is sora/sdora!
    base_opt, gate_opt = get_optimizer(
        model,
        adapter_name,
        lr=learning_rate,
        sparse_lambda=sparse_lambda,
        gate_lr_multiplier=gate_lr_multiplier,
        gate_lr=gate_lr,
    )

    # Scheduler
    num_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(base_opt, num_warmup_steps=20, num_training_steps=num_steps)
    if gate_opt:
        gate_scheduler = get_linear_schedule_with_warmup(gate_opt, num_warmup_steps=20, num_training_steps=num_steps)

    # Prepare with Accelerator
    model, base_opt, train_loader, val_loader, scheduler = accelerator.prepare(
        model, base_opt, train_loader, val_loader, scheduler
    )
    if gate_opt:
        gate_opt, gate_scheduler = accelerator.prepare(gate_opt, gate_scheduler)

    tracker = MetricsTracker(output_dir)
    
    if accelerator.is_main_process:
        print(f"{'='*70}")
        print("Starting Training")
        print(f"{'='*70}\n")
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                
                # REVISION: REMOVED EXPLICIT L1 LOSS ADDITION HERE.
                # We rely ENTIRELY on the Proximal Operator in gate_opt.step() to handle sparsity.
                # This ensures we get "True Zeros" and avoid double penalization.
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                    base_opt.step()
                    
                    if gate_opt:
                        # CRITICAL: This .step() contains the Proximal Operator (Soft Thresholding)
                        # It will look at the parameters and force small ones to exactly 0.
                        gate_opt.step() 
                    
                    scheduler.step()
                    if gate_opt:
                        gate_scheduler.step()

                    base_opt.zero_grad()
                    if gate_opt:
                        gate_opt.zero_grad()

            total_loss += loss.item()

            if accelerator.is_main_process:
                current_loss = total_loss / (step + 1)
                current_lr = scheduler.get_last_lr()[0]
                
                if adapter_name.lower() in ['sora', 'sdora']:
                    # Monitor sparsity in real-time
                    current_sparsity = get_sparsity_stats(model, threshold=sparsity_threshold)
                    pbar.set_postfix({
                        'loss': f'{current_loss:.4f}', 
                        'lr': f'{current_lr:.2e}', 
                        'sparsity': f'{current_sparsity:.3f}'
                    })
                else:
                    pbar.set_postfix({'loss': f'{current_loss:.4f}', 'lr': f'{current_lr:.2e}'})

                if adapter_name.lower() in ['sora', 'sdora']:
                    if (step % log_gate_stats_steps == 0):
                        log_gate_stats(model)
                    
                    if debug_print_steps and (step % debug_print_steps == 0):
                        ds = get_sora_delta_stats(model)
                        if ds:
                            ratios = sorted(ds, key=lambda x: x['ratio'], reverse=True)
                            topk = ratios[:3]
                            print(f"SoRA delta ratios (top {len(topk)}):")
                            for r in topk:
                                print(f"  {r['name']}: est_delta_max={r['est_delta_max']:.6e}, base_w_max={r['base_w_max']:.6e}, ratio={r['ratio']:.2f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False, disable=not accelerator.is_local_main_process):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
                    outputs = model(**batch)
                val_loss += outputs.loss.item()
        
        val_loss = accelerator.reduce(torch.tensor(val_loss).to(device), reduction="mean").item()
        val_loss /= len(val_loader)
        model.train()

        if accelerator.is_main_process:
            epoch_time = time.time() - epoch_start
            metrics = {
                'train_loss': total_loss / len(train_loader), 
                'val_loss': val_loss,
                'epoch_time': epoch_time
            }
            
            if adapter_name.lower() in ['sora', 'sdora']:
                metrics['sparsity'] = get_sparsity_stats(model, threshold=sparsity_threshold)
            
            tracker.update(metrics)
            tracker.print_summary(epoch + 1, metrics)
            
        accelerator.wait_for_everyone()

    # Final Save
    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}\n")
        
        unwrapped_model = accelerator.unwrap_model(model)
        
        if adapter_name.lower() in ['sora', 'sdora']:
            final_sparsity = get_sparsity_stats(unwrapped_model, threshold=sparsity_threshold)
            print(f"Final Sparsity (threshold {sparsity_threshold}): {final_sparsity:.4f}")
            
            
            print("Pruning SoRA model (Removing Zero-Gates)...")
            prune_sora_model(unwrapped_model)
            print("Pruning complete.")
        
        save_path = os.path.join(output_dir, "model.pt")
        merge_and_save(unwrapped_model, adapter_name, save_path)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✓ Model saved to: {save_path}")
        print(f"✓ History saved to: {os.path.join(output_dir, 'training_history.json')}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter_name", default="lora", choices=["lora", "sora", "dora", "sdora"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    # Default lambda set to 0.3, a strong value for >0.5 sparsity
    parser.add_argument("--sparse_lambda", type=float, default=0.3)
    # Default multiplier set to 10 to give gates high mobility
    parser.add_argument("--gate_lr_multiplier", type=int, default=10)
    parser.add_argument("--gate_lr", type=float, default=None)
    parser.add_argument("--debug_print_steps", type=int, default=0)
    
    # REVISION: Made this redundant by logic, but kept for script compatibility. 
    # Logic now defaults to PROX_ONLY behavior for SoRA.
    parser.add_argument("--use_prox_only", action='store_true', default=True, help="Enforce Proximal Step only (Default True)")
    
    parser.add_argument("--sparsity_threshold", type=float, default=1e-3)
    parser.add_argument("--log_gate_stats_steps", type=int, default=1000)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    train(
        base_model=args.base_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        adapter_name=args.adapter_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        sparse_lambda=args.sparse_lambda,
        gate_lr_multiplier=args.gate_lr_multiplier,
        gate_lr=args.gate_lr,
        debug_print_steps=args.debug_print_steps,
        use_prox_only=args.use_prox_only,
        sparsity_threshold=args.sparsity_threshold,
        log_gate_stats_steps=args.log_gate_stats_steps,
        micro_batch_size=args.micro_batch_size,
        batch_size=args.batch_size,
    )