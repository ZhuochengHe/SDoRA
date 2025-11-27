import os
import json
import time
import torch
import csv
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from utils import replace_linear_with_lora, get_optimizer, merge_and_save, log_gate_stats
from tqdm import tqdm
import argparse
from accelerate import Accelerator

class MetricsTracker:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.history = {
            'train_loss': [], 'val_loss': [],
            'epoch_time': [], 'sparsity': [],
            'gpu_memory_mb': [], 'effective_params': []
        }
        # CSV file to record loss every k steps
        self.csv_path = os.path.join(output_dir, 'loss_log.csv')
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'train_loss', 'learning_rate', 'gpu_memory_mb', 'sparsity'])
        
        # CSV file for detailed sparsity tracking (SoRA/SDoRA only)
        self.sparsity_csv_path = os.path.join(output_dir, f'sparsity_log.csv')
        self.sparsity_initialized = False
    
    def log_step(self, epoch, step, loss, lr, gpu_memory, sparsity=None):
        """Log loss to CSV every k steps"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            sparsity_str = f'{sparsity:.6f}' if sparsity is not None else 'N/A'
            writer.writerow([epoch, step, f'{loss:.6f}', f'{lr:.6e}', f'{gpu_memory:.2f}', sparsity_str])
    
    def log_sparsity_details(self, epoch, step, layer_sparsity_dict):
        """Log per-layer sparsity details for SoRA/SDoRA"""
        if not self.sparsity_initialized:
            # Initialize CSV with layer names as headers
            with open(self.sparsity_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = ['epoch', 'step', 'overall_sparsity'] + sorted(layer_sparsity_dict.keys())
                writer.writerow(headers)
            self.sparsity_initialized = True
        
        # Write sparsity data
        with open(self.sparsity_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            overall_sparsity = sum(layer_sparsity_dict.values()) / len(layer_sparsity_dict) if layer_sparsity_dict else 0.0
            row = [epoch, step, f'{overall_sparsity:.6f}']
            for layer_name in sorted(layer_sparsity_dict.keys()):
                row.append(f'{layer_sparsity_dict[layer_name]:.6f}')
            writer.writerow(row)
    
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
            elif isinstance(v, int):
                print(f"  {k:20s}: {v:,}")
        print(f"{'='*70}\n")


def get_gpu_memory():
    """Get current GPU memory usage (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

def get_sparsity_stats(model, detailed=False):
    """Calculate sparsity statistics
    
    Args:
        model: The model to analyze
        detailed: If True, return per-layer sparsity dict; if False, return overall sparsity float
    
    Returns:
        float or dict: Overall sparsity or per-layer sparsity dictionary
    """
    total_gates = 0
    zero_gates = 0
    layer_stats = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'gate') and module.gate is not None:
            gate = module.gate
            layer_total = gate.numel()
            layer_zeros = (gate.abs() < 1e-6).sum().item()
            
            total_gates += layer_total
            zero_gates += layer_zeros
            
            if detailed:
                layer_sparsity = layer_zeros / layer_total if layer_total > 0 else 0.0
                # Simplify layer name for readability
                simple_name = name.replace('base_model.model.', '').replace('.base_layer', '')
                layer_stats[simple_name] = layer_sparsity
    
    if detailed:
        return layer_stats
    else:
        return zero_gates / total_gates if total_gates > 0 else 0.0

def compute_effective_params(model, adapter_config):
    """Calculate effective parameter count (accounting for sparsity)"""
    trainable_params = 0
    effective_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            trainable_params += param_count
            
            # For SoRA/SDoRA, only non-zero elements in gate parameters count as effective
            if adapter_config['adapter_name'].lower() in ['sora', 'sdora'] and 'gate' in name:
                effective_params += (param.abs() >= 1e-6).sum().item()
            else:
                effective_params += param_count
    
    return trainable_params, effective_params


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


def generate_and_tokenize_prompt(batch, tokenizer):
    # 结合了原代码中的 generate_prompt 和 tokenization 逻辑
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
    # 注意：这里 hardcode 的 max_length=256 应该与 train 参数中的 cutoff_len 保持一致，
    # 但为简化，我们暂时保留 256
    tokens = tokenizer(
        prompts,
        truncation=True,
        max_length=256,
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


# Dataloader Collator 保持不变
def collate_fn(batch):
    # 错误检查逻辑保持不变
    for i, x in enumerate(batch):
        if "input_ids" not in x:
            print(f"[collate] Missing input_ids in sample {i}: keys={list(x.keys())}")
        else:
            if not isinstance(x["input_ids"], (list, tuple)):
                print(f"[collate] BAD input_ids type in sample {i}: {type(x['input_ids'])} -> value={x['input_ids']}")
                print("Full sample:", x)
                raise TypeError(f"Bad input_ids type at sample {i}: {type(x['input_ids'])}")
    
    # 填充逻辑保持不变
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
    cutoff_len: int = 256,
    val_set_size: int = 120,
):
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]

    gradient_accumulation_steps = batch_size // micro_batch_size

    # 1. Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    device = accelerator.device
    
    # Ensure only main process performs file system operations
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

    # Remove device_map="auto", let Accelerator manage devices
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
    
    # Save adapter config for later calculations
    adapter_config = {
        'adapter_name': adapter_name,
        'r': lora_r,
        'alpha': lora_alpha,
        'dropout': lora_dropout,
        'target_modules': target_modules
    }

    # Count parameters (only print on main process)
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
    train_data = train_data.map(lambda batch: generate_and_tokenize_prompt(batch, tokenizer), remove_columns=train_data.column_names, batched=True, num_proc=4)
    val_data = val_data.map(lambda batch: generate_and_tokenize_prompt(batch, tokenizer), remove_columns=val_data.column_names, batched=True, num_proc=2)

    # Dataloader
    train_loader = DataLoader(train_data, batch_size=micro_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_data, batch_size=micro_batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # Optimizer
    base_opt, gate_opt = get_optimizer(model, adapter_name, lr=learning_rate, sparse_lambda=sparse_lambda)

    # Scheduler
    num_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(base_opt, num_warmup_steps=20, num_training_steps=num_steps)
    if gate_opt:
        gate_scheduler = get_linear_schedule_with_warmup(gate_opt, num_warmup_steps=20, num_training_steps=num_steps)

    # 2. Use Accelerator.prepare for unified management
    model, base_opt, train_loader, val_loader, scheduler = accelerator.prepare(
        model, base_opt, train_loader, val_loader, scheduler
    )
    if gate_opt:
        gate_opt, gate_scheduler = accelerator.prepare(gate_opt, gate_scheduler)

    # Metrics tracker
    tracker = MetricsTracker(output_dir)
    
    # Training loop
    if accelerator.is_main_process:
        print(f"{'='*70}")
        print("Starting Training")
        print(f"{'='*70}\n")
    
    model.train()
    
    # Add checkpoint save configuration
    save_steps = 3000  # Save checkpoint every 500 steps
    log_steps = 100   # Log to CSV every 100 steps
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0
        
        # Disable tqdm for non-main processes
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Use accelerator.accumulate for automatic gradient accumulation
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                    base_opt.step()
                    if gate_opt: gate_opt.step()
                    
                    scheduler.step()
                    if gate_opt: gate_scheduler.step()

                    base_opt.zero_grad()
                    if gate_opt: gate_opt.zero_grad()

            total_loss += loss.item()

            # Update progress bar and log GPU memory
            if accelerator.is_main_process:
                current_loss = total_loss / (step + 1)
                current_lr = scheduler.get_last_lr()[0]
                gpu_mem = get_gpu_memory()
                
                # Calculate sparsity for SoRA/SDoRA
                current_sparsity = None
                if adapter_config['adapter_name'].lower() in ['sora', 'sdora']:
                    current_sparsity = get_sparsity_stats(model, detailed=False)
                    pbar.set_postfix({'loss': f'{current_loss:.4f}', 'lr': f'{current_lr:.2e}', 'gpu_mb': f'{gpu_mem:.0f}', 'sparsity': f'{current_sparsity:.3f}'})
                else:
                    pbar.set_postfix({'loss': f'{current_loss:.4f}', 'lr': f'{current_lr:.2e}', 'gpu_mb': f'{gpu_mem:.0f}'})
                
                # Log to CSV every log_steps
                if (step + 1) % log_steps == 0:
                    tracker.log_step(epoch + 1, step + 1, current_loss, current_lr, gpu_mem, current_sparsity)

                    log_gate_stats(model)
                    # Log detailed per-layer sparsity for SoRA/SDoRA
                    if adapter_config['adapter_name'].lower() in ['sora', 'sdora']:
                        layer_sparsity = get_sparsity_stats(model, detailed=True)
                        tracker.log_sparsity_details(epoch + 1, step + 1, layer_sparsity)
                
                # Save checkpoint every save_steps
                if (step + 1) % save_steps == 0:
                    ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch+1}-step{step+1}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
                    print(f"\n✓ Checkpoint saved to: {ckpt_dir}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False, disable=not accelerator.is_local_main_process):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
                    outputs = model(**batch)
                val_loss += outputs.loss.item()
        
        # Aggregate validation loss (needed for multi-GPU)
        val_loss = accelerator.reduce(torch.tensor(val_loss).to(device), reduction="mean").item()
        val_loss /= len(val_loader)
        model.train()

        # Collect metrics (main process only)
        if accelerator.is_main_process:
            epoch_time = time.time() - epoch_start
            avg_train_loss = total_loss / len(train_loader)
            
            # Calculate parameter statistics
            trainable_params, effective_params = compute_effective_params(model, adapter_config)
            gpu_mem = get_gpu_memory()
            
            metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss, 
                'val_loss': val_loss,
                'epoch_time': epoch_time,
                'gpu_memory_mb': gpu_mem,
                'trainable_params': trainable_params,
                'effective_params': effective_params
            }
            
            # Add sparsity for SoRA/SDoRA
            if adapter_name.lower() in ['sora', 'sdora']:
                metrics['sparsity'] = get_sparsity_stats(model)
            
            # Update and log to JSON
            tracker.update(metrics)
            tracker.print_summary(epoch + 1, metrics)
            
            # Save checkpoint at the end of each epoch
            epoch_ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch+1}")
            os.makedirs(epoch_ckpt_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(epoch_ckpt_dir, "model.pt"))
            print(f"✓ Epoch checkpoint saved to: {epoch_ckpt_dir}")
            
        accelerator.wait_for_everyone()  # Wait for all processes to complete this epoch

    # Final Save (main process only)
    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}\n")
        
        # Unwrap model and save
        unwrapped_model = accelerator.unwrap_model(model)
        
        save_path = os.path.join(output_dir, "model.pt")
        merge_and_save(unwrapped_model, adapter_name, save_path)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✓ Model saved to: {save_path}")
        print(f"✓ Training log (JSON) saved to: {os.path.join(output_dir, 'training_log.json')}")
        print(f"✓ Loss log (CSV) saved to: {tracker.csv_path}")
        print(f"\n📊 Training Summary:")
        print(f"  Total Epochs: {num_epochs}")
        print(f"  Adapter: {adapter_name}")
        print(f"  LoRA r: {adapter_config.get('r', 'N/A')}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Output Directory: {output_dir}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--data_path", default="/home/ubuntu/LLM-inference/liangzhao-project/yunqi/DoRA-SoRA-and-LoRA/adapters/dora/commonsense_reasoning/commonsense_170k.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter_name", default="lora", choices=["lora", "sora", "dora", "sdora"])
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--sparse_lambda", type=float, default=0.3)
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
        micro_batch_size=args.micro_batch_size,
        batch_size=args.batch_size,
    )