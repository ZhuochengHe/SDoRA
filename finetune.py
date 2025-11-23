import os
import json
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from utils import replace_linear_with_lora, get_optimizer, merge_and_save
from tqdm import tqdm
import argparse

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


def get_sparsity_stats(model):
    total_gates = 0
    zero_gates = 0
    
    for module in model.modules():
        if hasattr(module, 'gate') and module.gate is not None:
            gate = module.gate
            total_gates += gate.numel()
            zero_gates += (gate.abs() < 1e-6).sum().item()
    
    return zero_gates / total_gates if total_gates > 0 else 0.0


def generate_prompt(data_point):  # from original dora finetune file
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


def tokenize(prompt, tokenizer, cutoff_len=256):  # from original dora finetune file
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result


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
    micro_batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    sparse_lambda: float = 0.3,
    cutoff_len: int = 256,
    val_set_size: int = 120,
):
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Training {adapter_name.upper()}")
    print(f"{'='*70}")
    print(f"Model: {base_model}")
    print(f"Rank: {lora_r}, Alpha: {lora_alpha}, Epochs: {num_epochs}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply adapter
    model = replace_linear_with_lora(
        model, target_modules, adapter_name, 
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
    )

    # Count parameters
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
    def generate_and_tokenize_prompt(data_point):
        return tokenize(generate_prompt(data_point), tokenizer, cutoff_len)

    train_data = train_data.map(generate_and_tokenize_prompt, remove_columns=train_data.column_names)
    val_data = val_data.map(generate_and_tokenize_prompt, remove_columns=val_data.column_names)

    # Dataloader
    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = [x["input_ids"] + [0]*(max_len-len(x["input_ids"])) for x in batch]
        attention_mask = [x["attention_mask"] + [0]*(max_len-len(x["attention_mask"])) for x in batch]
        labels = [x["labels"] + [-100]*(max_len-len(x["labels"])) for x in batch]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)
        }

    train_loader = DataLoader(train_data, batch_size=micro_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=micro_batch_size, collate_fn=collate_fn)

    # Optimizer
    base_opt, gate_opt = get_optimizer(model, adapter_name, lr=learning_rate, sparse_lambda=sparse_lambda)

    # Scheduler
    gradient_accumulation_steps = batch_size // micro_batch_size
    num_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(base_opt, num_warmup_steps=20, num_training_steps=num_steps)
    if gate_opt:
        gate_scheduler = get_linear_schedule_with_warmup(gate_opt, num_warmup_steps=20, num_training_steps=num_steps)

    # Metrics tracker
    tracker = MetricsTracker(output_dir)
    
    # Training loop
    print(f"{'='*70}")
    print("Starting Training")
    print(f"{'='*70}\n")
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0
        
        # Progress bar with real-time metrics
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(pbar):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                base_opt.step()
                if gate_opt: gate_opt.step()
                
                scheduler.step()
                if gate_opt: gate_scheduler.step()

                base_opt.zero_grad()
                if gate_opt: gate_opt.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            
            # Update progress bar
            current_loss = total_loss / (step + 1)
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)
        model.train()

        # Collect metrics
        epoch_time = time.time() - epoch_start
        metrics = {
            'train_loss': total_loss / len(train_loader),
            'val_loss': val_loss,
            'epoch_time': epoch_time
        }
        
        # Add sparsity for SoRA/SDoRA
        if adapter_name.lower() in ['sora', 'sdora']:
            metrics['sparsity'] = get_sparsity_stats(model)
        
        # Update and print
        tracker.update(metrics)
        tracker.print_summary(epoch + 1, metrics)

    # Save
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}\n")
    
    save_path = os.path.join(output_dir, "model.pt")
    merge_and_save(model, adapter_name, save_path)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Model saved to: {save_path}")
    print(f"✓ History saved to: {os.path.join(output_dir, 'training_history.json')}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--data_path", default="/home/ubuntu/LLM-inference/liangzhao-project/yunqi/DoRA-SoRA-and-LoRA/adapters/dora/commonsense_reasoning/commonsense_170k.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter_name", default="lora", choices=["lora", "sora", "dora", "sdora"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--sparse_lambda", type=float, default=0.3)

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
    )