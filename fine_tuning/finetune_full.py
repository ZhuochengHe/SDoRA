import os
import json
import time
import torch
import csv
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import argparse
from accelerate import Accelerator


def get_full_tuning_optimizer(model, lr: float):
    return torch.optim.AdamW(model.parameters(), lr=lr)

def save_full_model(model, tokenizer, output_dir):
    unwrapped_model = model
    if hasattr(model, 'module'):
         unwrapped_model = model.module
         
    unwrapped_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

class MetricsTracker:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.history = {
            'train_loss': [], 'val_loss': [],
            'epoch_time': [],
            'gpu_memory_mb': [], 'trainable_params': []
        }
        self.csv_path = os.path.join(output_dir, 'loss_log.csv')
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'train_loss', 'learning_rate', 'gpu_memory_mb'])
        
        self.sparsity_csv_path = None
        self.sparsity_initialized = False
    
    def log_step(self, epoch, step, loss, lr, gpu_memory):
        """Log loss to CSV every k steps"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, f'{loss:.6f}', f'{lr:.6e}', f'{gpu_memory:.2f}'])
    
    def log_sparsity_details(self, epoch, step, layer_sparsity_dict):
        pass
    
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
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024 / 1024 
    return 0.0

def compute_trainable_params(model):
    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
    
    return total_params, trainable_params


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
            raise ValueError(f"[collate] Missing input_ids in sample {i}")
        else:
            if not isinstance(x["input_ids"], (list, tuple)):
                raise TypeError(f"Bad input_ids type at sample {i}: {type(x['input_ids'])}")
    
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = [x["input_ids"] + [0]*(max_len-len(x["input_ids"])) for x in batch]
    attention_mask = [x["attention_mask"] + [0]*(max_len-len(x["attention_mask"])) for x in batch]
    labels = [x["labels"] + [-100]*(max_len-len(x["labels"])) for x in batch]
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels)
    }



def full_parameter_train(
    base_model: str,
    data_path: str,
    output_dir: str,
    batch_size: int = 32,
    micro_batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    cutoff_len: int = 256,
    val_set_size: int = 120,
):

    gradient_accumulation_steps = batch_size // micro_batch_size

    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    device = accelerator.device
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n{'='*70}")
        print("Training FULL PARAMETER")
        print(f"{'='*70}")
        print(f"Model: {base_model}")
        print(f"Epochs: {num_epochs}, LR: {learning_rate}")
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
    
    for p in model.parameters():
        p.requires_grad_(True)
    
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    if accelerator.is_main_process:
        total_params, trainable_params = compute_trainable_params(model)

        ratio = 100.0 * trainable_params / total_params

        print("\n[DEBUG] Parameter freezing status (Full Tuning)")
        print(f"  Total parameters      : {total_params:,}")
        print(f"  Trainable params      : {trainable_params:,} ({ratio:.2f}%)")
        print("  NOTE: ratio should be close to 100.00% for Full Tuning.\n")

    # Load dataset & Tokenize
    if data_path.endswith(".json"):
        data = load_dataset("json", data_files=data_path)["train"]
    else:
        data = load_dataset(data_path)["train"]

    # Split train/val
    data = data.train_test_split(test_size=val_set_size, seed=42)
    train_data = data["train"]
    val_data = data["test"]

    train_data = train_data.map(lambda batch: generate_and_tokenize_prompt(batch, tokenizer, cutoff_len), remove_columns=train_data.column_names, batched=True, num_proc=4)
    val_data = val_data.map(lambda batch: generate_and_tokenize_prompt(batch, tokenizer, cutoff_len), remove_columns=val_data.column_names, batched=True, num_proc=2)

    # Dataloader
    train_loader = DataLoader(train_data, batch_size=micro_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_data, batch_size=micro_batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    optimizer = get_full_tuning_optimizer(model, lr=learning_rate)

    num_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=num_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    tracker = MetricsTracker(output_dir)
    
    if accelerator.is_main_process:
        print(f"{'='*70}")
        print("Starting Full Parameter Training")
        print(f"{'='*70}\n")
    
    model.train()
    
    log_steps = 100
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    
                    scheduler.step()

                    optimizer.zero_grad()

            total_loss += loss.item()

            # Update progress bar and log GPU memory
            if accelerator.is_main_process:
                current_loss = total_loss / (step + 1)
                current_lr = scheduler.get_last_lr()[0]
                gpu_mem = get_gpu_memory()
                
                pbar.set_postfix({'loss': f'{current_loss:.4f}', 'lr': f'{current_lr:.2e}', 'gpu_mb': f'{gpu_mem:.0f}'})
                
                if (step + 1) % log_steps == 0:
                    tracker.log_step(epoch + 1, step + 1, current_loss, current_lr, gpu_mem)


        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            unwrapped_model = accelerator.unwrap_model(model)
            for batch in tqdm(val_loader, desc="Validating", leave=False, disable=not accelerator.is_local_main_process):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = unwrapped_model(**batch)
                val_loss += outputs.loss.item()
        
        val_loss = accelerator.reduce(torch.tensor(val_loss).to(device), reduction="mean").item()
        val_loss /= len(val_loader)
        model.train()

        if accelerator.is_main_process:
            epoch_time = time.time() - epoch_start
            avg_train_loss = total_loss / len(train_loader)
            
            total_params, trainable_params = compute_trainable_params(model)
            gpu_mem = get_gpu_memory()
            
            metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss, 
                'val_loss': val_loss,
                'epoch_time': epoch_time,
                'gpu_memory_mb': gpu_mem,
                'trainable_params': trainable_params,
            }
            
            # Update and log to JSON
            tracker.update(metrics)
            tracker.print_summary(epoch + 1, metrics)
            
            epoch_ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch+1}")
            os.makedirs(epoch_ckpt_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            

            torch.save(unwrapped_model.state_dict(), os.path.join(epoch_ckpt_dir, "model.pt"))
            unwrapped_model.config.save_pretrained(epoch_ckpt_dir)
            tokenizer.save_pretrained(epoch_ckpt_dir) 
            
            print(f"✓ Epoch checkpoint saved to: {epoch_ckpt_dir}/model.pt")
            
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}\n")
        
        final_model_path = os.path.join(output_dir, "model.pt")
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), final_model_path)
        tokenizer.save_pretrained(output_dir)
        unwrapped_model.config.save_pretrained(output_dir)
        
        print(f"✓ Final model weights saved to: {final_model_path}")
        print(f"✓ Training log (JSON) saved to: {os.path.join(output_dir, 'training_history.json')}")
        print(f"✓ Loss log (CSV) saved to: {tracker.csv_path}")
        print(f"\n📊 Training Summary:")
        print(f"  Total Epochs: {num_epochs}")
        print(f"  Finetuning Method: Full Parameter Tuning")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Output Directory: {output_dir}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--data_path", default="/home/ubuntu/LLM-inference/liangzhao-project/yunqi/DoRA-SoRA-and-LoRA/adapters/dora/commonsense_reasoning/commonsense_170k.json")
    parser.add_argument("--output_dir", required=True)
    # parser.add_argument("--adapter_name", default="lora", choices=["lora", "sora", "dora", "sdora"])
    # parser.add_argument("--lora_r", type=int, default=16) 
    # parser.add_argument("--lora_alpha", type=int, default=32) 
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5) 
    # parser.add_argument("--sparse_lambda", type=float, default=0.3)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cutoff_len", type=int, default=256)

    args = parser.parse_args()

    full_parameter_train(
        base_model=args.base_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        micro_batch_size=args.micro_batch_size,
        batch_size=args.batch_size,
        cutoff_len=args.cutoff_len,
    )