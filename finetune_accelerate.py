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
from accelerate import Accelerator # <-- 新增

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

    # 1. 初始化 Accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    device = accelerator.device
    
    # 确保只有主进程进行文件系统操作
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

    # 移除 device_map="auto"，由 Accelerator 管理设备
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

    # Count parameters (仅在主进程打印)
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

    # 2. 使用 Accelerator.prepare 统一管理
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
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0
        
        # 禁用非主进程的 tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(pbar):
            # 3. 移除手动 to(device) 和 autocast
            # 将数据移动到设备 (保留，以防万一，但prepare通常已处理)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 使用 accelerator.accumulate 自动处理梯度累积
            with accelerator.accumulate(model):
                # 移除 with torch.autocast(...)
                outputs = model(**batch)
                loss = outputs.loss
                
                # 4. 使用 accelerator.backward() 自动处理混合精度和梯度累积
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # 在优化器执行前，使用 accelerator 包装的裁剪
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                    base_opt.step()
                    if gate_opt: gate_opt.step()
                    
                    scheduler.step()
                    if gate_opt: gate_scheduler.step()

                    base_opt.zero_grad()
                    if gate_opt: gate_opt.zero_grad()

            # 在这里，loss 已经是平均到 micro_batch_size 的损失
            total_loss += loss.item()

            # 更新进度条 (只有在主进程)
            if accelerator.is_main_process:
                current_loss = total_loss / (step + 1)
                pbar.set_postfix({'loss': f'{current_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False, disable=not accelerator.is_local_main_process):
                batch = {k: v.to(device) for k, v in batch.items()}
                # 在验证时仍保留 autocast，确保 bfloat16 计算
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
                    outputs = model(**batch)
                val_loss += outputs.loss.item()
        
        # 聚合验证损失（多卡时需要）
        # 单卡模式下，这仍是 val_loss，但如果未来扩展，这会确保正确平均
        val_loss = accelerator.reduce(torch.tensor(val_loss).to(device), reduction="mean").item()
        
        val_loss /= len(val_loader)
        model.train()

        # Collect metrics (仅主进程)
        if accelerator.is_main_process:
            epoch_time = time.time() - epoch_start
            metrics = {
                # total_loss 现在是在整个训练集上平均的 loss
                'train_loss': total_loss / len(train_loader), 
                'val_loss': val_loss,
                'epoch_time': epoch_time
            }
            
            # Add sparsity for SoRA/SDoRA
            if adapter_name.lower() in ['sora', 'sdora']:
                # 在主进程上计算稀疏度
                metrics['sparsity'] = get_sparsity_stats(model)
            
            # Update and print
            tracker.update(metrics)
            tracker.print_summary(epoch + 1, metrics)
            
        accelerator.wait_for_everyone() # 等待所有进程完成该 epoch

    # Final Save (仅主进程)
    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}\n")
        
        # 5. 解包模型并保存
        unwrapped_model = accelerator.unwrap_model(model)
        
        save_path = os.path.join(output_dir, "model.pt")
        merge_and_save(unwrapped_model, adapter_name, save_path)
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
    # 新增参数，便于控制，如果不需要可以删除
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