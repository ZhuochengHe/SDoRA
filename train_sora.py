import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from sora_implementation.sora import (
    wrap_linears,
    build_sora_optimizers,
    SoRA_Linear,
    prune_sora_model,
    merge_sora_model,
)

from sd_lora_implementation.sd_lora import SDoRA_Linear

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 1. Select target linear layers (cover self-attention and feed-forward projections)
target_layers = []
for layer_idx in range(12):
    prefix = f"roberta.encoder.layer.{layer_idx}"
    target_layers.extend(
        [
            f"{prefix}.attention.output.dense",
            f"{prefix}.intermediate.dense",
            f"{prefix}.output.dense",
        ]
    )
wrap_linears(model, target_layers, r=32, lora_alpha=16, lora_dropout=0.05)

# 2. Build dataloaders (GLUE/SST-2 example)
def encode_batch(batch):
    sentences = [sample["sentence"] for sample in batch]
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    encoded["labels"] = labels
    return encoded

from datasets import load_dataset
dataset = load_dataset("glue", "sst2")
train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=True, collate_fn=encode_batch)
val_loader = DataLoader(dataset["validation"], batch_size=64, shuffle=False, collate_fn=encode_batch)

# 3. Build SoRA + standard optimizers
# We will rebuild optimizers inside the training loop for the scheduler
# But for initial setup, we define the params
base_lr = 1e-4
gate_lr = 1e-3
weight_decay = 0.01

def evaluate(model: torch.nn.Module, dataloader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    model.train()
    return total_loss / total_samples, total_correct / total_samples


def summarize_gate_sparsity(model: torch.nn.Module) -> float:
    eps = 1e-6
    total = 0
    zeros = 0
    for module in model.modules():
        if isinstance(module, SDoRA_Linear) and module.gate is not None:
            gate = module.gate.data
            total += gate.numel()
            zeros += (gate.abs() < eps).sum().item()
    return zeros / total if total > 0 else 0.0


def dump_sora_weights(model: torch.nn.Module) -> None:
    print("\n=== SoRA parameter dump ===")
    for name, module in model.named_modules():
        if isinstance(module, SDoRA_Linear) and module.gate is not None:
            print(f"Module: {name}")
            print("lora_A:\n", module.lora_A.weight.data)
            print("lora_B:\n", module.lora_B.weight.data)
            print("gate:\n", module.gate.data)

def train_one_epoch(model, train_loader, base_opt, gate_opt, base_scheduler, gate_scheduler, epoch_idx):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_examples = 0
    global_step = 0
    
    for batch in train_loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}

        labels = batch.pop("labels")
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        preds = outputs.logits.argmax(dim=-1)
        running_correct += (preds == labels).sum().item()
        running_examples += labels.size(0)

        # Step regular optimizer first, then gate optimizer (so gate's proximal step is last)
        base_opt.step()
        gate_opt.step()
        # Step schedulers after optimizers
        base_scheduler.step()
        gate_scheduler.step()
        # Zero grads
        base_opt.zero_grad()
        gate_opt.zero_grad()

        running_loss += loss.item()
        global_step += 1
        if global_step % 100 == 0:
            avg_loss = running_loss / 100
            avg_acc = running_correct / running_examples if running_examples else 0.0
            print(
                f"Epoch {epoch_idx} | Step {global_step} | loss = {avg_loss:.4f} | accuracy = {avg_acc:.4f}"
            )
            running_loss = 0.0
            running_correct = 0
            running_examples = 0

# 5. Training loop with Fixed Lambda (Recommended)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fixed sparsity parameter
sparse_lambda = 0.3  # Use a fixed value (0.2-0.4 works well for most tasks)
num_epochs = 5

# Build optimizers once with fixed lambda
gate_opt, base_opt = build_sora_optimizers(
    model,
    base_lr=base_lr,
    gate_lr=gate_lr,
    sparse_lambda=sparse_lambda,
    betas=(0.9, 0.98),
    weight_decay=weight_decay,
)

num_training_steps = len(train_loader) * num_epochs
base_scheduler = get_linear_schedule_with_warmup(
    base_opt, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
)
gate_scheduler = get_linear_schedule_with_warmup(
    gate_opt, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
)

# Training loop
for epoch in range(1, num_epochs + 1):
    train_one_epoch(model, train_loader, base_opt, gate_opt, base_scheduler, gate_scheduler, epoch)
    
    val_loss, val_acc = evaluate(model, val_loader)
    gate_zero_ratio = summarize_gate_sparsity(model)
    print(
        f"Epoch {epoch} validation -> loss: {val_loss:.4f}, acc: {val_acc:.4f}, gate zero ratio: {gate_zero_ratio:.4f}"
    )

# ============================================================================
# OPTIONAL: Scheduling (Algorithm 1) - For research experiments only
# ============================================================================
# Uncomment below to explore accuracy vs sparsity curves with progressive lambda
"""
# Scheduler parameters
xi_start = 0.0
xi_max = 0.5
xi_step = 0.1
epochs_per_stage = 3

current_xi = xi_start
stage = 0

while current_xi <= xi_max + 1e-6:
    print(f"\n=== Starting Stage {stage} with lambda (xi) = {current_xi:.2f} ===")
    
    # Rebuild optimizers with new lambda
    gate_opt, base_opt = build_sora_optimizers(
        model,
        base_lr=base_lr,
        gate_lr=gate_lr,
        sparse_lambda=current_xi,
        betas=(0.9, 0.98),
        weight_decay=weight_decay,
    )
    
    num_training_steps = len(train_loader) * epochs_per_stage
    base_scheduler = get_linear_schedule_with_warmup(
        base_opt, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
    )
    gate_scheduler = get_linear_schedule_with_warmup(
        gate_opt, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
    )

    for epoch in range(1, epochs_per_stage + 1):
        train_one_epoch(model, train_loader, base_opt, gate_opt, base_scheduler, gate_scheduler, epoch)
        
        val_loss, val_acc = evaluate(model, val_loader)
        gate_zero_ratio = summarize_gate_sparsity(model)
        print(
            f"Stage {stage} Epoch {epoch} validation -> loss: {val_loss:.4f}, acc: {val_acc:.4f}, gate zero ratio: {gate_zero_ratio:.4f}"
        )

    current_xi += xi_step
    stage += 1
"""
# ============================================================================

dump_sora_weights(model)

print("\n=== Pruning and Merging SoRA Model ===")
prune_sora_model(model)
print("Pruning complete. Merging into base weights...")
merge_sora_model(model)
print("Merge complete. Verifying post-merge performance...")
val_loss, val_acc = evaluate(model, val_loader)
print(f"Post-merge validation -> loss: {val_loss:.4f}, acc: {val_acc:.4f}")