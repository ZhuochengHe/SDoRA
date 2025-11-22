# Using SoRA: A Step-by-Step Guide

This guide explains how to use the SoRA implementation (`sora.py`) based on the example in `train_sora.py`. SoRA (Sparse Low-Rank Adaptation) is a parameter-efficient fine-tuning method that adds sparse, low-rank branches to transformer layers.

**Note**: SoRA inherits from LoRA, sharing the same base structure while adding a gating mechanism for sparsity.

## Prerequisites

```python
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from sora import (
    wrap_linears,
    build_sora_optimizers,
    SoRA_Linear,
)
```

## Step 1: Load Model and Tokenizer

Load a pre-trained transformer model and its tokenizer:

```python
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
```

## Step 2: Select Target Layers

Choose which linear layers to adapt. For RoBERTa, target attention and feed-forward layers:

```python
target_layers = []
for layer_idx in range(12):  # RoBERTa has 12 encoder layers
    prefix = f"roberta.encoder.layer.{layer_idx}"
    target_layers.extend([
        f"{prefix}.attention.output.dense",    # Attention output projection
        f"{prefix}.intermediate.dense",        # Feed-forward expansion
        f"{prefix}.output.dense",              # Feed-forward contraction
    ])
```

## Step 3: Wrap Model with SoRA

Apply SoRA to the selected layers. This replaces the original linear layers with `SoRA_Linear` modules:

```python
wrap_linears(model, target_layers, r=32, lora_alpha=16, lora_dropout=0.05)
```

**Parameters:**
- `r`: Low-rank dimension (rank of A and B matrices)
- `lora_alpha`: Scaling factor for the adaptation strength (consistent with LoRA/DoRA naming)
- `lora_dropout`: Dropout probability for the low-rank branch (consistent with LoRA/DoRA naming)

## Step 4: Prepare Data

Create data loaders for your dataset. Here's an example for GLUE SST-2:

```python
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
```

## Step 5: Build Optimizers

Create separate optimizers for gate parameters (SoRA) and base model parameters. We use a higher learning rate for gates to encourage faster sparsification:

```python
gate_opt, base_opt = build_sora_optimizers(
    model,
    base_lr=1e-4,        # Learning rate for base parameters
    gate_lr=1e-3,        # Higher LR for gate parameters (10x base_lr)
    sparse_lambda=0.0,   # Initial sparsity regularization strength (can be scheduled)
    betas=(0.9, 0.98),   # AdamW betas
    weight_decay=0.01,   # Weight decay (automatically set to 0.0 for gates)
)
```

**Key Parameters:**
- `gate_lr`: Set higher than `base_lr` to allow the soft-thresholding mechanism to effectively prune gate values (which start at 1.0).
- `sparse_lambda`: Controls sparsity threshold. The actual threshold is `sparse_lambda * current_lr`.
- The function automatically splits parameters: gate parameters go to `gate_opt`, others to `base_opt`.
- **Note**: `weight_decay` is forced to `0.0` for `gate_opt` to ensure a pure Lasso ($L_1$) optimization problem.

## Step 6: Training Loop

### Basic Training (Recommended for Most Use Cases)

For normal usage, use a **fixed `sparse_lambda`** throughout training:

```python
# Use a single fixed lambda value
gate_opt, base_opt = build_sora_optimizers(
    model,
    base_lr=1e-4,
    gate_lr=1e-3,
    sparse_lambda=0.4,  # Fixed value (0.3-0.5 works well)
    betas=(0.9, 0.98),
    weight_decay=0.01,
)

# Train normally
for epoch in range(num_epochs):
    for batch in train_loader:
        # ... forward pass, loss.backward() ...
        base_opt.step()
        gate_opt.step()
        base_opt.zero_grad()
        gate_opt.zero_grad()
```

### Advanced: Scheduling λ (Algorithm 1) - Research Use Only

**Note**: Scheduling is an **experimental tool** for exploring compression curves, not required for normal SoRA usage.

#### How It Works

**Stage** = A training phase using a fixed `xi` (sparse lambda) value. By training with progressively larger `xi`, you explore model performance at different sparsity levels.

**Relationship**: `xi` → `threshold` → `gate_zero_ratio` → `sparsity`

- Larger `xi` → Higher threshold in soft-thresholding → More gates pushed to 0 → Higher sparsity
- Each stage produces a different sparsity level, letting you plot **accuracy vs sparsity curves**

**Example**: With `xi_start=0.0, xi_max=0.5, xi_step=0.1`, you get **6 stages**:

| Stage | xi | Expected Gate Zero Ratio | Purpose |
|---|---|---|---|
| 0 | 0.0 | ~0% | Dense baseline |
| 1 | 0.1 | ~20% | Mild sparsity |
| 2 | 0.2 | ~40% | Moderate sparsity |
| 3 | 0.3 | ~60% | High sparsity |
| 4 | 0.4 | ~75% | Very high sparsity |
| 5 | 0.5 | ~85% | Extreme sparsity |

Each stage trains for `epochs_per_stage` epochs (e.g., 3), giving you data points to analyze the compression-performance tradeoff.

#### Code Example

```python
# WARNING: This is for research experiments only
xi_start = 0.0
xi_max = 0.5
xi_step = 0.1
epochs_per_stage = 3

current_xi = xi_start
stage = 0

while current_xi <= xi_max:
    print(f"Stage {stage}: xi={current_xi}")
    
    gate_opt, base_opt = build_sora_optimizers(
        model, sparse_lambda=current_xi, ...
    )
    
    for epoch in range(epochs_per_stage):
        train_one_epoch(...)
        # Record: val_acc, gate_zero_ratio
    
    current_xi += xi_step
    stage += 1

# Result: 6 stages × 3 epochs = 18 total epochs
```

**Use scheduling only if**: You want to study how sparsity affects performance and plot compression curves. For production models, use a single fixed `sparse_lambda`.

## Step 7: Pruning and Merging

After training, you can prune the model to remove zeroed-out ranks, or merge the weights into the base model for zero-overhead inference.

### Pruning
Physically removes rows/columns where the gate is zero. The model remains a `SoRA_Linear` but with reduced rank.

```python
from sora import prune_sora_model

prune_sora_model(model)
# Verify performance...
```

### Merging
Merges the adapter weights into the base model weights and removes the adapter entirely. The model becomes a standard dense model.

```python
from sora import merge_sora_model

merge_sora_model(model)
# Verify performance...
# Save the merged model
model.save_pretrained("path/to/merged_model")
```
num_epochs = 3
num_training_steps = len(train_loader) * num_epochs

base_scheduler = get_linear_schedule_with_warmup(
    base_opt, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
)
gate_scheduler = get_linear_schedule_with_warmup(
    gate_opt, 
    num_warmup_steps=int(0.02 * num_training_steps), # Short warmup (2%) for stability
    num_training_steps=num_training_steps
)
```

## Step 7: Define Evaluation Function

Create a function to evaluate model performance:

```python
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
```

## Step 8: Monitor Sparsity

Track how many gate parameters have been pruned to zero:

```python
def summarize_gate_sparsity(model: torch.nn.Module) -> float:
    total = 0
    zeros = 0
    for module in model.modules():
        if isinstance(module, SoRA_Linear) and module.gate is not None:
            gate = module.gate.data
            total += gate.numel()
            zeros += (gate == 0).sum().item()
    return zeros / total if total > 0 else 0.0
```

## Step 9: Training Loop

The main training loop with gradient accumulation, optimization, and logging:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

global_step = 0
for epoch in range(1, num_epochs + 1):
    running_loss = 0.0
    running_correct = 0
    running_examples = 0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")

        # Forward pass
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        loss.backward()

        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Accuracy tracking
        preds = outputs.logits.argmax(dim=-1)
        running_correct += (preds == labels).sum().item()
        running_examples += labels.size(0)

        # Optimization steps
        gate_opt.step()
        base_opt.step()
        gate_scheduler.step()
        base_scheduler.step()
        gate_opt.zero_grad()
        base_opt.zero_grad()

        # Logging
        running_loss += loss.item()
        global_step += 1
        if global_step % 100 == 0:
            avg_loss = running_loss / 100
            avg_acc = running_correct / running_examples if running_examples else 0.0
            print(f"Epoch {epoch} | Step {global_step} | loss = {avg_loss:.4f} | accuracy = {avg_acc:.4f}")
            running_loss = 0.0
            running_correct = 0
            running_examples = 0

    # End-of-epoch evaluation
    val_loss, val_acc = evaluate(model, val_loader)
    gate_zero_ratio = summarize_gate_sparsity(model)
    print(f"Epoch {epoch} validation -> loss: {val_loss:.4f}, acc: {val_acc:.4f}, gate zero ratio: {gate_zero_ratio:.4f}")
```

## Key Concepts

1. **Split Optimization**: Gate parameters (controlling sparsity) are optimized separately from base model parameters.

2. **Proximal Gradient**: The `gate_opt.step()` applies soft thresholding to induce sparsity.

3. **Layer Selection**: Choose layers that have the most impact on task performance (attention and feed-forward layers).

4. **Hyperparameter Tuning**:
   - `r`: Higher rank = more capacity but more parameters
   - `sparse_lambda`: Higher = more sparsity
   - `alpha`: Higher = stronger adaptation

## Customization

- **Different Models**: Change `model_name` and adjust `target_layers` for other architectures
- **Custom Datasets**: Modify `encode_batch` and data loading logic
- **Advanced Scheduling**: Implement custom learning rate schedules or sparsity schedules
- **Multi-GPU**: Add `torch.nn.DataParallel` or distributed training

## Expected Output

After training, you should see:
- Decreasing loss and increasing accuracy
- Increasing `gate_zero_ratio` (sparsity)
- Final parameter dump showing pruned gates

This minimal implementation provides full control over the training process while maintaining the core SoRA algorithm.