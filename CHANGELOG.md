# Changelog for finetune_accelerate.py

## Summary
Enhanced the training script with comprehensive logging, monitoring, and sparsity tracking capabilities for LoRA/DoRA/SoRA/SDoRA experiments.

---

## Changes Made

### 1. **Enhanced MetricsTracker Class**

#### Added CSV Logging for Step-Level Metrics
- **New attribute**: `csv_path` - Path to `loss_log.csv`
- **New method**: `log_step(epoch, step, loss, lr, gpu_memory, sparsity=None)`
  - Records training metrics every k steps (default: 100)
  - Columns: `[epoch, step, train_loss, learning_rate, gpu_memory_mb, sparsity]`

#### Added Detailed Sparsity Tracking (SoRA/SDoRA)
- **New attribute**: `sparsity_csv_path` - Path to `sparsity_log.csv`
- **New attribute**: `sparsity_initialized` - Flag for CSV header initialization
- **New method**: `log_sparsity_details(epoch, step, layer_sparsity_dict)`
  - Records per-layer sparsity statistics
  - Dynamic headers based on model architecture
  - Columns: `[epoch, step, overall_sparsity, layer1, layer2, ...]`

#### Updated History Tracking
- Added `gpu_memory_mb` to history dict
- Added `effective_params` to history dict

---

### 2. **New Utility Functions**

#### GPU Memory Monitoring
```python
def get_gpu_memory()
```
- Returns current GPU memory usage in MB
- Uses `torch.cuda.memory_allocated()`
- Returns 0.0 if CUDA unavailable

#### Enhanced Sparsity Statistics
```python
def get_sparsity_stats(model, detailed=False)
```
- **New parameter**: `detailed` flag
- **Returns**: 
  - `float`: Overall sparsity (when `detailed=False`)
  - `dict`: Per-layer sparsity breakdown (when `detailed=True`)
- Simplifies layer names for readability
- Calculates zero gates (threshold: 1e-6)

#### Effective Parameter Calculation
```python
def compute_effective_params(model, adapter_config)
```
- Calculates trainable vs effective parameters
- For SoRA/SDoRA: Counts only non-zero gate parameters as effective
- Returns: `(trainable_params, effective_params)`

---

### 3. **Training Loop Enhancements**

#### Checkpoint Saving
- **Every 500 steps**: Mid-epoch checkpoints
  - Saves to: `checkpoint-epoch{X}-step{Y}/model.pt`
- **Every epoch end**: Epoch checkpoints
  - Saves to: `checkpoint-epoch{X}/model.pt`

#### Step-Level Logging (Every 100 Steps)
- Logs to `loss_log.csv`:
  - Training loss
  - Learning rate
  - GPU memory usage
  - Sparsity (for SoRA/SDoRA)
- Logs per-layer sparsity to `sparsity_log.csv` (SoRA/SDoRA only)

#### Real-Time Monitoring
- Progress bar now shows:
  - Loss
  - Learning rate
  - GPU memory (MB)
  - **Sparsity** (for SoRA/SDoRA only)

#### Epoch-Level Metrics
- Added to metrics dict:
  - `gpu_memory_mb`
  - `trainable_params`
  - `effective_params`
  - `sparsity` (for SoRA/SDoRA)

---

### 4. **Configuration Updates**

#### Default Model and Hyperparameters
- **Model**: Changed from `meta-llama/Meta-Llama-3-8B` → `Qwen/Qwen2.5-4B`
- **LoRA rank**: Changed from `8` → `16`
- **LoRA alpha**: Changed from `16` → `32`
- **Learning rate**: Changed from `1e-4` → `2e-4`

#### Adapter Configuration Storage
- Added `adapter_config` dict to store:
  - `adapter_name`
  - `r` (rank)
  - `alpha`
  - `dropout`
  - `target_modules`
- Used for parameter calculations and sparsity tracking

---

### 5. **Comment Translation**
All Chinese comments have been translated to English, including:
- Function docstrings
- Inline comments explaining logic
- Configuration notes
- Code section markers

This improves code accessibility for international collaboration and makes the codebase more maintainable.

---

## Output Files Generated

### For All Adapters (LoRA/DoRA/SoRA/SDoRA)
```
{output_dir}/
├── loss_log.csv                    # Step-level: [epoch, step, loss, lr, gpu_mb, sparsity]
├── training_history.json           # Epoch-level: Complete metrics history
├── training_log.json               # Alias for training_history.json
├── checkpoint-epoch1-step500/      # Mid-epoch checkpoints (every 500 steps)
│   └── model.pt
├── checkpoint-epoch1-step1000/
│   └── model.pt
├── checkpoint-epoch1/              # Epoch-end checkpoints
│   └── model.pt
├── checkpoint-epoch2/
│   └── model.pt
├── checkpoint-epoch3/
│   └── model.pt
├── model.pt                        # Final merged model
└── tokenizer files                 # Tokenizer config and vocab
```

### For SoRA/SDoRA Only (Additional)
```
{output_dir}/
└── sparsity_log.csv                # Per-layer sparsity: [epoch, step, overall, layer1, layer2, ...]
```

---

## Usage Examples

### Using Shell Script (Recommended for SSH disconnection tolerance)
```bash
# DoRA training (r=16, alpha=32, lr=1e-4)
nohup ./qwen2.5_4B_train_accelerate.sh dora 16 32 0 > dora_train.log 2>&1 &

# LoRA training (r=16, alpha=32, lr=3e-4)
nohup ./qwen2.5_4B_train_accelerate.sh lora 16 32 0 > lora_train.log 2>&1 &

# SoRA training with sparsity experiments (RECOMMENDED for research)
nohup ./qwen2.5_4B_train_accelerate.sh sora 16 32 0 0.1 > sora_lambda0.1_train.log 2>&1 &
nohup ./qwen2.5_4B_train_accelerate.sh sora 16 32 0 0.3 > sora_lambda0.3_train.log 2>&1 &
nohup ./qwen2.5_4B_train_accelerate.sh sora 16 32 0 0.5 > sora_lambda0.5_train.log 2>&1 &

# SDoRA with different sparsity levels
nohup ./qwen2.5_4B_train_accelerate.sh sdora 16 32 0 0.3 > sdora_lambda0.3_train.log 2>&1 &

# Check training progress
tail -f dora_train.log

# Check if still running
ps aux | grep finetune_accelerate
```

### Direct Python Execution (Not recommended for long training)
```bash
# LoRA training
python finetune_accelerate.py \
    --adapter_name lora \
    --output_dir outputs/lora_r16 \
    --learning_rate 3e-4 \
    --num_epochs 3

# DoRA training
python finetune_accelerate.py \
    --adapter_name dora \
    --output_dir outputs/dora_r16 \
    --learning_rate 1e-4 \
    --num_epochs 3
```

---

## Benefits

1. **Comprehensive Monitoring**: Track every aspect of training (loss, LR, GPU memory, sparsity)
2. **Fault Tolerance**: Regular checkpoints enable recovery from crashes
3. **Detailed Analysis**: Per-layer sparsity tracking for SoRA/SDoRA research
4. **Production Ready**: Multi-GPU support via Hugging Face Accelerate
5. **Research Insights**: Effective vs trainable parameter counting for efficiency studies
6. **Better Collaboration**: English comments improve code accessibility

---

## Sparsity Experiment Strategy

### Final Decision: Record + Test Multiple Levels

**For LoRA/DoRA**: No sparsity - just record trainable parameters

**For SoRA/SDoRA**: Test multiple sparsity levels to analyze parameter efficiency trade-offs

#### Recommended Sparsity Experiments:
```bash
# Low sparsity (more parameters kept)
nohup ./qwen2.5_4B_train_accelerate.sh sora 16 32 0 0.1 > sora_lambda0.1_train.log 2>&1 &

# Medium sparsity (balanced, default)
nohup ./qwen2.5_4B_train_accelerate.sh sora 16 32 0 0.3 > sora_lambda0.3_train.log 2>&1 &

# High sparsity (aggressive pruning)
nohup ./qwen2.5_4B_train_accelerate.sh sora 16 32 0 0.5 > sora_lambda0.5_train.log 2>&1 &
```

#### What Gets Recorded (Automatic):
1. **Step-level sparsity** in `loss_log.csv` (every 100 steps)
2. **Per-layer sparsity** in `sparsity_log.csv` (every 100 steps)
3. **Effective parameters** in `training_history.json` (per epoch)
4. **Overall sparsity** in `training_history.json` (per epoch)

#### Analysis:
- Compare accuracy vs sparsity across lambda values
- Identify which layers get pruned most aggressively
- Calculate parameter efficiency: accuracy / effective_params
- Plot sparsity evolution over training steps

---

## Monitoring Training Progress

### Check Training Status
```bash
# Check if training is still running
ps aux | grep finetune_accelerate

# Monitor real-time log output
tail -f dora_train.log

# Check last 100 lines of log
tail -n 100 dora_train.log

# Search for errors
grep -i "error" dora_train.log
```

### Find Total Training Time

#### Method 1: From Shell Script Log (dora_train.log)
```bash
# Look for start and end timestamps
grep "time:" dora_train.log

# Output example:
# Start time: Mon Nov 25 10:30:15 UTC 2025
# End time: Mon Nov 25 18:45:30 UTC 2025
# Total: ~8.25 hours
```

#### Method 2: From training_history.json
```bash
# Sum all epoch times
cd outputs/dora_r16
python3 << EOF
import json
with open('training_history.json') as f:
    data = json.load(f)
total_seconds = sum(data['epoch_time'])
hours = total_seconds / 3600
print(f"Total training time: {hours:.2f} hours ({total_seconds:.0f} seconds)")
EOF
```

#### Method 3: From Training Summary in Log
```bash
# Search for epoch summaries
grep "Epoch.*Summary" dora_train.log -A 10
```

### Monitor GPU Usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Check GPU memory from CSV
cd outputs/dora_r16
head -20 loss_log.csv  # See GPU memory in 5th column
```

---

## Technical Notes

### Sparsity Threshold
- Gates with `|value| < 1e-6` are considered zero/pruned
- Adjustable in `get_sparsity_stats()` function

### Checkpoint Frequency
- **Step checkpoints**: Every 500 steps (configurable via `save_steps`)
- **Log frequency**: Every 100 steps (configurable via `log_steps`)
- **Epoch checkpoints**: End of each epoch (always)

### Memory Tracking
- Uses `torch.cuda.memory_allocated()` for current memory usage
- Recorded in MB for readability
- Tracked at every training step

### Multi-GPU Behavior
- Only main process performs file I/O
- Validation loss aggregated across GPUs
- Progress bars disabled on non-main processes

---

## Migration from finetune.py

### Key Differences
| Feature | finetune.py | finetune_accelerate.py |
|---------|-------------|------------------------|
| Framework | Vanilla PyTorch | Hugging Face Accelerate |
| Multi-GPU | ❌ | ✅ |
| Step-level logging | ❌ | ✅ (CSV) |
| GPU memory tracking | ❌ | ✅ |
| Checkpoints | Final only | Every 500 steps + epochs |
| Sparsity tracking | Epoch-level | Step + per-layer |
| Default model | Llama-3-8B | Qwen2.5-4B |
| Default rank | 8 | 16 |

### Compatibility
- Both files use the same `utils.py` module
- Both support identical adapter types: LoRA, DoRA, SoRA, SDoRA
- Command-line arguments are compatible (finetune_accelerate.py has additional options)

---

## Future Enhancements (Potential)

- [ ] Tensorboard/Wandb integration
- [ ] Learning rate finder
- [ ] Automatic mixed precision (AMP) configuration
- [ ] FLOPS calculation
- [ ] Layer-wise learning rate scheduling
- [ ] Gradient statistics logging
- [ ] Resume from checkpoint functionality
- [ ] Early stopping based on validation loss

---

## Summary Tables

### Table 1: Code Changes Overview

| Category | Change Type | Description | Impact |
|----------|-------------|-------------|--------|
| **Logging** | New Feature | Added CSV logging for step-level metrics | Enables detailed training analysis |
| **Logging** | New Feature | Added per-layer sparsity tracking for SoRA/SDoRA | Research insights into pruning patterns |
| **Monitoring** | New Feature | GPU memory tracking at every step | Monitor resource usage in real-time |
| **Monitoring** | New Feature | Effective parameter calculation | Quantify parameter efficiency with sparsity |
| **Checkpointing** | New Feature | Save checkpoints every 500 steps | Fault tolerance and training recovery |
| **Checkpointing** | Enhancement | Save checkpoints at every epoch end | Track model evolution per epoch |
| **Configuration** | Update | Changed default model to Qwen2.5-4B | Faster training, smaller model |
| **Configuration** | Update | Increased default rank from 8 to 16 | Better capacity vs efficiency trade-off |
| **Configuration** | Update | Increased learning rate from 1e-4 to 2e-4 | Adjusted for smaller model |
| **Code Quality** | Translation | All Chinese comments converted to English | Improved code accessibility |
| **Visualization** | Enhancement | Real-time sparsity in progress bar | Immediate feedback for SoRA/SDoRA |

---

### Table 2: Generated Files and Their Purposes

| File Name | When Created | Update Frequency | Purpose | Used For |
|-----------|--------------|------------------|---------|----------|
| **loss_log.csv** | Training start | Every 100 steps | Step-level training metrics | Plotting loss curves, analyzing convergence, detecting instabilities |
| **sparsity_log.csv** | First SoRA/SDoRA log | Every 100 steps (SoRA/SDoRA only) | Per-layer sparsity evolution | Analyzing which layers get pruned, sparsity dynamics over time |
| **training_history.json** | Training start | Every epoch | Epoch-level summary metrics | Quick overview of training progress, validation performance |
| **checkpoint-epoch{X}-step{Y}/model.pt** | Every 500 steps | Every 500 steps | Mid-epoch model checkpoint | Resume training after crashes, analyze intermediate models |
| **checkpoint-epoch{X}/model.pt** | End of each epoch | Every epoch | End-of-epoch model checkpoint | Compare model quality across epochs, select best epoch |
| **model.pt** | Training completion | Once (final) | Final merged model weights | Inference, evaluation, deployment |
| **tokenizer files** | Training completion | Once (final) | Tokenizer configuration | Text processing for inference/evaluation |

---

### Table 3: File Usage Patterns

| Analysis Task | Primary Files | Secondary Files | Tools/Commands |
|---------------|---------------|-----------------|----------------|
| **Loss Convergence** | loss_log.csv | training_history.json | pandas, matplotlib: `plot loss vs step` |
| **Sparsity Analysis** | sparsity_log.csv | - | pandas, seaborn: `plot sparsity heatmap by layer` |
| **GPU Memory Profiling** | loss_log.csv | - | pandas: `plot gpu_memory vs step` |
| **Learning Rate Schedule** | loss_log.csv | - | pandas: `plot lr vs step` |
| **Epoch Comparison** | training_history.json | checkpoint-epoch*/model.pt | Load checkpoints and evaluate on test set |
| **Resume Training** | checkpoint-epoch*-step*/model.pt | training_history.json | Load state_dict and continue training |
| **Parameter Efficiency** | training_history.json | - | Compare trainable_params vs effective_params |
| **Overfitting Detection** | training_history.json | - | Compare train_loss vs val_loss trends |

---

### Table 4: Adapter-Specific Outputs

| Adapter Type | loss_log.csv | sparsity_log.csv | effective_params | Notes |
|--------------|--------------|------------------|------------------|-------|
| **LoRA** | ✅ | ❌ | = trainable_params | No sparsity tracking |
| **DoRA** | ✅ | ❌ | = trainable_params | No sparsity tracking |
| **SoRA** | ✅ (with sparsity column) | ✅ | < trainable_params | Full sparsity tracking |
| **SDoRA** | ✅ (with sparsity column) | ✅ | < trainable_params | Full sparsity tracking |

---

### Table 5: Configuration Changes

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `base_model` | meta-llama/Meta-Llama-3-8B | Qwen/Qwen2.5-4B | Faster training, smaller model size |
| `lora_r` | 8 | 16 | Better model capacity |
| `lora_alpha` | 16 | 32 | Maintains 2× scaling ratio |
| `learning_rate` | 1e-4 | 1e-4 (DoRA), 3e-4 (LoRA) | Adapter-specific tuning |
| `save_steps` | N/A | 500 | Add mid-epoch checkpointing |
| `log_steps` | N/A | 100 | Add step-level logging |

---

### Table 6: Quick File Reference

| Question | Answer | File to Check |
|----------|--------|---------------|
| Is training converging smoothly? | Check loss trends | loss_log.csv |
| Which layers are being pruned? | Check per-layer sparsity | sparsity_log.csv |
| Is the model overfitting? | Compare train vs val loss | training_history.json |
| How much GPU memory is used? | Check gpu_memory_mb | loss_log.csv |
| What's the effective parameter count? | Check effective_params | training_history.json |
| Can I resume from a crash? | Load checkpoint | checkpoint-epoch*-step*/model.pt |
| Which epoch performed best? | Compare val_loss | training_history.json |
| How long did training take? | Check start/end in log or sum epoch_time | dora_train.log or training_history.json |

---

### Table 7: Training Commands Reference

| Adapter | Command | Learning Rate | Sparsity | Notes |
|---------|---------|---------------|----------|-------|
| **LoRA** | `nohup ./qwen2.5_4B_train_accelerate.sh lora 16 32 0 > lora_train.log 2>&1 &` | 3e-4 | N/A | Baseline method |
| **DoRA** | `nohup ./qwen2.5_4B_train_accelerate.sh dora 16 32 0 > dora_train.log 2>&1 &` | 1e-4 | N/A | Weight-decomposed |
| **SoRA (λ=0.1)** | `nohup ./qwen2.5_4B_train_accelerate.sh sora 16 32 0 0.1 > sora_l01_train.log 2>&1 &` | 1e-4 | 0.1 | Low sparsity |
| **SoRA (λ=0.3)** | `nohup ./qwen2.5_4B_train_accelerate.sh sora 16 32 0 0.3 > sora_l03_train.log 2>&1 &` | 1e-4 | 0.3 | Medium (default) |
| **SoRA (λ=0.5)** | `nohup ./qwen2.5_4B_train_accelerate.sh sora 16 32 0 0.5 > sora_l05_train.log 2>&1 &` | 1e-4 | 0.5 | High sparsity |
| **SDoRA** | `nohup ./qwen2.5_4B_train_accelerate.sh sdora 16 32 0 0.3 > sdora_train.log 2>&1 &` | 1e-4 | 0.3 | Sparse DoRA |

---

### Table 8: Where to Find Training Time

| Location | What to Look For | Command |
|----------|------------------|---------|
| **Shell log (dora_train.log)** | Start/End timestamps | `grep "time:" dora_train.log` |
| **training_history.json** | Sum of epoch_time array | `python -c "import json; print(sum(json.load(open('outputs/dora_r16/training_history.json'))['epoch_time'])/3600, 'hours')"` |
| **Training summary in log** | Final summary section | `grep "Training Summary" dora_train.log -A 5` |
| **Each epoch in log** | Individual epoch durations | `grep "epoch_time" dora_train.log` |

---

## Quick Start Guide

### Step 1: Prepare Environment
```bash
cd /home/ubuntu/LLM-inference/liangzhao-project/ruijie
conda activate vlm
```

### Step 2: Estimate Training Time (BEFORE starting)
```bash
python3 estimate_training_time.py
```
This shows:
- Time per experiment
- Total estimated time (~29 hours for all 6 experiments)
- Expected completion date/time

### Step 3: Run ALL Experiments in One Command
```bash
# Recommended: Run all experiments sequentially in one log
nohup ./run_all_experiments.sh > all_experiments.log 2>&1 &
```

**Why use this approach?**
- ✅ All results in one log file for easy comparison
- ✅ Time estimates shown upfront
- ✅ Each experiment's duration recorded
- ✅ No need to manually start each training
- ✅ Automatic error handling

### Step 4: Monitor Progress
```bash
# Watch unified log in real-time
tail -f all_experiments.log

# Check which experiment is running
grep "EXPERIMENT" all_experiments.log | tail -1

# Check if still running
ps aux | grep finetune_accelerate

# See all time estimates
grep "Duration:" all_experiments.log

# Exit monitoring: Ctrl+C (training continues in background)
```

### Step 5: After Training Completes
```bash
# Check total time for all experiments
grep "TOTAL" all_experiments.log | tail -1

# View all results
cd outputs
ls -d */  # List all experiment outputs

# Compare results
python3 analyze_results.py  # (create this script to compare metrics)

# Check individual experiment times
grep "Duration:" all_experiments.log
```

### Step 5: Disconnect SSH Safely
```bash
# Training runs in background with nohup - safe to disconnect
exit
```

---

## Training Commands Quick Reference

### RECOMMENDED: Run All Experiments at Once

#### Step 1: Estimate Training Time
```bash
python3 estimate_training_time.py
```
Output example:
```
LoRA                : ~     4h 1m
DoRA                : ~     5h 1m
SoRA (λ=0.1)        : ~     5h 1m
SoRA (λ=0.3)        : ~     5h 1m
SoRA (λ=0.5)        : ~     5h 1m
SDoRA (λ=0.3)       : ~     5h 1m
-------------------------------
TOTAL               : ~  1d 5h 9m
```

#### Step 2: Start Unified Training (All 6 Experiments)
```bash
nohup ./run_all_experiments.sh > all_experiments.log 2>&1 &
```

**This single command will:**
- ✅ Train all 4 adapters (LoRA, DoRA, SoRA×3, SDoRA)
- ✅ Record everything in ONE log file
- ✅ Show time estimates BEFORE starting
- ✅ Report duration for EACH experiment
- ✅ Run sequentially (no GPU conflicts)
- ✅ Continue after SSH disconnect

#### Step 3: Monitor Progress
```bash
# Watch real-time progress
tail -f all_experiments.log

# Check which experiment is running
grep "EXPERIMENT" all_experiments.log | tail -1

# See completion estimates
grep "Estimated completion" all_experiments.log

# Check if still running
ps aux | grep finetune_accelerate

# Monitor GPU usage
watch -n 1 nvidia-smi
```

---

### Alternative: Run Individual Experiments

#### Individual Training Commands
```bash
# DoRA (r=16, alpha=32, lr=1e-4)
nohup ./qwen2.5_4B_train_accelerate.sh dora 16 32 0 > dora_train.log 2>&1 &

# LoRA (r=16, alpha=32, lr=3e-4)
nohup ./qwen2.5_4B_train_accelerate.sh lora 16 32 0 > lora_train.log 2>&1 &

# SoRA - Sparsity Experiments
nohup ./qwen2.5_4B_train_accelerate.sh sora 16 32 0 0.1 > sora_lambda0.1_train.log 2>&1 &
nohup ./qwen2.5_4B_train_accelerate.sh sora 16 32 0 0.3 > sora_lambda0.3_train.log 2>&1 &
nohup ./qwen2.5_4B_train_accelerate.sh sora 16 32 0 0.5 > sora_lambda0.5_train.log 2>&1 &

# SDoRA (r=16, alpha=32, lr=1e-4, lambda=0.3)
nohup ./qwen2.5_4B_train_accelerate.sh sdora 16 32 0 0.3 > sdora_train.log 2>&1 &
```

#### Monitoring Individual Training
```bash
# Watch real-time log
tail -f dora_train.log
# Press Ctrl+C to exit (training continues)

# Check if running
ps aux | grep finetune_accelerate

# Check GPU usage
nvidia-smi
```

---

## Finding Training Time

### Method 1: From Log File (Shell Script)
```bash
grep "time:" dora_train.log
```
Output example:
```
Start time: Mon Nov 25 10:30:15 UTC 2025
End time: Mon Nov 25 18:45:30 UTC 2025
```
Calculate: 18:45:30 - 10:30:15 = ~8.25 hours

### Method 2: From training_history.json
```bash
cd outputs/dora_r16
python3 -c "import json; data=json.load(open('training_history.json')); print(f'{sum(data[\"epoch_time\"])/3600:.2f} hours')"
```

### Method 3: Check Each Epoch Time
```bash
grep "epoch_time" dora_train.log
```

### Method 4: From Unified Log (All Experiments)
```bash
# See individual experiment durations
grep "Duration:" all_experiments.log

# See total time
grep "TOTAL" all_experiments.log | tail -1
```

---

## Output Files Structure

### All Adapters Generate:
```
outputs/{adapter}_r16/
├── loss_log.csv              # Step-level metrics (every 100 steps)
├── training_history.json     # Epoch summaries
├── checkpoint-epoch1/        # Epoch checkpoints
├── checkpoint-epoch2/
├── checkpoint-epoch3/
├── checkpoint-epoch*-step*/  # Mid-epoch checkpoints (every 500 steps)
└── model.pt                  # Final model
```

### SoRA/SDoRA Additionally Generate:
```
outputs/sora_r16_lambda0.3/
└── sparsity_log.csv          # Per-layer sparsity (every 100 steps)
```

---

## Troubleshooting

### Training Not Starting
```bash
# Check if process exists
ps aux | grep finetune_accelerate

# Check log for errors
tail -50 dora_train.log
grep -i "error" dora_train.log
```

### Out of Memory
- Reduce `--micro_batch_size` in shell script (default: 8)
- Reduce `--batch_size` (default: 32)
- Try gradient checkpointing

### Training Stopped Unexpectedly
```bash
# Check last lines of log
tail -100 dora_train.log

# Look for error messages
grep -i "error\|killed\|exception" dora_train.log

# Check system logs
dmesg | tail -50
```

### Resume from Checkpoint
```bash
# Modify finetune_accelerate.py to add --resume_from_checkpoint flag
# Then point to: checkpoint-epoch*-step*/model.pt
```

---

## Experiment Tracking Table

| Run | Adapter | r | alpha | lr | Lambda | Start Time | End Time | Total Hours | Output Dir |
|-----|---------|---|-------|----| -------|------------|----------|-------------|------------|
| 1 | LoRA | 16 | 32 | 3e-4 | N/A | | | ~4h | outputs/lora_r16 |
| 2 | DoRA | 16 | 32 | 1e-4 | N/A | | | ~5h | outputs/dora_r16 |
| 3 | SoRA | 16 | 32 | 1e-4 | 0.1 | | | ~5h | outputs/sora_r16_lambda0.1 |
| 4 | SoRA | 16 | 32 | 1e-4 | 0.3 | | | ~5h | outputs/sora_r16_lambda0.3 |
| 5 | SoRA | 16 | 32 | 1e-4 | 0.5 | | | ~5h | outputs/sora_r16_lambda0.5 |
| 6 | SDoRA | 16 | 32 | 1e-4 | 0.3 | | | ~5h | outputs/sdora_r16_lambda0.3 |
| **TOTAL** | | | | | | | | **~29h** | |

---

## Sparsity Experiment Guide

### Why Test Multiple Lambda Values?
To analyze parameter efficiency vs accuracy trade-offs:

| Lambda | Sparsity Level | Expected Behavior | Use Case |
|--------|----------------|-------------------|----------|
| 0.1 | Low | More parameters kept, higher accuracy | When accuracy is critical |
| 0.3 | Medium | Balanced pruning (default) | Best balance |
| 0.5 | High | Aggressive pruning, fewer params | When efficiency is critical |

### What Gets Recorded Automatically
- ✅ Step-level sparsity (every 100 steps) → `loss_log.csv`
- ✅ Per-layer sparsity breakdown → `sparsity_log.csv`
- ✅ Effective parameter counts → `training_history.json`
- ✅ Overall sparsity per epoch → `training_history.json`

### Analysis Tasks
After training completes:
1. Compare accuracy vs sparsity across lambda values
2. Identify which layers get pruned most aggressively
3. Calculate parameter efficiency: accuracy / effective_params
4. Plot sparsity evolution over training steps
5. Analyze layer-wise pruning patterns

---

## Quick Commands Cheatsheet

```bash
# =========================
# BEFORE TRAINING
# =========================

# Estimate time
python3 estimate_training_time.py

# Start all experiments
nohup ./run_all_experiments.sh > all_experiments.log 2>&1 &

# =========================
# DURING TRAINING
# =========================

# Monitor progress
tail -f all_experiments.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check which experiment is running
grep "EXPERIMENT" all_experiments.log | tail -1

# Check if process is running
ps aux | grep finetune_accelerate

# =========================
# AFTER TRAINING
# =========================

# Find total time
grep "TOTAL" all_experiments.log | tail -1

# Check individual times
grep "Duration:" all_experiments.log

# View results
cd outputs && ls -d */

# Check validation loss
cd outputs/dora_r16
grep "val_loss" training_history.json
```

---

**Last Updated**: November 25, 2025  
**Author**: AI Assistant  
**Version**: 1.3
