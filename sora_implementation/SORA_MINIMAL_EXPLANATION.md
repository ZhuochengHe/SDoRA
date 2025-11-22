# SoRA Implementation vs. Original Implementation

This document explains the structural and architectural differences between the original SoRA implementation (based on `OpenDelta`) and the simplified `sora.py` implementation.

## 0. Inheritance Structure

### Current Implementation (`sora.py`)
- **Structure**: `SoRA_Linear` inherits from `LoRA_Linear` (from `lora_implementation`), which provides the base low-rank adaptation functionality.
- **Extension**: SoRA only adds the `gate` parameter to the inherited LoRA structure.
- **Benefits**: 
  - Code reuse and consistency with LoRA/DoRA implementations
  - All three methods (LoRA, DoRA, SoRA) share the same base class
  - Easier to compare and swap between methods
- **Parameter Structure**: Uses `nn.Linear` modules for `lora_A` and `lora_B` (inherited from LoRA), not `nn.Parameter`

### Comparison Table
```
LoRA_Linear (base class)
├── Dora_Linear    (adds magnitude parameter)
└── SoRA_Linear    (adds gate parameter)
```

## 1. `SoRA_Linear` vs. Original (`LowRankLinear` + `LoraModel`)

### Original Implementation
- **Structure**: `LoraModel` inherits from `DeltaBase` and relies on the `OpenDelta` library to scan the model structure and locate linear layers (like `attn.q`, `attn.v`).
- **Mechanism**: It appends `LowRankLinear` as a "parallel branch" to the existing layers.
- **Responsibility**: The `LowRankLinear` class only knows how to perform the computation $(xA^T \cdot \text{gate} \cdot B^T)$ but does not know where it belongs in the model hierarchy.

### `sora.py` Implementation
- **Structure**: `SoRA_Linear` inherits from `LoRA_Linear` and accepts an existing `nn.Linear` layer instance.
- **Mechanism**: It wraps the original layer, freezes the original weights, and adds the `gate` parameter on top of the inherited LoRA functionality.
- **Responsibility**: It handles both the "insertion position" (by directly replacing the `nn.Linear` instance during wrapping) and the forward computation. This centralizes the logic into a single class, removing the need for `OpenDelta` or `DeltaBase`.
- **Naming Convention**: Uses consistent naming with LoRA/DoRA (`lora_alpha`, `lora_dropout`, `self.linear`, `self.r`)

---

## 2. `wrap_linears` vs. Original (`LoraModel.add_all_delta_to_backbone`)

### Original Implementation
- **Configuration**: Requires listing `modified_modules` (e.g., `attn.q`) in a JSON file or command-line arguments.
- **Process**: `LoraModel` uses `OpenDelta`'s name matching mechanism to find modules, then calls `update_module` $\to$ `new_module_like` $\to$ `insert_parallel_module`.

### `sora.py` Implementation
- **Configuration**: Accepts a simple list of strings (e.g., `["encoder.layer.0.attention.output.dense"]`).
- **Process**: It iterates through `model.named_modules()`, finds the target parent nodes, and performs an in-place replacement with `SoRA_Linear`. It eliminates the need for external configuration files, complex naming rules, or address parsing logic.

---

## 3. `SoRAOptimizer` vs. Original (`src/sparse_optimizer*.py` + `trainer.py`)

### Original Implementation
- **Definition**: `SparseAdamW` is defined in `src/sparse_optimizer*.py` but is not instantiated directly by the user.
- **Integration**: `trainer.py` (inheriting from HuggingFace `Trainer`) separates the `gate` parameters inside `create_optimizer` and passes them to `SparseAdamW`. It also optionally handles lambda scheduling via `_build_lambda_list` and `step_lambda`.

### `sora.py` Implementation
- **Definition**: `SoRAOptimizer` inherits directly from `torch.optim.AdamW`.
- **Integration**: It executes the soft-thresholding operation (threshold fixed at `sparse_lambda * lr`) at the end of its own `step()` method.
- **Independence**: It does not rely on Trainer callbacks or external schedules.
- **Correctness**: It strictly enforces `weight_decay=0.0` for gate parameters to solve the pure Lasso ($L_1$) problem, avoiding the Elastic Net ($L_1 + L_2$) formulation that would occur if weight decay were applied.

---

## 4. Parameter Splitting & Optimizer Construction

### Original Implementation (`src/util.py` + `trainer.py`)
- **Logic**: `trainer.py` calls utility functions like `compute_trainable_sparse_param` during optimizer construction.
- **Process**: It identifies `gate` parameters by name, adds them to the gate optimizer, and assigns the remaining parameters to a standard `AdamW` optimizer. This logic is scattered across `trainer.py` and `util.py`.

### `sora.py` Implementation (`split_sora_params` + `build_sora_optimizers`)
- **Logic**: Provides an explicit `split_sora_params(model)` function.
- **Process**: It iterates through `model.named_parameters()`. Any parameter ending with `"gate"` (and requiring gradients) is classified as a gate parameter.
- **Usage**: `build_sora_optimizers` explicitly returns a tuple: `(gate_optimizer, regular_optimizer)`. The user manually calls `gate_opt.step()` and `regular_opt.step()` in their custom training loop. This decouples the optimization logic from the HuggingFace `Trainer`, making it easier to transplant into custom scripts.

---

## 5. Pruning and Merging (New Features)

### Pruning (`prune_sora_model`)
- **Purpose**: To physically remove the dimensions (ranks) that have been zeroed out by the gate during training.
- **Mechanism**: The `prune()` method in `SoRA_Linear` identifies indices where the gate is zero. It then creates new, smaller `lora_A` and `lora_B` nn.Linear modules containing only the active ranks. The gate values are merged into the `lora_B` weights, and the gate parameter itself is removed.
- **Implementation Note**: Works with inherited `nn.Linear` structure, creating new modules with reduced dimensions while preserving the LoRA weight shapes `(r, in)` and `(out, r)`.
- **Benefit**: Reduces the parameter count and memory footprint of the adapter without affecting the output.

### Merging (`merge_sora_model`)
- **Purpose**: To achieve zero-overhead inference by merging the adapter weights into the base model weights.
- **Mechanism**: The `merge()` method computes the effective weight update $\Delta W = (B \cdot \text{diag}(g) \cdot A) \times \text{scaling}$ and adds it directly to the frozen `linear.weight` (inherited base layer). The entire LoRA branch is then discarded, and the `merged` flag is set to `True`.
- **Implementation Note**: Works with `lora_A.weight` and `lora_B.weight` from the inherited `nn.Linear` modules.
- **Benefit**: The model becomes a standard dense model again, with no additional inference latency compared to the original base model.

