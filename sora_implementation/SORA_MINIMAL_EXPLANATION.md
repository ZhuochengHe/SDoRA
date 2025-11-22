# SoRA Minimal Implementation vs. Original Implementation

This document explains the structural and architectural differences between the original SoRA implementation (based on `OpenDelta`) and the simplified `sora_minimal.py` implementation.

## 1. `SoRALinear` vs. Original (`LowRankLinear` + `LoraModel`)

### Original Implementation
- **Structure**: `LoraModel` inherits from `DeltaBase` and relies on the `OpenDelta` library to scan the model structure and locate linear layers (like `attn.q`, `attn.v`).
- **Mechanism**: It appends `LowRankLinear` as a "parallel branch" to the existing layers.
- **Responsibility**: The `LowRankLinear` class only knows how to perform the computation $(xA^T \cdot \text{gate} \cdot B^T)$ but does not know where it belongs in the model hierarchy.

### `sora_minimal.py` Implementation
- **Structure**: `SoRALinear` directly accepts an existing `nn.Linear` layer instance.
- **Mechanism**: It wraps the original layer, freezes the original weights, and internally maintains the `A`, `B`, and `gate` parameters.
- **Responsibility**: It handles both the "insertion position" (by directly replacing the `nn.Linear` instance during wrapping) and the forward computation. This centralizes the logic into a single class, removing the need for `OpenDelta` or `DeltaBase`.

---

## 2. `wrap_linears` vs. Original (`LoraModel.add_all_delta_to_backbone`)

### Original Implementation
- **Configuration**: Requires listing `modified_modules` (e.g., `attn.q`) in a JSON file or command-line arguments.
- **Process**: `LoraModel` uses `OpenDelta`'s name matching mechanism to find modules, then calls `update_module` $\to$ `new_module_like` $\to$ `insert_parallel_module`.

### `sora_minimal.py` Implementation
- **Configuration**: Accepts a simple list of strings (e.g., `["encoder.layer.0.attention.output.dense"]`).
- **Process**: It iterates through `model.named_modules()`, finds the target parent nodes, and performs an in-place replacement with `SoRALinear`. It eliminates the need for external configuration files, complex naming rules, or address parsing logic.

---

## 3. `SoRAOptimizer` vs. Original (`src/sparse_optimizer*.py` + `trainer.py`)

### Original Implementation
- **Definition**: `SparseAdamW` is defined in `src/sparse_optimizer*.py` but is not instantiated directly by the user.
- **Integration**: `trainer.py` (inheriting from HuggingFace `Trainer`) separates the `gate` parameters inside `create_optimizer` and passes them to `SparseAdamW`. It also optionally handles lambda scheduling via `_build_lambda_list` and `step_lambda`.

### `sora_minimal.py` Implementation
- **Definition**: `SoRAOptimizer` inherits directly from `torch.optim.AdamW`.
- **Integration**: It executes the soft-thresholding operation (threshold fixed at `sparse_lambda * lr`) at the end of its own `step()` method.
- **Independence**: It does not rely on Trainer callbacks or external schedules.
- **Correctness**: It strictly enforces `weight_decay=0.0` for gate parameters to solve the pure Lasso ($L_1$) problem, avoiding the Elastic Net ($L_1 + L_2$) formulation that would occur if weight decay were applied.

---

## 4. Parameter Splitting & Optimizer Construction

### Original Implementation (`src/util.py` + `trainer.py`)
- **Logic**: `trainer.py` calls utility functions like `compute_trainable_sparse_param` during optimizer construction.
- **Process**: It identifies `gate` parameters by name, adds them to the gate optimizer, and assigns the remaining parameters to a standard `AdamW` optimizer. This logic is scattered across `trainer.py` and `util.py`.

### `sora_minimal.py` Implementation (`split_sora_params` + `build_sora_optimizers`)
- **Logic**: Provides an explicit `split_sora_params(model)` function.
- **Process**: It iterates through `model.named_parameters()`. Any parameter ending with `"gate"` (and requiring gradients) is classified as a gate parameter.
- **Usage**: `build_sora_optimizers` explicitly returns a tuple: `(gate_optimizer, regular_optimizer)`. The user manually calls `gate_opt.step()` and `regular_opt.step()` in their custom training loop. This decouples the optimization logic from the HuggingFace `Trainer`, making it easier to transplant into custom scripts.

---

## 5. Pruning and Merging (New Features)

### Pruning (`prune_sora_model`)
- **Purpose**: To physically remove the dimensions (ranks) that have been zeroed out by the gate during training.
- **Mechanism**: The `prune()` method in `SoRALinear` identifies indices where the gate is zero. It then creates new, smaller `lora_A` and `lora_B` matrices containing only the active ranks. The gate parameter is then removed, and the scaling is baked into the weights.
- **Benefit**: Reduces the parameter count and memory footprint of the adapter without affecting the output.

### Merging (`merge_sora_model`)
- **Purpose**: To achieve zero-overhead inference by merging the adapter weights into the base model weights.
- **Mechanism**: The `merge()` method computes the effective weight update $\Delta W = (B \cdot \text{diag}(g) \cdot A) \times \text{scaling}$ and adds it directly to the frozen `base_layer.weight`. The entire LoRA branch is then discarded.
- **Benefit**: The model becomes a standard dense model again, with no additional inference latency compared to the original base model.

