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
- **Independence**: It does not rely on Trainer callbacks or external schedules. (Note: Lambda scheduling logic can be easily ported into this class if needed).

---

## 4. Parameter Splitting & Optimizer Construction

### Original Implementation (`src/util.py` + `trainer.py`)
- **Logic**: `trainer.py` calls utility functions like `compute_trainable_sparse_param` during optimizer construction.
- **Process**: It identifies `gate` parameters by name, adds them to the gate optimizer, and assigns the remaining parameters to a standard `AdamW` optimizer. This logic is scattered across `trainer.py` and `util.py`.

### `sora_minimal.py` Implementation (`split_sora_params` + `build_sora_optimizers`)
- **Logic**: Provides an explicit `split_sora_params(model)` function.
- **Process**: It iterates through `model.named_parameters()`. Any parameter ending with `"gate"` (and requiring gradients) is classified as a gate parameter.
- **Usage**: `build_sora_optimizers` explicitly returns a tuple: `(gate_optimizer, regular_optimizer)`. The user manually calls `gate_opt.step()` and `regular_opt.step()` in their custom training loop. This decouples the optimization logic from the HuggingFace `Trainer`, making it easier to transplant into custom scripts.
