import os
import math
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------

def extract_lora_ab(state, layer_keyword=''):
    A = {}
    B = {}
    m_w = {}
    for name, p in state.items():
        if layer_keyword and layer_keyword not in name:
            continue
        if 'lora_A' in name:
            key = name.replace('.lora_A.weight','')
            A[key] = p
        elif 'lora_B' in name:
            key = name.replace('.lora_B.weight','')
            B[key] = p
        elif 'weight_m_wdecomp.weight' in name:
            key = name.replace('.weight_m_wdecomp.weight','')
            m_w[key] = p
    return A, B, m_w

def compute_delta_d_m(A, B, m_w, base_W=None, method='dora', target_module='q_proj'):
    delta = {}
    if method == 'full':
        W1_state = A    
        W0_state = base_W
        for k_full, W1 in W1_state.items():
            if "bias" in k_full.lower() or target_module not in k_full or W1.dim() != 2:
                continue
            k_unified = k_full.replace('.linear.weight', '').replace('.weight', '')
            if k_unified not in W0_state:
                continue
            W0 = W0_state[k_unified]
            D0 = W0 / (W0.norm(dim=1, keepdim=True) + 1e-6)
            D1 = W1 / (W1.norm(dim=1, keepdim=True) + 1e-6)
            delta_D_value = (D1 - D0).norm(dim=1).mean().item()
            M0 = W0.norm(dim=1)
            M1 = W1.norm(dim=1)
            delta_M_value = (M1 - M0).mean().item()
            delta[k_unified] = (delta_D_value, delta_M_value)
        return delta
    base_D = {}
    if base_W:
        for k_full, W0 in base_W.items():
            if "bias" in k_full.lower():
                continue

            # W0 的单位方向 D0 = W0 / ||W0||_2 (按行)
            k = k_full.replace('.weight', '')
            W0_norm = W0.norm(dim=1, keepdim=True)
            # D0 形状: (out_features, in_features)
            base_D[k] = W0 / (W0_norm + 1e-6)
    for k in A:
        if k in B:
            deltaW = B[k] @ A[k]  # LoRA增量矩阵
            # 1. 计算更新后的权重向量 V = W0 + deltaW
            W0 = base_W[k] if base_W and k in base_W else torch.zeros_like(deltaW)
            M0 = W0.norm(dim=1, keepdim=True)
            new_weight_v = W0 + deltaW
            # 2. 计算当前单位方向 D = V / ||V|| (按行)
            current_D = new_weight_v / (new_weight_v.norm(dim=1, keepdim=True) + 1e-6)
            # --- Delta M 的计算 (幅度) ---
            if method in ['dora','sdora']:
                # DoRA/SDoRA: 使用可训练的幅度 M 的平均值
                magnitude_M = m_w[k]
                delta_M_value = (magnitude_M - M0).mean().item()
            else: # LoRA/SoRA: 使用增量 DeltaW 的幅度 ||DeltaW|| 的平均值
                # 注意：在 LoRA/SoRA 的 W = W0 + DeltaW 框架下，Delta M 应该反映幅度变化，
                # 但这里我们沿用原论文分析，使用 DeltaW 的幅度。
                M1 = new_weight_v.norm(dim=1, keepdim=True)
                delta_M_value = (M1 - M0).mean().item()
            # --- Delta D 的计算 (方向变化量) ---
            if k in base_D:
                row_diff_norm = (current_D - base_D[k]).norm(dim=1)
                delta_D_value = row_diff_norm.mean().item()
            else:
                print(f"Warning: Base direction D0 for layer {k} not found. Using ||D||_F.")
                delta_D_value = current_D.norm().item()
            delta[k] = (delta_D_value, delta_M_value)

    return delta


def plot_updates(all_results, adapter, save_path):
    plt.figure(figsize=(10, 6)) 
    
    colors = plt.cm.get_cmap('Set1') 
    
    markers = ['s', '*', 'd'] 
    
    step_labels = ['Epoch 1', 'Epoch 2', 'Epoch 3']
    
    layer_names = list(all_results.keys())

    all_dirs = []
    all_mags = []

    for i, layer_name in enumerate(layer_names):
        entries = all_results[layer_name]
        
        if i > 7:
            break
        for j, (d, m) in enumerate(entries):
            marker_style = markers[j % len(markers)]
            
            facecolor = colors(i)
            edgecolor = colors(i)
            linewidth = 0.5
            
            plt.scatter(
                d, 
                m, 
                color=facecolor, 
                marker=marker_style, 
                s=100, 
                alpha=0.9,
                edgecolor=edgecolor,
                linewidth=linewidth
            )
            
            all_dirs.append(d)
            all_mags.append(m)

    if all_dirs and all_mags:
        all_dirs_np = np.array(all_dirs)
        all_mags_np = np.array(all_mags)
        
        slope, intercept = np.polyfit(all_dirs_np, all_mags_np, 1)
        x_fit = np.linspace(all_dirs_np.min() * 0.95, all_dirs_np.max() * 1.05, 100)
        y_fit = slope * x_fit + intercept
        
        plt.plot(
            x_fit, 
            y_fit, 
            color='dodgerblue', 
            linestyle='-', 
            linewidth=2,
            zorder=0 
        )
        slope_str = f"{slope:.3e}"

        if intercept >= 0:
            intercept_str = f"+ {intercept:.3e}"
        else:
            intercept_str = f"- {abs(intercept):.3e}"

        eq_text = f"$y = {slope_str} \\, x \\, {intercept_str}$"

    plt.title(adapter.upper(), fontsize=20, y=1.05) 
    plt.xlabel('$\\Delta D$', fontsize=20) 
    plt.ylabel('$\\Delta M$', fontsize=20, rotation=0, labelpad=30) 
    
    plt.grid(False) 
    ax = plt.gca()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    layer_handles = []
    layer_labels = [f'layer {i+1}' for i in range(8)] 
    
    for i in range(8):
        layer_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=colors(i), markersize=10, 
                                        label=layer_labels[i]))

    step_handles = []
    for j in range(len(markers)):
        step_handles.append(plt.Line2D([0], [0], marker=markers[j], color='w', 
                                        markerfacecolor='black', markersize=10, 
                                        label=step_labels[j]))
    
    legend1 = plt.legend(layer_handles, layer_labels, 
                         loc='upper left', bbox_to_anchor=(-0.4, 0.8),
                         frameon=False, numpoints=1)

    plt.legend(step_handles, step_labels, 
               loc='upper right', bbox_to_anchor=(1.4, 0.8),
               frameon=False, numpoints=1)

    plt.gca().add_artist(legend1) 
    
    plt.tight_layout(rect=[0.1, 0, 1.1, 1]) 
    plt.savefig(save_path)
    plt.close()
    
    print(f'Saved plot to {save_path}')

def print_state_dict_keys(path_or_dict, source_name, target_modules=['q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj']):
    """
    加载并打印状态字典中与目标模块相关的键名。
    
    Args:
        path_or_dict: 模型的路径（字符串）或已加载的状态字典（dict）。
        source_name: 打印输出的来源名称（例如: "BASE_MODEL"）。
        target_modules: 要过滤并显示的模块名称列表。
    """
    print(f"--- 🔑 {source_name} STATE DICT KEYS ---")
    
    state_dict = {}
    if isinstance(path_or_dict, str):
        print(f"Loading state dict from path: {path_or_dict}")
        try:
            # 尝试加载 LoRA Checkpoint
            state_dict = torch.load(path_or_dict, map_location='cpu')
        except:
            # 尝试加载 Hugging Face 模型（用于基座）
            try:
                # 假设您的模型是 LLM
                model = AutoModelForCausalLM.from_pretrained(path_or_dict, low_cpu_mem_usage=True) 
                state_dict = model.state_dict()
            except Exception as e:
                print(f"ERROR: Could not load model/checkpoint from {path_or_dict}. Error: {e}")
                return
    elif isinstance(path_or_dict, dict):
        state_dict = path_or_dict

    # 分组存储键名，以便清晰展示
    key_groups = defaultdict(list)
    
    for k in state_dict.keys():
        if any(module in k for module in target_modules):
            # 将键名按模块类型和维度分组
            key_suffix = ''
            if k.endswith('.weight'):
                key_suffix = '.weight'
            elif k.endswith('.bias'):
                key_suffix = '.bias'
            elif 'lora_A' in k:
                key_suffix = '.lora_A'
            elif 'lora_B' in k:
                key_suffix = '.lora_B'
            elif 'weight_m_w' in k: # DoRA/SoRA幅度
                key_suffix = '.weight_m_w'
            
            # 使用一个示例层来展示路径前缀
            if 'layers.0.' in k:
                key_groups[key_suffix].append(k)
    
    if not key_groups:
        print("No relevant keys found in the state dict.")
        return

    # 打印分组结果
    print("\n[ Example Keys for layers.0 ]")
    for suffix, keys in key_groups.items():
        print(f"Found {len(keys)} keys ending with {suffix}:")
        # 只打印前几个示例
        for i, key in enumerate(keys[:5]): 
            print(f"  {i+1}. {key}")
        if len(keys) > 5:
            print("  ... (more keys hidden) ...")
    
    print("----------------------------------------\n")

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['full', 'lora','sora','dora','sdora'])
    parser.add_argument('r', type=int)
    parser.add_argument('lam', type=float, default=0.1)
    parser.add_argument('lr', type=str)
    parser.add_argument('alpha', type=int)
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-3B')
    args = parser.parse_args()

    print(f'Loading base model {args.base_model} for epoch0 reference...')
    target_module = 'q_proj'
    base_model = AutoModel.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    base_state = {}
    for k_full, v in base_model.state_dict().items():
        if target_module in k_full and k_full.endswith('.weight'):
            
            k_module = k_full.replace('.weight', '') # 变为 'layers.X.q_proj'
            
            if not k_module.startswith('model.'):
                k_unified = 'model.' + k_module # 变为 'model.layers.X.q_proj'
            else:
                k_unified = k_module
                
            base_state[k_unified] = v.clone().cpu()

    folder = f"outputs/Qwen2.5-3B_{args.method}_r{args.r}_lr{args.lr}_alpha{args.alpha}" if args.method in ['lora','dora'] else f"outputs/Qwen2.5-3B_{args.method}_r{args.r}_lambda{args.lam}_lr{args.lr}_alpha{args.alpha}"
    if args.method == 'full':
        folder = f"outputs/Qwen2.5-3B_full-tuning"
    checkpoints = sorted([d for d in os.listdir(folder) if d.startswith('checkpoint-epoch') and not 'step' in d])
    print(checkpoints)

    all_results = {}

    for ck in checkpoints:
        ck_path = os.path.join(folder, ck, 'model.pt')
        state = torch.load(ck_path, map_location='cpu')
        if args.method == 'full':
            delta = compute_delta_d_m(state, None, None, base_W=base_state, method=args.method, target_module=target_module)
        else:
            A,B,m_w = extract_lora_ab(state, target_module)
            delta = compute_delta_d_m(A,B,m_w, base_W=base_state, method=args.method, target_module=target_module)

        for k, (d,m) in delta.items():
            all_results.setdefault(k,[]).append((d,m))

    save_folder = os.path.join(os.getcwd(), 'exp_deltaM&Q')
    os.makedirs(save_folder, exist_ok=True)
    save_img = os.path.join(save_folder, f'q_update_plot_{args.method}_r{args.r}_lr{args.lr}_alpha{args.alpha}.png') if args.method in ['lora','dora'] else os.path.join(save_folder, f'q_update_plot_{args.method}_r{args.r}_lambda{args.lam}_lr{args.lr}_alpha{args.alpha}.png')
    if args.method == 'full':
        save_img = os.path.join(save_folder, f'q_update_plot_full-tuning.png')

    plot_updates(all_results, args.method, save_img)
    print(f'Saved plot to {save_img}')