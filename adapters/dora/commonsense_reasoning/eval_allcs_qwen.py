# eval_all_commonsense_qwen.py
# Evaluate a finetuned Qwen2.5-3B (LoRA / SoRA / DoRA etc.)
# on multiple commonsense_reasoning datasets and log results to experiment/results.csv.

import copy
import csv
import json
import os
import re
import argparse

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
sys.path.append(PROJECT_ROOT)

from utils import replace_linear_with_lora

device = "cuda" if torch.cuda.is_available() else "cpu"

# Default commonsense datasets to evaluate
DEFAULT_DATASETS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
]


# -----------------------------
# Prompt & dataset helpers
# -----------------------------
def generate_prompt(instruction: str, input_text: str = None) -> str:
    """Format the prompt in the same way as during training."""
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Response:
"""


def load_data(dataset_name: str) -> list:
    """Load the test.json file for a given dataset."""
    file_path = os.path.join("dataset", dataset_name, "test.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def create_batch(dataset, batch_size: int):
    """Split the dataset into mini-batches."""
    batches = []
    n = len(dataset)
    num_batch = n // batch_size + (1 if n % batch_size != 0 else 0)
    for i in range(num_batch):
        batch = dataset[i * batch_size : min((i + 1) * batch_size, n)]
        batches.append(batch)
    return batches


# -----------------------------
# Answer extraction
# -----------------------------
def extract_answer(dataset: str, sentence: str) -> str:
    """
    Extract the model's predicted answer token (e.g., 'answer1', 'true', 'ending2')
    from the generated sentence using regex, following the original repo's logic.
    """
    sentence_ = sentence.strip()
    if dataset == "boolq":
        pred_answers = re.findall(r"true|false", sentence_, flags=re.IGNORECASE)
        return pred_answers[0].lower() if pred_answers else ""
    elif dataset == "piqa":
        pred_answers = re.findall(r"solution1|solution2", sentence_)
        return pred_answers[0] if pred_answers else ""
    elif dataset in ["social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa"]:
        pred_answers = re.findall(r"answer1|answer2|answer3|answer4|answer5", sentence_)
        return pred_answers[0] if pred_answers else ""
    elif dataset == "hellaswag":
        pred_answers = re.findall(r"ending1|ending2|ending3|ending4", sentence_)
        return pred_answers[0] if pred_answers else ""
    elif dataset == "winogrande":
        pred_answers = re.findall(r"option1|option2", sentence_)
        return pred_answers[0] if pred_answers else ""
    else:
        return ""


# -----------------------------
# Model loading
# -----------------------------
def load_model_and_tokenizer(
    base_model: str,
    model_dir: str,
    checkpoint_name: str,
    adapter_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules=None,
):
    """
    Load tokenizer + base Qwen, then inject the same adapter structure
    (LoRA / SoRA / DoRA / SDoRA) with replace_linear_with_lora, and finally
    load the finetuned checkpoint state_dict.

    This matches the training-time architecture, so there should be no
    missing/unexpected keys if everything is consistent.
    """
    if target_modules is None:
        # Must match training script
        target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]

    # 1) Load tokenizer from finetuned directory to keep added tokens and template
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    # 2) Instantiate base model architecture
    print(f"Loading base model from HF: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map=None,
    )

    # 3) Rebuild the same adapter structure used during finetuning
    model = replace_linear_with_lora(
        model,
        target_modules=target_modules,
        adapter_name=adapter_name,  # "lora", "sora", "dora", "sdora"
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_as_merged=True if adapter_name.lower() in ["sora", "sdora"] else False,
    )

    # 4) Load finetuned weights
    ckpt_path = os.path.join(model_dir, checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading finetuned weights from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    
    def is_sora_gate_param(name: str) -> bool:
        # True SoRA gate params look like "...something.gate"
        last = name.split(".")[-1]
        return last == "gate"

    adapter = adapter_name.lower()
    if adapter in ["sora", "sdora"]:
        # Filter out SoRA-specific low-rank branch parameters
        filtered_state = {}
        for k, v in state.items():
            # Drop lora_A / lora_B tensors
            if "lora_A" in k or "lora_B" in k:
                continue
            # Drop true gate tensors (ending with ".gate")
            if is_sora_gate_param(k):
                continue
            # Keep everything else (including gate_proj.weight etc.)
            filtered_state[k] = v

        to_load = filtered_state
        print(f"[INFO] Using filtered state_dict for {adapter_name} (size={len(to_load)})")
    else:
        # For plain LoRA / DoRA we keep the entire state_dict
        to_load = state
        print(f"[INFO] Using full state_dict for {adapter_name} (size={len(to_load)})")

    missing, unexpected = model.load_state_dict(to_load, strict=False)

    if missing:
        print("[WARN] Missing keys when loading state_dict:", missing[:20], "...")
    if unexpected:
        print("[WARN] Unexpected keys when loading state_dict:", unexpected[:20], "...")

    model.to(device)
    model.eval()
    return tokenizer, model


# -----------------------------
# Evaluate a single dataset
# -----------------------------
def evaluate_dataset(
    dataset_name: str,
    base_model: str,
    model_dir: str,
    checkpoint_name: str,
    adapter_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    batch_size: int = 16,
    max_new_tokens: int = 32,
    temperature: float = 0.1,
    top_p: float = 0.75,
    top_k: int = 40,
    num_beams: int = 4,
):
    """
    Run evaluation on a single dataset and return (correct, total, accuracy).
    Also save per-sample outputs to experiment/<model_tag>-<dataset>.json.
    """
    tokenizer, model = load_model_and_tokenizer(
        base_model=base_model,
        model_dir=model_dir,
        checkpoint_name=checkpoint_name,
        adapter_name=adapter_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    data = load_data(dataset_name)
    batches = create_batch(data, batch_size)

    os.makedirs("experiment", exist_ok=True)
    model_tag = os.path.basename(model_dir.rstrip("/"))
    save_file = os.path.join("experiment", f"{model_tag}-{dataset_name}.json")

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )

    total_batches = len(batches)
    correct = 0
    current = 0
    output_data = []

    pbar = tqdm(total=total_batches, desc=f"Evaluating {dataset_name}")
    for idx, batch in enumerate(batches):
        current += len(batch)

        instructions = [item.get("instruction", "") for item in batch]
        inputs = [item.get("input", "") for item in batch]

        prompts = [generate_prompt(instr, inp) for instr, inp in zip(instructions, inputs)]

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=max_new_tokens,
            )
        sequences = gen_out.sequences
        texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        outputs = [t.split("### Response:")[-1].strip() for t in texts]

        for item, output in zip(batch, outputs):
            label = item.get("answer", "")
            pred = extract_answer(dataset_name, output)
            flag = (label == pred)
            if flag:
                correct += 1

            new_item = copy.deepcopy(item)
            new_item["output_pred"] = output
            new_item["pred"] = pred
            new_item["flag"] = flag
            output_data.append(new_item)

        acc = correct / current if current > 0 else 0.0
        print(f"\rBatch {idx+1}/{total_batches} | correct={correct}  accuracy={acc:.4f}")
        with open(save_file, "w") as f:
            json.dump(output_data, f, indent=4)

        pbar.update(1)

    pbar.close()
    final_acc = correct / current if current > 0 else 0.0
    print(f"\n[{dataset_name}] Accuracy: {correct}/{current} = {final_acc:.4f}")
    print(f"Saved per-sample outputs to: {save_file}")

    return correct, current, final_acc


# -----------------------------
# Evaluate all datasets & write CSV
# -----------------------------
def evaluate_all(
    model_dir: str,
    base_model: str,
    checkpoint_name: str,
    adapter_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    datasets=None,
    batch_size: int = 16,
):
    """
    Evaluate a finetuned model on multiple commonsense datasets and append
    results to experiment/results.csv.

    Each row in results.csv has:
      model_tag, dataset, correct, total, accuracy
    """
    if datasets is None:
        datasets = DEFAULT_DATASETS

    os.makedirs("experiment", exist_ok=True)
    results_csv = os.path.join("experiment", "results.csv")

    csv_exists = os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(["model", "dataset", "correct", "total", "accuracy"])

        model_tag = os.path.basename(model_dir.rstrip("/"))

        for ds in datasets:
            print("\n" + "=" * 70)
            print(f"Evaluating dataset: {ds} for model: {model_tag}")
            print("=" * 70)

            correct, total, acc = evaluate_dataset(
                dataset_name=ds,
                base_model=base_model,
                model_dir=model_dir,
                checkpoint_name=checkpoint_name,
                adapter_name=adapter_name,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                batch_size=batch_size,
            )

            writer.writerow([model_tag, ds, correct, total, f"{acc:.4f}"])
            print(f"[RESULT] {model_tag} | {ds} | acc={acc:.4f}")
            print("-" * 70)


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Finetuned model directory, e.g. outputs/Qwen2.5-3B_lora_r16",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="HF base model name used to instantiate architecture.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="model.pt",
        help="Checkpoint inside model_dir to load, e.g. model.pt or checkpoint-epoch2/model.pt",
    )
    parser.add_argument("--batch_size", type=int, default=16)

    # Adapter hyperparameters must match training
    parser.add_argument(
        "--adapter_name",
        type=str,
        default="lora",
        choices=["lora", "sora", "dora", "sdora"],
        help="Adapter type used during finetuning.",
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of datasets to evaluate; default is all commonsense datasets.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_all(
        model_dir=args.model_dir,
        base_model=args.base_model,
        checkpoint_name=args.checkpoint_name,
        adapter_name=args.adapter_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        datasets=args.datasets,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()