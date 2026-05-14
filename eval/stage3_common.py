import json
import os
from datetime import datetime
from typing import Dict

import torch
from transformers import AutoTokenizer


def resolve_dtype(name: str):
    """Map a dtype name to a torch dtype. / 将字符串 dtype 映射为 torch dtype。"""
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_stage0_tokenizer(cfg):
    """Load the tokenizer used by serving-side probes. / 加载第三阶段探测所用 tokenizer。"""
    tokenizer_source = cfg.get("stage0", {}).get("merged_output_dir", cfg["model"]["base_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_prompt_text(target_tokens: int) -> str:
    """Create a synthetic prompt. / 构造合成提示词。"""
    unit = "Please rewrite the following sentence in a personalized style. "
    repeat = max(1, target_tokens // 10)
    return (unit * repeat).strip()


def build_single_prompt_batch(tokenizer, prompt_length: int, batch_size: int, device: str) -> Dict[str, torch.Tensor]:
    """Tokenize a synthetic batch for latency probes. / 为时延探测构造一批输入。"""
    text = build_prompt_text(prompt_length)
    enc = tokenizer(
        [text] * batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=prompt_length,
    )
    return {k: v.to(device) for k, v in enc.items()}


def cuda_sync(device: str):
    """Synchronize CUDA if available. / 如使用 CUDA 则执行同步。"""
    if device.startswith("cuda"):
        torch.cuda.synchronize(device=device)


def stage0_adapter_checkpoint_dir(cfg) -> str:
    """Return the stage-0 adapter checkpoint path. / 返回 stage-0 adapter checkpoint 路径。"""
    return os.path.join(
        cfg.get("stage0", {}).get("adapter_output_dir", "outputs/stage0_cloud_adapter"),
        "checkpoint-1921",
    )


def timestamped_result_path(prefix: str) -> str:
    """Build a timestamped eval/results path. / 生成带时间戳的结果文件路径。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("eval", "results", f"{prefix}_{ts}.json")


def dump_json(path: str, payload):
    """Write a JSON file with parent dir creation. / 写出 JSON，并自动创建父目录。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
