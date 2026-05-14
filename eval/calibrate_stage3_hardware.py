import argparse
import json
import os
import sys
import time
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.common import load_config
from eval.stage3_common import (
    build_single_prompt_batch,
    cuda_sync,
    dump_json,
    load_stage0_tokenizer,
    resolve_dtype,
    stage0_adapter_checkpoint_dir,
    timestamped_result_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate Stage-3 serving latency on the current GPU server.")
    parser.add_argument("--config", default="config_unified.yaml")
    parser.add_argument("--profile", default="", help="Optional profile inside a unified config file.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch_sizes", default="1,2,4,8")
    parser.add_argument("--prompt_lengths", default="64,128,256")
    parser.add_argument("--decode_lengths", default="32,64")
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--measure_steps", type=int, default=5)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def parse_int_list(spec: str) -> List[int]:
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def measure_prefill(model, batch, device: str, warmup_steps: int, measure_steps: int) -> Dict:
    times_ms = []
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(**batch)
            cuda_sync(device)
        for _ in range(measure_steps):
            start = time.perf_counter()
            _ = model(**batch)
            cuda_sync(device)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    batch_size = batch["input_ids"].shape[0]
    seq_len = batch["input_ids"].shape[1]
    total_tokens = batch_size * seq_len
    mean_ms = sum(times_ms) / len(times_ms)
    return {
        "mode": "prefill",
        "batch_size": int(batch_size),
        "prompt_length": int(seq_len),
        "mean_latency_ms": float(mean_ms),
        "tokens_per_sec": float((total_tokens / mean_ms) * 1000.0) if mean_ms > 0 else 0.0,
    }


def measure_decode(model, batch, decode_len: int, tokenizer, device: str, warmup_steps: int, measure_steps: int) -> Dict:
    times_ms = []
    model.eval()
    gen_kwargs = {
        "max_new_tokens": int(decode_len),
        "min_new_tokens": int(decode_len),
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": None,
    }
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model.generate(**batch, **gen_kwargs)
            cuda_sync(device)
        for _ in range(measure_steps):
            start = time.perf_counter()
            _ = model.generate(**batch, **gen_kwargs)
            cuda_sync(device)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    batch_size = batch["input_ids"].shape[0]
    total_decode_tokens = batch_size * decode_len
    mean_ms = sum(times_ms) / len(times_ms)
    return {
        "mode": "decode",
        "batch_size": int(batch_size),
        "prompt_length": int(batch["input_ids"].shape[1]),
        "decode_length": int(decode_len),
        "mean_latency_ms": float(mean_ms),
        "tokens_per_sec": float((total_decode_tokens / mean_ms) * 1000.0) if mean_ms > 0 else 0.0,
    }


def load_model_variants(cfg, device: str, dtype):
    base_id = cfg["model"]["base_model_path"]
    merged_dir = cfg.get("stage0", {}).get("merged_output_dir", "outputs/stage0_cloud_merged")
    adapter_dir = stage0_adapter_checkpoint_dir(cfg)

    variants = {
        "base_original": base_id,
    }
    if os.path.isdir(merged_dir) and os.path.exists(os.path.join(merged_dir, "config.json")):
        variants["warm_merged"] = merged_dir
    if os.path.isdir(adapter_dir) and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        variants["stage0_adapter"] = {"base": base_id, "adapter": adapter_dir}

    loaded = {}
    for name, spec in variants.items():
        if isinstance(spec, dict):
            base_model = AutoModelForCausalLM.from_pretrained(spec["base"], torch_dtype=dtype).to(device)
            model = PeftModel.from_pretrained(base_model, spec["adapter"]).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(spec, torch_dtype=dtype).to(device)
        loaded[name] = model
    return loaded


def main():
    args = parse_args()
    cfg = load_config(args.config, profile=args.profile or None)
    device = args.device
    dtype = resolve_dtype(args.dtype)
    batch_sizes = parse_int_list(args.batch_sizes)
    prompt_lengths = parse_int_list(args.prompt_lengths)
    decode_lengths = parse_int_list(args.decode_lengths)

    tokenizer = load_stage0_tokenizer(cfg)

    variants = load_model_variants(cfg, device=device, dtype=dtype)
    results = []

    for variant_name, model in variants.items():
        for batch_size in batch_sizes:
            for prompt_len in prompt_lengths:
                batch = build_single_prompt_batch(tokenizer, prompt_length=prompt_len, batch_size=batch_size, device=device)
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device=device)

                prefill = measure_prefill(
                    model,
                    batch,
                    device=device,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                )
                prefill["variant"] = variant_name
                if device.startswith("cuda"):
                    prefill["peak_alloc_gb"] = float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 3))
                results.append(prefill)

                for decode_len in decode_lengths:
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device=device)
                    decode = measure_decode(
                        model,
                        batch,
                        decode_len=decode_len,
                        tokenizer=tokenizer,
                        device=device,
                        warmup_steps=args.warmup_steps,
                        measure_steps=args.measure_steps,
                    )
                    decode["variant"] = variant_name
                    if device.startswith("cuda"):
                        decode["peak_alloc_gb"] = float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 3))
                    results.append(decode)

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if not args.output_json:
        args.output_json = timestamped_result_path("stage3_hardware_calibration")

    dump_json(
        args.output_json,
        {
            "config": args.config,
            "device": device,
            "dtype": args.dtype,
            "batch_sizes": batch_sizes,
            "prompt_lengths": prompt_lengths,
            "decode_lengths": decode_lengths,
            "results": results,
        },
    )

    print(json.dumps({"output_json": args.output_json, "num_results": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
