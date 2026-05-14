import argparse
import json
import os
import sys
import time
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

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
    parser = argparse.ArgumentParser(description="Measure real adapter-switching overhead on the current GPU server.")
    parser.add_argument("--config", default="config_unified.yaml")
    parser.add_argument("--profile", default="", help="Optional profile inside a unified config file.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--prompt_length", type=int, default=128)
    parser.add_argument("--decode_length", type=int, default=32)
    parser.add_argument("--num_requests", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--measure_rounds", type=int, default=3)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def load_two_adapter_model(cfg, device: str, dtype):
    base_id = cfg["model"]["base_model_path"]
    adapter_dir = stage0_adapter_checkpoint_dir(cfg)
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_dir}")

    base_model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base_model, adapter_dir, adapter_name="adapter_a").to(device)
    model.load_adapter(adapter_dir, adapter_name="adapter_b", is_trainable=False, torch_device=device)
    model.eval()
    return model


def build_sequences(num_requests: int) -> Dict[str, List[str]]:
    half = max(1, num_requests // 2)
    grouped = ["adapter_a"] * half + ["adapter_b"] * (num_requests - half)
    alternating = [("adapter_a" if idx % 2 == 0 else "adapter_b") for idx in range(num_requests)]
    single = ["adapter_a"] * num_requests
    return {
        "single_adapter_only": single,
        "grouped_adapters": grouped,
        "alternating_adapters": alternating,
    }


def run_prefill_sequence(model, batch, adapter_sequence: List[str], device: str, warmup_steps: int, measure_rounds: int):
    for _ in range(warmup_steps):
        for adapter_name in adapter_sequence:
            model.set_adapter(adapter_name)
            _ = model(**batch)
        cuda_sync(device)

    latencies_ms = []
    wall_ms = []
    with torch.no_grad():
        for _ in range(measure_rounds):
            per_request = []
            start_round = time.perf_counter()
            for adapter_name in adapter_sequence:
                model.set_adapter(adapter_name)
                cuda_sync(device)
                start = time.perf_counter()
                _ = model(**batch)
                cuda_sync(device)
                end = time.perf_counter()
                per_request.append((end - start) * 1000.0)
            end_round = time.perf_counter()
            latencies_ms.append(sum(per_request) / len(per_request))
            wall_ms.append((end_round - start_round) * 1000.0)

    return float(sum(latencies_ms) / len(latencies_ms)), float(sum(wall_ms) / len(wall_ms))


def run_decode_sequence(model, batch, adapter_sequence: List[str], tokenizer, decode_length: int, device: str, warmup_steps: int, measure_rounds: int):
    gen_kwargs = {
        "max_new_tokens": int(decode_length),
        "min_new_tokens": int(decode_length),
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": None,
    }
    for _ in range(warmup_steps):
        for adapter_name in adapter_sequence:
            model.set_adapter(adapter_name)
            _ = model.generate(**batch, **gen_kwargs)
        cuda_sync(device)

    latencies_ms = []
    wall_ms = []
    with torch.no_grad():
        for _ in range(measure_rounds):
            per_request = []
            start_round = time.perf_counter()
            for adapter_name in adapter_sequence:
                model.set_adapter(adapter_name)
                cuda_sync(device)
                start = time.perf_counter()
                _ = model.generate(**batch, **gen_kwargs)
                cuda_sync(device)
                end = time.perf_counter()
                per_request.append((end - start) * 1000.0)
            end_round = time.perf_counter()
            latencies_ms.append(sum(per_request) / len(per_request))
            wall_ms.append((end_round - start_round) * 1000.0)

    return float(sum(latencies_ms) / len(latencies_ms)), float(sum(wall_ms) / len(wall_ms))


def main():
    args = parse_args()
    cfg = load_config(args.config, profile=args.profile or None)
    device = args.device
    dtype = resolve_dtype(args.dtype)

    tokenizer = load_stage0_tokenizer(cfg)

    model = load_two_adapter_model(cfg, device=device, dtype=dtype)
    batch = build_single_prompt_batch(tokenizer, prompt_length=args.prompt_length, batch_size=1, device=device)
    sequences = build_sequences(args.num_requests)

    results = []
    for sequence_name, adapter_sequence in sequences.items():
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device=device)
        prefill_mean_ms, prefill_wall_ms = run_prefill_sequence(
            model,
            batch,
            adapter_sequence=adapter_sequence,
            device=device,
            warmup_steps=args.warmup_steps,
            measure_rounds=args.measure_rounds,
        )
        peak_prefill = float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)) if device.startswith("cuda") else None

        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device=device)
        decode_mean_ms, decode_wall_ms = run_decode_sequence(
            model,
            batch,
            adapter_sequence=adapter_sequence,
            tokenizer=tokenizer,
            decode_length=args.decode_length,
            device=device,
            warmup_steps=args.warmup_steps,
            measure_rounds=args.measure_rounds,
        )
        peak_decode = float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)) if device.startswith("cuda") else None

        results.append(
            {
                "sequence": sequence_name,
                "num_requests": len(adapter_sequence),
                "prompt_length": int(batch["input_ids"].shape[1]),
                "decode_length": int(args.decode_length),
                "prefill_mean_request_ms": prefill_mean_ms,
                "prefill_total_round_ms": prefill_wall_ms,
                "prefill_requests_per_sec": float((len(adapter_sequence) / prefill_wall_ms) * 1000.0),
                "prefill_peak_alloc_gb": peak_prefill,
                "decode_mean_request_ms": decode_mean_ms,
                "decode_total_round_ms": decode_wall_ms,
                "decode_requests_per_sec": float((len(adapter_sequence) / decode_wall_ms) * 1000.0),
                "decode_peak_alloc_gb": peak_decode,
            }
        )

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    if not args.output_json:
        args.output_json = timestamped_result_path("stage3_adapter_switching")

    dump_json(
        args.output_json,
        {
            "config": args.config,
            "device": device,
            "dtype": args.dtype,
            "prompt_length": args.prompt_length,
            "decode_length": args.decode_length,
            "num_requests": args.num_requests,
            "results": results,
        },
    )

    print(json.dumps({"output_json": args.output_json, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
