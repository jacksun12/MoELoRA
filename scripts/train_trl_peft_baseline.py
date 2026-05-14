import json
import os
from typing import Dict, List

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from data.data_loader import MovieLens1MSequential


LETTERS = ["A", "B", "C", "D"]


def load_config(path: str = "config_unified.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_text_rows(ds: MovieLens1MSequential) -> List[Dict]:
    rows = []
    for sample in ds.samples:
        prompt = ds._format_prompt(sample)
        answer = LETTERS[int(sample["target_option_idx"])]
        rows.append(
            {
                "prompt": prompt,
                "answer": answer,
                "text": prompt + answer,
            }
        )
    return rows


def extract_letter(text: str):
    t = text.strip().upper()
    for ch in t:
        if ch in LETTERS:
            return ch
    return None


@torch.no_grad()
def evaluate_generate_hit1(model, tokenizer, test_rows: List[Dict], max_new_tokens: int = 4):
    model.eval()
    total = 0
    correct = 0

    for item in test_rows:
        inputs = tokenizer(item["prompt"], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = extract_letter(gen)
        if pred == item["answer"]:
            correct += 1
        total += 1

    hit1 = (correct / total) if total > 0 else 0.0
    return {"generate_hit@1": hit1, "correct": correct, "total": total}


def main():
    cfg = load_config("config_unified.yaml")

    model_id = cfg["model"]["base_model_path"]
    out_root = cfg.get("outputs", {}).get("checkpoint_root", "checkpoints")
    out_dir = os.path.join(out_root, "trl_peft_baseline")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    common_kwargs = {
        "tokenizer": tokenizer,
        "max_seq_len": cfg["data"]["max_seq_len"],
        "history_size": cfg["data"].get("history_size", 10),
        "min_history": cfg["data"].get("min_history", 3),
        "min_user_samples": cfg["data"].get("min_user_samples", 3),
        "neg_sample_size": cfg["data"].get("neg_sample_size", 3),
        "seed": cfg["data"].get("seed", 42),
    }
    train_ds = MovieLens1MSequential(split="train", **common_kwargs)
    val_ds = MovieLens1MSequential(split="val", **common_kwargs)
    test_ds = MovieLens1MSequential(split="test", **common_kwargs)

    train_rows = to_text_rows(train_ds)
    val_rows = to_text_rows(val_ds)
    test_rows = to_text_rows(test_ds)

    print(f"Loaded train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    train_dataset = Dataset.from_list(train_rows)
    val_dataset = Dataset.from_list(val_rows)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()

    peft_cfg = LoraConfig(
        r=cfg["rank_allocation"].get("min_rank", 2) * 4,
        lora_alpha=cfg["model"].get("alpha", 16),
        lora_dropout=cfg["model"].get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["model"]["target_modules"],
    )

    train_args = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=cfg["data"].get("batch_size", 2),
        per_device_eval_batch_size=cfg["data"].get("batch_size", 2),
        gradient_accumulation_steps=cfg["training"].get("gradient_accumulation_steps", 4),
        learning_rate=cfg["training"].get("lora_lr", 2e-4),
        num_train_epochs=max(1, int(cfg["training"].get("cluster_epochs", 1))),
        logging_steps=max(1, int(cfg["training"].get("curve_log_every", 20))),
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        bf16=False,
        fp16=False,
        max_length=cfg["data"]["max_seq_len"],
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()

    adapter_dir = os.path.join(out_dir, "final_adapter")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter saved to: {adapter_dir}")

    # 重新以 PEFT 包装器加载模型，用于生成式评测。
    # Reload the model with a PEFT wrapper for generation-style evaluation.
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    if torch.cuda.is_available():
        base_model = base_model.cuda()
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    metrics = evaluate_generate_hit1(peft_model, tokenizer, test_rows)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
