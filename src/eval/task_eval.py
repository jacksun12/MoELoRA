import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.pipeline.common import get_cfg_batch_size, get_dataset_task_type, get_model_device, maybe_subset
from src.utils.metrics import hit_rate_at_k, ndcg_at_k


@torch.no_grad()
def evaluate_recall_ndcg(model, dataset, batch_size, device, tokenizer, k=10, max_samples=2000):
    eval_dataset = maybe_subset(dataset, max_samples)
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    model_device = get_model_device(model)
    letter_token_ids = []
    for letter in ["A", "B", "C", "D"]:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) == 0:
            raise ValueError(f"Tokenizer failed to encode option letter: {letter}")
        letter_token_ids.append(ids[0])

    effective_k = min(int(k), 4)
    hr_values = []
    ndcg_values = []
    for batch in tqdm(loader, desc="Eval HR/NDCG", leave=False):
        input_ids = batch["eval_input_ids"].to(model_device)
        attention_mask = batch["eval_attention_mask"].to(model_device)
        targets = batch["target_option_idx"].to(model_device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        seq_lens = attention_mask.sum(dim=1)
        pred_pos = (seq_lens - 1).clamp(min=0)
        row_idx = torch.arange(input_ids.size(0), device=model_device)
        predictions = logits[row_idx, pred_pos, :][:, letter_token_ids]
        hr_values.append(hit_rate_at_k(predictions, targets, k=effective_k))
        ndcg_values.append(ndcg_at_k(predictions, targets, k=effective_k))

    return {
        f"hr@{effective_k}": float(np.mean(hr_values)) if hr_values else 0.0,
        f"ndcg@{effective_k}": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "eval_samples": len(eval_dataset),
    }


@torch.no_grad()
def evaluate_generate_hit1(model, dataset, batch_size, device, tokenizer, max_samples=1000):
    eval_dataset = maybe_subset(dataset, max_samples)
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    model_device = get_model_device(model)

    correct = 0
    total = 0
    letters = ["A", "B", "C", "D"]
    for batch in tqdm(loader, desc="Eval Generate Hit@1", leave=False):
        input_ids = batch["eval_input_ids"].to(model_device)
        attention_mask = batch["eval_attention_mask"].to(model_device)
        targets = batch["target_option_idx"].to(model_device)
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=2, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        gen_tokens = out[:, input_ids.size(1):]
        pred_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        preds = []
        for txt in pred_text:
            t = txt.strip().upper()
            p = -1
            for ch in t:
                if ch in letters:
                    p = letters.index(ch)
                    break
            preds.append(p)
        pred_idx = torch.tensor(preds, dtype=torch.long, device=model_device)
        valid = pred_idx >= 0
        if valid.any():
            correct += int((pred_idx[valid] == targets[valid]).sum().item())
            total += int(valid.sum().item())
    return {"generate_hit@1": float((correct / total) if total > 0 else 0.0), "eval_samples": int(total)}


def _normalize_text(text):
    return " ".join(str(text).strip().lower().split())


def _get_ngrams(tokens, n):
    if len(tokens) < n or n <= 0:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _bleu_n(pred_text, target_text, n=4):
    pred_tokens = _normalize_text(pred_text).split()
    target_tokens = _normalize_text(target_text).split()
    if not pred_tokens or not target_tokens:
        return 0.0
    precisions = []
    for k in range(1, n + 1):
        pred_ngrams = _get_ngrams(pred_tokens, k)
        target_ngrams = _get_ngrams(target_tokens, k)
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        pred_counts = {}
        target_counts = {}
        for gram in pred_ngrams:
            pred_counts[gram] = pred_counts.get(gram, 0) + 1
        for gram in target_ngrams:
            target_counts[gram] = target_counts.get(gram, 0) + 1
        overlap = 0
        for gram, cnt in pred_counts.items():
            overlap += min(cnt, target_counts.get(gram, 0))
        precisions.append((overlap + 1.0) / (len(pred_ngrams) + 1.0))
    bp = 1.0
    if len(pred_tokens) < len(target_tokens):
        bp = math.exp(1.0 - (len(target_tokens) / max(len(pred_tokens), 1)))
    return float(bp * math.exp(sum(math.log(max(p, 1e-12)) for p in precisions) / n))


def _lcs_length(a, b):
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def _rouge_l_f1(pred_text, target_text):
    pred_tokens = _normalize_text(pred_text).split()
    target_tokens = _normalize_text(target_text).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, target_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(target_tokens)
    if precision + recall <= 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


@torch.no_grad()
def evaluate_text_generation(model, dataset, batch_size, device, tokenizer, max_samples=500, max_new_tokens=96):
    eval_dataset = maybe_subset(dataset, max_samples)
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    model_device = get_model_device(model)
    bleu1_scores = []
    rouge_l_scores = []

    for batch in tqdm(loader, desc="Eval Text Generation", leave=False):
        input_ids = batch["eval_input_ids"].to(model_device)
        attention_mask = batch["eval_attention_mask"].to(model_device)
        targets = batch["target_text"]
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        gen_tokens = out[:, input_ids.size(1):]
        pred_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        for pred_text, target_text in zip(pred_texts, targets):
            bleu1_scores.append(_bleu_n(pred_text, target_text, n=1))
            rouge_l_scores.append(_rouge_l_f1(pred_text, target_text))

    return {
        "bleu1": float(np.mean(bleu1_scores)) if bleu1_scores else 0.0,
        "rougeL_f1": float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0,
        "eval_samples": len(eval_dataset),
    }


@torch.no_grad()
def evaluate_model_for_task(model, dataset, cfg, device, tokenizer):
    task_type = get_dataset_task_type(dataset)
    eval_batch_size = get_cfg_batch_size(cfg, "eval_batch_size")
    eval_cfg = cfg.get("evaluation", {})
    if task_type == "multiple_choice":
        return {
            "ranking_metrics": evaluate_recall_ndcg(model, dataset, batch_size=eval_batch_size, device=device, tokenizer=tokenizer, k=eval_cfg.get("k", 10), max_samples=eval_cfg.get("max_samples", 2000)),
            "generation_metrics": evaluate_generate_hit1(model, dataset, batch_size=eval_batch_size, device=device, tokenizer=tokenizer, max_samples=eval_cfg.get("max_samples", 1000)),
        }

    return {
        "text_generation_metrics": evaluate_text_generation(
            model,
            dataset,
            batch_size=eval_batch_size,
            device=device,
            tokenizer=tokenizer,
            max_samples=eval_cfg.get("max_samples", 500),
            max_new_tokens=int(eval_cfg.get("max_new_tokens", 96)),
        )
    }
