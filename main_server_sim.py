import math
import os
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, get_cosine_schedule_with_warmup
from trl import SFTConfig, SFTTrainer

from data.data_loader import build_dataset, extract_clustering_features
from src.clustering.spherical_cluster import DBCHKSelector
from src.models.moe_layer import MoELoRALinear
from src.system.hetero_batcher import HeteroBatchScheduler, InferenceRequest
from src.system.overlap_manager import ClusterOverlapManager
from src.system.rank_allocator import BudgetedRankAllocator, normalized_complexity
from src.utils.cluster_viz import visualize_clusters
from src.utils.logger import ExperimentLogger
from src.utils.metrics import hit_rate_at_k, ndcg_at_k
from src.utils.training_plotter import TrainingCurveTracker


@dataclass
class ClusterLayout:
    k: int
    labels: np.ndarray
    centroids: np.ndarray
    ranks: List[int]


def refresh_optimizer(model, lr=1e-4):
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def build_scheduler(optimizer, total_steps, warmup_ratio=0.06):
    if total_steps <= 0:
        return None
    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1)) if total_steps > 1 else 0
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(cfg):
    preferred = cfg.get("runtime", {}).get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if preferred.startswith("cuda") and torch.cuda.is_available():
        # When CUDA_VISIBLE_DEVICES is set, visible GPUs are re-indexed from 0.
        # If config points to an out-of-range index (e.g., cuda:7 with only one visible GPU),
        # gracefully fall back to cuda:0 to avoid selecting unintended physical GPUs.
        try:
            if ":" in preferred:
                idx = int(preferred.split(":")[1])
            else:
                idx = 0
        except Exception:
            idx = 0
        if idx >= torch.cuda.device_count():
            return "cuda:0"
    return preferred


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_subset(dataset, max_samples):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    seed = 42
    idx = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    return Subset(dataset, idx[:max_samples].tolist())


def _rows_from_dataset_for_sft(dataset):
    base = dataset
    indices = np.arange(len(dataset))

    # Support nested Subset(Subset(...)) by progressively mapping indices.
    while isinstance(base, Subset):
        parent = base.dataset
        parent_indices = np.asarray(base.indices)
        indices = parent_indices[indices]
        base = parent

    if not hasattr(base, "samples") or not hasattr(base, "_format_prompt"):
        raise AttributeError("Base dataset for TRL conversion must expose .samples and ._format_prompt().")

    letters = ["A", "B", "C", "D"]
    rows = []
    for i in indices:
        s = base.samples[int(i)]
        answer = letters[int(s["target_option_idx"])]
        rows.append({"text": base._format_prompt(s) + answer})
    return rows


class CurveLogCallback(TrainerCallback):
    def __init__(self, curve_tracker, stage, track, global_step_state=None):
        self.curve_tracker = curve_tracker
        self.stage = stage
        self.track = track
        self.global_step_state = global_step_state
        self.offset = 0 if global_step_state is None else int(global_step_state.get("step", 0))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.curve_tracker is None or logs is None:
            return
        if "loss" not in logs:
            return
        gstep = self.offset + int(state.global_step)
        self.curve_tracker.log(
            stage=self.stage,
            track=self.track,
            step=gstep,
            loss=float(logs["loss"]),
        )


def run_trl_sft(
    model,
    dataset,
    cfg,
    lr,
    epochs,
    stage,
    track,
    logger,
    curve_tracker=None,
    global_step_state=None,
):
    rows = _rows_from_dataset_for_sft(dataset)
    if len(rows) == 0:
        return float("nan")

    trl_ds = Dataset.from_list(rows)
    run_root = cfg.get("outputs", {}).get("trl_run_root", "eval/trl_runs")
    os.makedirs(run_root, exist_ok=True)
    out_dir = os.path.join(run_root, f"{stage}_{track}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    trl_bs = int(cfg["training"].get("trl_per_device_batch_size", cfg["data"].get("batch_size", 2)))
    trl_accum = int(
        cfg["training"].get(
            "trl_gradient_accumulation_steps",
            cfg["training"].get("gradient_accumulation_steps", 1),
        )
    )
    use_bf16 = bool(cfg["training"].get("trl_bf16", torch.cuda.is_available()))
    use_fp16 = bool(cfg["training"].get("trl_fp16", False))
    if use_bf16:
        use_fp16 = False

    train_cfg = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=max(1, trl_bs),
        gradient_accumulation_steps=max(1, trl_accum),
        learning_rate=float(lr),
        num_train_epochs=int(max(1, epochs)),
        logging_steps=max(1, int(cfg["training"].get("curve_log_every", 20))),
        report_to="none",
        bf16=use_bf16,
        fp16=use_fp16,
        max_length=int(cfg["data"]["max_seq_len"]),
        gradient_checkpointing=bool(cfg["training"].get("trl_gradient_checkpointing", True)),
        dataloader_num_workers=int(cfg["training"].get("trl_dataloader_num_workers", 0)),
        save_strategy="no",
    )

    callback = CurveLogCallback(
        curve_tracker=curve_tracker,
        stage=stage,
        track=track,
        global_step_state=global_step_state,
    )
    trainer = SFTTrainer(
        model=model,
        args=train_cfg,
        train_dataset=trl_ds,
        processing_class=cfg["_tokenizer_obj"],
        callbacks=[callback],
    )

    # Save VRAM during training for decoder-only models.
    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False
    result = trainer.train()
    if old_use_cache is not None:
        model.config.use_cache = old_use_cache
    if global_step_state is not None:
        global_step_state["step"] = int(global_step_state.get("step", 0)) + int(trainer.state.global_step)
    loss = float(result.training_loss) if hasattr(result, "training_loss") else float("nan")
    logger.info(f"[{stage}][{track}] trl_steps={trainer.state.global_step}, trl_loss={loss:.4f}")
    return loss


def build_feature_cache_path(cfg, split_name, dataset_len):
    cache_dir = cfg["data"].get("feature_cache_dir", "data/feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    model_id = cfg["model"]["base_model_path"].replace("/", "_")
    strategy = cfg["clustering"].get("feature_strategy", "sentence_mean")
    max_seq = cfg["data"]["max_seq_len"]
    name = f"{split_name}_{model_id}_{strategy}_n{dataset_len}_seq{max_seq}.npz"
    return os.path.join(cache_dir, name)


def save_model_checkpoint(model, cfg, stage_name, logger, extra_meta=None):
    root_dir = cfg.get("outputs", {}).get("checkpoint_root", "checkpoints")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "stage": stage_name,
        "timestamp": ts,
        "model_state_dict": model.state_dict(),
        "extra_meta": extra_meta or {},
    }
    try:
        os.makedirs(root_dir, exist_ok=True)
        ckpt_path = os.path.join(root_dir, f"{stage_name}_{ts}.pt")
        torch.save(payload, ckpt_path)
        logger.info(f"[Checkpoint] saved: {ckpt_path}")
        return ckpt_path
    except Exception as e:
        fallback_dir = "checkpoints"
        os.makedirs(fallback_dir, exist_ok=True)
        ckpt_path = os.path.join(fallback_dir, f"{stage_name}_{ts}.pt")
        torch.save(payload, ckpt_path)
        logger.warning(f"[Checkpoint] failed to save to {root_dir}: {e}; fallback={ckpt_path}")
        return ckpt_path


def inject_system_into_base_model(model, target_modules, initial_ranks, top_k=1, alpha=16, dropout=0.05):
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, torch.nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)

            moe_lora_layer = MoELoRALinear(
                module,
                initial_ranks=initial_ranks,
                top_k=top_k,
                alpha=alpha,
                dropout=dropout,
            )
            moe_lora_layer = moe_lora_layer.to(device=module.weight.device, dtype=module.weight.dtype)
            setattr(parent, child_name, moe_lora_layer)


def get_moe_layers(model):
    return [m for m in model.modules() if isinstance(m, MoELoRALinear)]


def set_trainable_expert_only(model, expert_idx: int):
    for p in model.parameters():
        p.requires_grad = False

    for layer in get_moe_layers(model):
        layer.set_force_expert(expert_idx)
        for p in layer.experts[expert_idx].parameters():
            p.requires_grad = True


def set_trainable_router_only(model):
    for p in model.parameters():
        p.requires_grad = False

    for layer in get_moe_layers(model):
        layer.clear_force_expert()
        for p in layer.router.parameters():
            p.requires_grad = True


def clear_forced_experts(model):
    for layer in get_moe_layers(model):
        layer.clear_force_expert()


@torch.no_grad()
def set_model_router_centroids(model, centroids):
    for layer in get_moe_layers(model):
        layer.set_router_centroids(centroids)


@torch.no_grad()
def collect_features(model, dataset, batch_size, device, feature_strategy="sentence_mean"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    feats = []
    for batch in tqdm(loader, desc="Feature Extract", leave=False):
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        feature = extract_clustering_features(model, inputs, strategy=feature_strategy)
        feats.append(feature.detach().cpu().float())

    return torch.cat(feats, dim=0).numpy()


def collect_features_with_cache(model, dataset, batch_size, device, feature_strategy, cache_path, logger=None):
    if os.path.exists(cache_path):
        arr = np.load(cache_path)["features"]
        if logger is not None:
            logger.info(f"[FeatureCache] hit: {cache_path}, shape={arr.shape}")
        return arr

    if logger is not None:
        logger.info(f"[FeatureCache] miss: {cache_path}, start extracting...")
    feats = collect_features(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
        feature_strategy=feature_strategy,
    )
    np.savez_compressed(cache_path, features=feats)
    if logger is not None:
        logger.info(f"[FeatureCache] saved: {cache_path}, shape={feats.shape}")
    return feats


def run_lm_epoch(
    model,
    optimizer,
    dataloader,
    device,
    on_step_end=None,
    grad_clip=1.0,
    scheduler=None,
    grad_accum_steps=1,
):
    model.train()
    total_loss = 0.0
    steps = 0
    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Train", leave=False), start=1):
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        }
        outputs = model(**inputs)
        loss = outputs.loss
        loss_val = float(loss.item())
        if not np.isfinite(loss_val):
            continue

        total_loss += loss_val
        steps += 1
        (loss / grad_accum_steps).backward()

        if (batch_idx % grad_accum_steps) == 0:
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            if on_step_end is not None:
                on_step_end(loss_val)

    if (len(dataloader) % grad_accum_steps) != 0:
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        if on_step_end is not None:
            on_step_end(loss_val if steps > 0 else float("nan"))

    return total_loss / steps if steps > 0 else float("nan")


@torch.no_grad()
def evaluate_recall_ndcg(model, dataset, batch_size, device, tokenizer, k=10, max_samples=2000):
    """
    Multiple-choice HR/NDCG on option letters (A/B/C/D) without answer leakage.
    """
    eval_dataset = maybe_subset(dataset, max_samples)
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
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
        input_ids = batch["eval_input_ids"].to(device)
        attention_mask = batch["eval_attention_mask"].to(device)
        targets = batch["target_option_idx"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        seq_lens = attention_mask.sum(dim=1)
        pred_pos = (seq_lens - 1).clamp(min=0)
        row_idx = torch.arange(input_ids.size(0), device=device)

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

    correct = 0
    total = 0
    letters = ["A", "B", "C", "D"]

    for batch in tqdm(loader, desc="Eval Generate Hit@1", leave=False):
        input_ids = batch["eval_input_ids"].to(device)
        attention_mask = batch["eval_attention_mask"].to(device)
        targets = batch["target_option_idx"].to(device)

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_tokens = out[:, input_ids.size(1) :]
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

        pred_idx = torch.tensor(preds, dtype=torch.long, device=device)
        valid = pred_idx >= 0
        if valid.any():
            correct += int((pred_idx[valid] == targets[valid]).sum().item())
            total += int(valid.sum().item())

    hit1 = (correct / total) if total > 0 else 0.0
    return {"generate_hit@1": float(hit1), "eval_samples": int(total)}


@torch.no_grad()
def estimate_cluster_sft_loss(
    model,
    dataset,
    cluster_indices,
    batch_size,
    device,
    max_samples_per_cluster=None,
    random_state=42,
    tokenizer=None,
):
    if len(cluster_indices) == 0:
        return 1.0

    sampled_indices = cluster_indices
    if max_samples_per_cluster is not None and max_samples_per_cluster > 0 and len(cluster_indices) > max_samples_per_cluster:
        rng = np.random.default_rng(random_state)
        sampled_indices = rng.choice(cluster_indices, size=max_samples_per_cluster, replace=False).tolist()

    loader = DataLoader(Subset(dataset, sampled_indices), batch_size=batch_size, shuffle=False)
    model.eval()
    losses = []

    for batch in tqdm(loader, desc="Cluster SFT-Loss", leave=False):
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        }
        out = model(**inputs)
        loss_val = float(out.loss.detach().float().item())
        if np.isfinite(loss_val):
            losses.append(loss_val)

    mean_loss = float(np.mean(losses)) if losses else float(np.log(4.0))
    return mean_loss


def svd_complexity(cluster_features: np.ndarray):
    if cluster_features.shape[0] < 2:
        return 0.0

    x = cluster_features - cluster_features.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    if s.sum() <= 1e-8:
        return 0.0

    energy = s / s.sum()
    entropy = -np.sum(energy * np.log(np.clip(energy, 1e-12, None)))
    return float(entropy)


def svd_energy_curve(cluster_features: np.ndarray, max_rank: int):
    """
    Return per-rank spectral energy shares.
    gain[r] corresponds to the marginal utility of adding the (r+1)-th rank component.
    """
    if cluster_features.shape[0] < 2:
        return np.zeros(max_rank, dtype=np.float32)

    x = cluster_features - cluster_features.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    if s.size == 0:
        return np.zeros(max_rank, dtype=np.float32)

    energy = np.square(s.astype(np.float32))
    total = float(np.clip(energy.sum(), 1e-8, None))
    normalized = energy / total
    curve = np.zeros(max_rank, dtype=np.float32)
    take = min(max_rank, normalized.shape[0])
    curve[:take] = normalized[:take]
    return curve


def build_cluster_rank_utility_curves(
    ppl_values,
    svd_curves,
    min_rank: int,
    max_rank: int,
    ppl_weight: float,
    svd_weight: float,
):
    ppl = np.asarray(ppl_values, dtype=np.float32)
    svd_curves = [np.asarray(curve, dtype=np.float32) for curve in svd_curves]

    if len(svd_curves) == 0:
        return [], []

    ppl_scores = np.asarray(normalized_complexity(ppl, np.zeros_like(ppl), ppl_weight=1.0, svd_weight=0.0), dtype=np.float32)
    utility_curves = []
    cluster_scores = []
    extra_slots = max(0, max_rank - min_rank)

    for cluster_id, svd_curve in enumerate(svd_curves):
        if extra_slots == 0:
            utility_curves.append(np.zeros(0, dtype=np.float32))
            cluster_scores.append(float(ppl_scores[cluster_id]))
            continue

        # Use the post-min-rank spectral tail as marginal utility for extra rank allocation.
        tail = svd_curve[min_rank:max_rank]
        if tail.shape[0] < extra_slots:
            tail = np.pad(tail, (0, extra_slots - tail.shape[0]))

        ppl_factor = 1.0 + ppl_weight * float(ppl_scores[cluster_id])
        marginal = svd_weight * tail * ppl_factor

        # Small residual share keeps allocation stable even when the tail is nearly flat.
        residual = float(tail.sum()) / max(1, extra_slots)
        marginal = marginal + residual * 1e-3

        utility_curves.append(marginal.astype(np.float32))
        cluster_scores.append(float(ppl_scores[cluster_id] + svd_curve[:max_rank].sum()))

    return utility_curves, cluster_scores


def build_rank_layout(model, dataset, features, labels, cfg, device, tokenizer):
    k = int(labels.max()) + 1
    sft_loss_values = []
    svd_values = []
    svd_curves = []

    for cluster_id in range(k):
        idx = np.where(labels == cluster_id)[0].tolist()
        ppl_sample_cap = cfg["rank_allocation"].get("ppl_max_samples_per_cluster", 512)
        sampled_n = min(len(idx), ppl_sample_cap) if ppl_sample_cap and ppl_sample_cap > 0 else len(idx)
        print(f"[RankLayout] cluster={cluster_id}, samples={len(idx)}, loss_sampled={sampled_n}")
        sft_loss_values.append(
            estimate_cluster_sft_loss(
                model,
                dataset,
                idx,
                batch_size=cfg["data"]["batch_size"],
                device=device,
                max_samples_per_cluster=ppl_sample_cap,
                random_state=cfg["clustering"].get("random_state", 42) + cluster_id,
                tokenizer=tokenizer,
            )
        )
        cluster_features = features[idx]
        svd_values.append(svd_complexity(cluster_features))
        svd_curves.append(
            svd_energy_curve(
                cluster_features,
                max_rank=int(cfg["rank_allocation"]["max_rank"]),
            )
        )

    complexity = normalized_complexity(
        sft_loss_values,
        svd_values,
        ppl_weight=cfg["rank_allocation"].get("ppl_weight", 0.6),
        svd_weight=cfg["rank_allocation"].get("svd_weight", 0.4),
    )

    allocator = BudgetedRankAllocator(
        total_rank_budget=cfg["rank_allocation"]["total_budget"],
        min_rank=cfg["rank_allocation"]["min_rank"],
        max_rank=cfg["rank_allocation"]["max_rank"],
        temperature=cfg["rank_allocation"].get("smooth_temperature", 1.5),
        min_extra_share=cfg["rank_allocation"].get("min_extra_share", 0.10),
    )
    utility_curves, cluster_scores = build_cluster_rank_utility_curves(
        ppl_values=sft_loss_values,
        svd_curves=svd_curves,
        min_rank=int(cfg["rank_allocation"]["min_rank"]),
        max_rank=int(cfg["rank_allocation"]["max_rank"]),
        ppl_weight=float(cfg["rank_allocation"].get("ppl_weight", 0.6)),
        svd_weight=float(cfg["rank_allocation"].get("svd_weight", 0.4)),
    )
    ranks = allocator.allocate(utility_curves, cluster_scores=cluster_scores)
    return ranks, sft_loss_values, svd_values, complexity


def train_experts_by_clusters(
    model,
    dataset,
    labels,
    cfg,
    device,
    logger,
    curve_tracker=None,
    global_step_state=None,
):
    k = int(labels.max()) + 1
    epochs = cfg["training"]["cluster_epochs"]
    trainer_backend = cfg["training"].get("trainer_backend", "custom").lower()
    curve_log_every = int(cfg["training"].get("curve_log_every", 20))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))

    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0].tolist()
        if not cluster_indices:
            continue

        set_trainable_expert_only(model, cluster_id)
        cluster_subset = Subset(dataset, cluster_indices)
        if trainer_backend == "trl":
            loss = run_trl_sft(
                model=model,
                dataset=cluster_subset,
                cfg=cfg,
                lr=cfg["training"]["lora_lr"],
                epochs=epochs,
                stage="stage1_expert",
                track=f"expert_{cluster_id}",
                logger=logger,
                curve_tracker=curve_tracker,
                global_step_state=global_step_state,
            )
            logger.info(f"[Stage-1][Expert {cluster_id}] trl_loss={loss:.4f}")
            if not np.isfinite(loss):
                logger.warning(f"[Stage-1][Expert {cluster_id}] TRL produced no finite loss.")
            continue

        optimizer = refresh_optimizer(model, lr=cfg["training"]["lora_lr"])
        loader = DataLoader(
            cluster_subset,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
        )
        scheduler = None
        if cfg["training"].get("use_scheduler", True):
            accum = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
            total_steps = math.ceil(len(loader) / accum) * max(1, epochs)
            scheduler = build_scheduler(
                optimizer,
                total_steps=total_steps,
                warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.06)),
            )

        for epoch in range(epochs):
            local_step = {"v": 0}

            def on_step_end(step_loss):
                local_step["v"] += 1
                if global_step_state is not None:
                    global_step_state["step"] += 1
                    gstep = global_step_state["step"]
                else:
                    gstep = local_step["v"]
                if curve_tracker is not None and gstep % max(1, curve_log_every) == 0:
                    curve_tracker.log(stage="stage1_expert", track=f"expert_{cluster_id}", step=gstep, loss=step_loss)

            loss = run_lm_epoch(
                model,
                optimizer,
                loader,
                device,
                on_step_end=on_step_end,
                grad_clip=grad_clip,
                scheduler=scheduler,
                grad_accum_steps=int(cfg["training"].get("gradient_accumulation_steps", 1)),
            )
            logger.info(f"[Stage-1][Expert {cluster_id}] epoch={epoch + 1} loss={loss:.4f}")
            if not np.isfinite(loss):
                logger.warning(f"[Stage-1][Expert {cluster_id}] epoch={epoch + 1} produced no finite steps.")

    clear_forced_experts(model)


def train_router(
    model,
    dataset,
    cfg,
    device,
    logger,
    curve_tracker=None,
    stage_tag="stage1",
    global_step_state=None,
):
    router_mode = cfg.get("model", {}).get("router_mode", "centroid").lower()
    if router_mode == "centroid":
        logger.info(f"[{stage_tag}][Router] centroid routing enabled; skip router training.")
        return

    trainer_backend = cfg["training"].get("trainer_backend", "custom").lower()
    set_trainable_router_only(model)
    if trainer_backend == "trl":
        loss = run_trl_sft(
            model=model,
            dataset=dataset,
            cfg=cfg,
            lr=cfg["training"]["router_lr"],
            epochs=cfg["training"]["router_epochs"],
            stage=stage_tag,
            track="router",
            logger=logger,
            curve_tracker=curve_tracker,
            global_step_state=global_step_state,
        )
        logger.info(f"[{stage_tag}][Router] trl_loss={loss:.4f}")
        if not np.isfinite(loss):
            logger.warning(f"[{stage_tag}][Router] TRL produced no finite loss.")
        return

    optimizer = refresh_optimizer(model, lr=cfg["training"]["router_lr"])
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], shuffle=True)
    curve_log_every = int(cfg["training"].get("curve_log_every", 20))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    scheduler = None
    if cfg["training"].get("use_scheduler", True):
        accum = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
        total_steps = math.ceil(len(loader) / accum) * max(1, cfg["training"]["router_epochs"])
        scheduler = build_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.06)),
        )

    for epoch in range(cfg["training"]["router_epochs"]):
        local_step = {"v": 0}

        def on_step_end(step_loss):
            local_step["v"] += 1
            if global_step_state is not None:
                global_step_state["step"] += 1
                gstep = global_step_state["step"]
            else:
                gstep = local_step["v"]
            if curve_tracker is not None and gstep % max(1, curve_log_every) == 0:
                curve_tracker.log(stage=stage_tag, track="router", step=gstep, loss=step_loss)

        loss = run_lm_epoch(
            model,
            optimizer,
            loader,
            device,
            on_step_end=on_step_end,
            grad_clip=grad_clip,
            scheduler=scheduler,
            grad_accum_steps=int(cfg["training"].get("gradient_accumulation_steps", 1)),
        )
        logger.info(f"[{stage_tag}][Router] epoch={epoch + 1} loss={loss:.4f}")
        if not np.isfinite(loss):
            logger.warning(f"[{stage_tag}][Router] epoch={epoch + 1} produced no finite steps.")


def rebuild_for_new_layout(model, new_ranks, inherit_map, device):
    for layer in get_moe_layers(model):
        layer.rebuild_experts(new_ranks=new_ranks, inherit_map=inherit_map)
        layer.to(device=device)


def stage3_demo_scheduler(layout: ClusterLayout, logger):
    scheduler = HeteroBatchScheduler(gpu_min_batch=4, gpu_min_tokens=256)
    requests = [
        InferenceRequest("req-1", activated_loras=(0,), token_length=64),
        InferenceRequest("req-2", activated_loras=(0,), token_length=72),
        InferenceRequest("req-3", activated_loras=(1,), token_length=40),
        InferenceRequest("req-4", activated_loras=(0,), token_length=80),
        InferenceRequest("req-5", activated_loras=(1,), token_length=32),
        InferenceRequest("req-6", activated_loras=(0,), token_length=64),
    ]

    # Keep signatures valid under current k.
    max_idx = max(layout.k - 1, 0)
    safe_requests = []
    for r in requests:
        sig = tuple(min(s, max_idx) for s in r.activated_loras)
        safe_requests.append(InferenceRequest(r.request_id, sig, r.token_length))

    plan = scheduler.schedule(safe_requests)
    logger.info(f"[Stage-3] GPU groups={len(plan['gpu'])}, CPU groups={len(plan['cpu'])}")


def main():
    cfg = load_config("config.yaml")
    logger = ExperimentLogger(exp_name="MoE_LoRA_ResearchPipeline")
    device = resolve_device(cfg)
    set_global_seed(int(cfg["data"].get("seed", 42)))

    logger.info("=== Starting 3-Stage Privacy MoE-LoRA Pipeline ===")
    logger.info(f"[Runtime] Using device={device}")
    curve_tracker = TrainingCurveTracker(
        out_dir=cfg["training"].get("curve_dir", "eval/figures"),
        run_name=cfg["training"].get("curve_run_name", "moe_lora"),
    )

    model_id = cfg["model"]["base_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    cfg["_tokenizer_obj"] = tokenizer
    global_step_state = {"step": 0}

    # Step 0: load base model (without LoRA injection) for feature extraction and cluster complexity estimation.
    base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    full_dataset = build_dataset(cfg, tokenizer=tokenizer, split="train")
    test_dataset = build_dataset(cfg, tokenizer=tokenizer, split="test")

    stage1_dataset = maybe_subset(full_dataset, cfg["data"].get("stage1_samples", 50000))
    logger.info(f"[Data] full={len(full_dataset)}, stage1_subset={len(stage1_dataset)}")

    logger.info("[Stage-1] Extracting private-data features...")
    feature_strategy = cfg["clustering"].get("feature_strategy", "sentence_mean")
    stage1_cache_path = build_feature_cache_path(cfg, split_name="stage1", dataset_len=len(stage1_dataset))
    features = collect_features_with_cache(
        base_model,
        stage1_dataset,
        batch_size=cfg["data"]["feature_batch_size"],
        device=device,
        feature_strategy=feature_strategy,
        cache_path=stage1_cache_path,
        logger=logger,
    )

    selector = DBCHKSelector(
        k_min=cfg["clustering"]["k_min"],
        k_max=cfg["clustering"]["k_max"],
        random_state=cfg["clustering"].get("random_state", 42),
    )
    best_k, labels, centroids = selector.select(features)
    logger.info(f"[Stage-1] DB/CH selected k={best_k}")
    viz_method = cfg["clustering"].get("visualization_method", "both")
    if viz_method != "none":
        stage1_max_points = (
            len(features)
            if cfg["clustering"].get("visualization_use_all_stage1", False)
            else cfg["clustering"].get("visualization_max_points", 3000)
        )
        paths = visualize_clusters(
            features,
            labels,
            out_dir=cfg["clustering"].get("visualization_dir", "eval/figures"),
            prefix="stage1",
            method=viz_method,
            max_points=stage1_max_points,
            random_state=cfg["clustering"].get("random_state", 42),
        )
        logger.info(f"[Stage-1] cluster viz saved: {paths}")

    ranks, sft_loss_vals, svd_vals, complexity = build_rank_layout(
        base_model, stage1_dataset, features, labels, cfg, device, tokenizer
    )
    logger.info(f"[Stage-1] per-cluster sft_loss={sft_loss_vals}")
    logger.info(f"[Stage-1] per-cluster svd_complexity={svd_vals}")
    logger.info(f"[Stage-1] complexity={complexity}")
    logger.info(f"[Stage-1] allocated ranks={ranks}")

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    inject_system_into_base_model(
        model,
        target_modules=cfg["model"]["target_modules"],
        initial_ranks=ranks,
        top_k=cfg["model"]["top_k"],
        alpha=cfg["model"]["alpha"],
        dropout=cfg["model"]["dropout"],
    )
    set_model_router_centroids(model, centroids)

    train_experts_by_clusters(
        model,
        stage1_dataset,
        labels,
        cfg,
        device,
        logger,
        curve_tracker=curve_tracker,
        global_step_state=global_step_state,
    )
    train_router(
        model,
        stage1_dataset,
        cfg,
        device,
        logger,
        curve_tracker=curve_tracker,
        stage_tag="stage1",
        global_step_state=global_step_state,
    )
    if cfg.get("evaluation", {}).get("enable", True):
        ranking_metrics = evaluate_recall_ndcg(
            model,
            test_dataset,
            batch_size=cfg["data"]["batch_size"],
            device=device,
            tokenizer=tokenizer,
            k=cfg.get("evaluation", {}).get("k", 10),
            max_samples=cfg.get("evaluation", {}).get("max_samples", 2000),
        )
        logger.info(f"[Stage-1] ranking_metrics={ranking_metrics}")
        gen_metrics = evaluate_generate_hit1(
            model,
            test_dataset,
            batch_size=cfg["data"]["batch_size"],
            device=device,
            tokenizer=tokenizer,
            max_samples=cfg.get("evaluation", {}).get("max_samples", 1000),
        )
        logger.info(f"[Stage-1] generation_metrics={gen_metrics}")
    save_model_checkpoint(
        model,
        cfg,
        stage_name="stage1",
        logger=logger,
        extra_meta={"k": best_k, "ranks": ranks},
    )

    current_layout = ClusterLayout(k=best_k, labels=labels, centroids=centroids, ranks=ranks)

    # Stage-2: simulate data drift with recent slice.
    logger.info("[Stage-2] Running incremental re-clustering and overlap-driven update...")
    drift_size = cfg["stage2"]["drift_sample_size"]
    drift_indices = list(range(max(0, len(full_dataset) - drift_size), len(full_dataset)))
    drift_dataset = Subset(full_dataset, drift_indices)

    drift_cache_path = build_feature_cache_path(cfg, split_name="stage2_drift", dataset_len=len(drift_dataset))
    drift_features = collect_features_with_cache(
        model,
        drift_dataset,
        batch_size=cfg["data"]["feature_batch_size"],
        device=device,
        feature_strategy=feature_strategy,
        cache_path=drift_cache_path,
        logger=logger,
    )

    new_k, new_labels, new_centroids = selector.select(drift_features)
    if viz_method != "none":
        paths = visualize_clusters(
            drift_features,
            new_labels,
            out_dir=cfg["clustering"].get("visualization_dir", "eval/figures"),
            prefix="stage2",
            method=viz_method,
            max_points=cfg["clustering"].get("visualization_max_points", 3000),
            random_state=cfg["clustering"].get("random_state", 42),
        )
        logger.info(f"[Stage-2] cluster viz saved: {paths}")

    # Build new rank needs using drift statistics.
    new_ranks, _, _, _ = build_rank_layout(
        model,
        drift_dataset,
        drift_features,
        new_labels,
        cfg,
        device,
        tokenizer,
    )

    overlap_manager = ClusterOverlapManager(
        high_overlap_threshold=cfg["stage2"]["high_overlap_threshold"],
        mid_overlap_threshold=cfg["stage2"]["mid_overlap_threshold"],
    )

    inherit_map, pairs, avg_overlap = overlap_manager.greedy_match(current_layout.centroids, new_centroids)
    strategy = overlap_manager.choose_strategy(avg_overlap, optimize_for=cfg["stage2"]["optimize_for"])

    logger.info(f"[Stage-2] overlap pairs={pairs}")
    logger.info(f"[Stage-2] avg_overlap={avg_overlap:.4f}, strategy={strategy}")

    if strategy == "full_retrain":
        inherit_map = {}

    rebuild_for_new_layout(model, new_ranks, inherit_map=inherit_map, device=device)
    set_model_router_centroids(model, new_centroids)
    train_router(
        model,
        drift_dataset,
        cfg,
        device,
        logger,
        curve_tracker=curve_tracker,
        stage_tag="stage2",
        global_step_state=global_step_state,
    )
    if cfg.get("evaluation", {}).get("enable", True):
        ranking_metrics = evaluate_recall_ndcg(
            model,
            test_dataset,
            batch_size=cfg["data"]["batch_size"],
            device=device,
            tokenizer=tokenizer,
            k=cfg.get("evaluation", {}).get("k", 10),
            max_samples=cfg.get("evaluation", {}).get("max_samples", 2000),
        )
        logger.info(f"[Stage-2] ranking_metrics={ranking_metrics}")
        gen_metrics = evaluate_generate_hit1(
            model,
            test_dataset,
            batch_size=cfg["data"]["batch_size"],
            device=device,
            tokenizer=tokenizer,
            max_samples=cfg.get("evaluation", {}).get("max_samples", 1000),
        )
        logger.info(f"[Stage-2] generation_metrics={gen_metrics}")
    save_model_checkpoint(
        model,
        cfg,
        stage_name="stage2",
        logger=logger,
        extra_meta={"k": new_k, "ranks": new_ranks, "avg_overlap": avg_overlap, "strategy": strategy},
    )

    # Stage-3: system-level scheduling demo.
    stage3_demo_scheduler(ClusterLayout(new_k, new_labels, new_centroids, new_ranks), logger)
    logger.info("=== Pipeline completed ===")


if __name__ == "__main__":
    main()
