import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from data.data_loader import build_dataset, extract_clustering_features
from eval.plot_run_summary import generate_summary_plot
from eval.plot_stage2_baseline_compare import generate_stage2_baseline_plot
from src.baselines import (
    continue_raie,
    continue_single_lora,
    evaluate_hydralora,
    evaluate_mocle,
    evaluate_raie,
    evaluate_raie_state,
    evaluate_raw_base,
    evaluate_single_lora,
    evaluate_stage1_moe,
    train_raie,
    train_hydralora,
    train_mocle,
    train_single_lora,
    train_stage1_moe,
)
from src.clustering.spherical_cluster import DBCHKSelector
from src.eval.task_eval import evaluate_model_for_task
from src.models.learned_moe_layer import LearnedMoELoRALinear
from src.models.moe_layer import MoELoRALinear
from src.pipeline.common import (
    build_feature_cache_path,
    build_stage2_adaptation_subset,
    clear_cuda_cache,
    clone_config_with_overrides,
    get_cfg_batch_size,
    get_dataset_base_indices,
    get_dataset_task_type,
    get_model_device,
    load_config,
    maybe_log_cluster_style_mix,
    maybe_subset,
    resolve_device,
    save_model_checkpoint,
    set_global_seed,
    split_historical_and_new,
)
from src.system.hetero_batcher import HeteroBatchScheduler, InferenceRequest
from src.system.overlap_manager import ClusterOverlapManager
from src.system.rank_allocator import BudgetedRankAllocator
from src.training.sft_utils import build_baseline_sft_trainer as _build_baseline_sft_trainer
from src.training.sft_utils import run_trl_sft, rows_from_dataset_for_sft
from src.utils.cluster_viz import visualize_clusters
from src.utils.logger import ExperimentLogger
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




def run_stage0_cloud_pretraining(cfg, device, logger):
    stage0_cfg = cfg.get("stage0", {})
    if not stage0_cfg.get("enable", False):
        return cfg["model"]["base_model_path"]

    merged_out = stage0_cfg.get("merged_output_dir", "outputs/stage0_cloud_merged")
    adapter_out = stage0_cfg.get("adapter_output_dir", "outputs/stage0_cloud_adapter")
    reuse_existing = bool(stage0_cfg.get("reuse_existing", True))

    if reuse_existing and os.path.isdir(merged_out) and os.path.exists(os.path.join(merged_out, "config.json")):
        logger.info(f"[Stage-0] reuse merged warm base: {merged_out}")
        return merged_out

    model_id = cfg["model"]["base_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    stage0_data_cfg = {
        "dataset_type": stage0_cfg.get("dataset_type", "composite_device"),
        "data_dir": stage0_cfg.get("data_dir", "data/yelp_cloud_pretrain"),
        "max_seq_len": cfg["data"]["max_seq_len"],
        "history_size": cfg["data"].get("history_size", 10),
        "min_history": cfg["data"].get("min_history", 3),
        "min_user_samples": cfg["data"].get("min_user_samples", 3),
        "neg_sample_size": cfg["data"].get("neg_sample_size", 3),
        "seed": cfg["data"].get("seed", 42),
    }
    stage0_dataset_cfg = clone_config_with_overrides(cfg, {"data": stage0_data_cfg})
    if not os.path.isdir(stage0_data_cfg["data_dir"]):
        raise FileNotFoundError(
            f"Stage-0 data dir not found: {stage0_data_cfg['data_dir']}. "
            "Build it first with data/build_yelp_cloud_pretrain.py or disable stage0.enable."
        )
    train_dataset = build_dataset(stage0_dataset_cfg, tokenizer=tokenizer, split="train")
    eval_dataset = build_dataset(stage0_dataset_cfg, tokenizer=tokenizer, split="val")

    max_samples = int(stage0_cfg.get("max_samples", 0) or 0)
    if max_samples > 0:
        train_dataset = maybe_subset(train_dataset, max_samples)

    rows = rows_from_dataset_for_sft(train_dataset)
    if len(rows) == 0:
        raise ValueError("Stage-0 pretraining dataset is empty after sampling.")
    train_rows = Dataset.from_list(rows)

    eval_rows = None
    eval_limit = int(stage0_cfg.get("eval_max_samples", 1000) or 0)
    if len(eval_dataset) > 0 and eval_limit != 0:
        eval_base = maybe_subset(eval_dataset, eval_limit) if eval_limit > 0 else eval_dataset
        eval_rows = Dataset.from_list(rows_from_dataset_for_sft(eval_base))

    logger.info(
        f"[Stage-0] cloud pretraining on {stage0_data_cfg['data_dir']}: "
        f"train={len(train_rows)}, eval={len(eval_rows) if eval_rows is not None else 0}"
    )

    use_bf16 = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if use_bf16 else torch.float32).to(device)

    train_cfg = SFTConfig(
        output_dir=adapter_out,
        per_device_train_batch_size=max(1, int(stage0_cfg.get("per_device_batch_size", 4))),
        per_device_eval_batch_size=max(1, int(stage0_cfg.get("per_device_batch_size", 4))),
        gradient_accumulation_steps=max(1, int(stage0_cfg.get("gradient_accumulation_steps", 4))),
        learning_rate=float(stage0_cfg.get("learning_rate", 2e-4)),
        num_train_epochs=max(1, int(stage0_cfg.get("epochs", 1))),
        logging_steps=max(1, int(cfg["training"].get("curve_log_every", 20))),
        report_to="none",
        bf16=use_bf16,
        fp16=False,
        max_length=int(cfg["data"]["max_seq_len"]),
        gradient_checkpointing=bool(cfg["training"].get("trl_gradient_checkpointing", True)),
        dataloader_num_workers=int(cfg["training"].get("trl_dataloader_num_workers", 0)),
        save_strategy="epoch",
        eval_strategy="no" if eval_rows is None else "epoch",
    )

    trainer_kwargs = {
        "model": model,
        "args": train_cfg,
        "train_dataset": train_rows,
        "processing_class": tokenizer,
    }
    if eval_rows is not None:
        trainer_kwargs["eval_dataset"] = eval_rows
    if stage0_cfg.get("use_peft", True):
        trainer_kwargs["peft_config"] = LoraConfig(
            r=max(4, int(cfg["rank_allocation"].get("min_rank", 1)) * 4),
            lora_alpha=int(cfg["model"].get("alpha", 16)),
            lora_dropout=float(cfg["model"].get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg["model"]["target_modules"],
        )

    trainer = SFTTrainer(**trainer_kwargs)
    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False
    trainer.train()
    if old_use_cache is not None:
        model.config.use_cache = old_use_cache

    os.makedirs(adapter_out, exist_ok=True)
    trainer.save_model(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    logger.info(f"[Stage-0] adapter saved: {adapter_out}")

    clear_cuda_cache()

    if stage0_cfg.get("use_peft", True):
        base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if use_bf16 else torch.float32).to(device)
        peft_model = PeftModel.from_pretrained(base_model, adapter_out)
        merged_model = peft_model.merge_and_unload()
        os.makedirs(merged_out, exist_ok=True)
        merged_model.save_pretrained(merged_out)
        tokenizer.save_pretrained(merged_out)
        del peft_model
        del base_model
        del merged_model
    else:
        os.makedirs(merged_out, exist_ok=True)
        trainer.model.save_pretrained(merged_out)
        tokenizer.save_pretrained(merged_out)

    del trainer
    del model
    clear_cuda_cache()
    logger.info(f"[Stage-0] merged warm base saved: {merged_out}")
    return merged_out




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


def inject_learned_moe_system(model, target_modules, initial_ranks, top_k=1, alpha=16, dropout=0.05):
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, torch.nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)

            moe_lora_layer = LearnedMoELoRALinear(
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


def get_learned_moe_layers(model):
    return [m for m in model.modules() if isinstance(m, LearnedMoELoRALinear)]


def set_trainable_expert_only(model, expert_idx: int):
    for p in model.parameters():
        p.requires_grad = False

    for layer in get_moe_layers(model):
        layer.set_force_expert(expert_idx)
        for p in layer.experts[expert_idx].parameters():
            p.requires_grad = True


def clear_forced_experts(model):
    for layer in get_moe_layers(model):
        layer.clear_force_expert()


def set_trainable_learned_moe(model):
    for p in model.parameters():
        p.requires_grad = False

    for layer in get_learned_moe_layers(model):
        for p in layer.router.parameters():
            p.requires_grad = True
        for expert in layer.experts:
            for p in expert.parameters():
                p.requires_grad = True


@torch.no_grad()
def set_model_router_centroids(model, centroids):
    for layer in get_moe_layers(model):
        layer.set_router_centroids(centroids)


@torch.no_grad()
def collect_features(model, dataset, batch_size, device, feature_strategy="sentence_mean"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    model_device = get_model_device(model)

    feats = []
    for batch in tqdm(loader, desc="Feature Extract", leave=False):
        inputs = {
            "input_ids": batch["input_ids"].to(model_device),
            "attention_mask": batch["attention_mask"].to(model_device),
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


def _load_causal_lm(model_id, device):
    use_bf16 = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)


def train_single_lora_baseline_model(train_dataset, cfg, device, logger, tokenizer, model_id):
    return train_single_lora(
        train_dataset,
        cfg,
        device,
        logger,
        tokenizer,
        model_id,
        load_causal_lm=_load_causal_lm,
        build_baseline_sft_trainer=_build_baseline_sft_trainer,
        clear_cuda_cache=clear_cuda_cache,
    )


def evaluate_single_lora_baseline(train_dataset, test_dataset, cfg, device, logger, tokenizer, model_id):
    return evaluate_single_lora(
        train_dataset,
        test_dataset,
        cfg,
        device,
        logger,
        tokenizer,
        model_id,
        load_causal_lm=_load_causal_lm,
        build_baseline_sft_trainer=_build_baseline_sft_trainer,
        clear_cuda_cache=clear_cuda_cache,
        evaluate_model_for_task=evaluate_model_for_task,
    )


def continue_single_lora_baseline_model(model, train_dataset, cfg, device, logger, tokenizer):
    return continue_single_lora(
        model,
        train_dataset,
        cfg,
        device,
        logger,
        tokenizer,
        build_baseline_sft_trainer=_build_baseline_sft_trainer,
        clear_cuda_cache=clear_cuda_cache,
        out_dir_suffix="stage2",
    )


def train_mocle_baseline_model(train_dataset, features, cfg, device, logger, model_id):
    return train_mocle(
        train_dataset,
        features,
        cfg,
        device,
        logger,
        model_id,
        load_causal_lm=_load_causal_lm,
        inject_system_into_base_model=inject_system_into_base_model,
        set_model_router_centroids=set_model_router_centroids,
        train_experts_by_clusters=train_experts_by_clusters,
        clear_cuda_cache=clear_cuda_cache,
    )


def evaluate_mocle_baseline(train_dataset, test_dataset, features, cfg, device, logger, tokenizer, model_id):
    return evaluate_mocle(
        train_dataset,
        test_dataset,
        features,
        cfg,
        device,
        logger,
        tokenizer,
        model_id,
        load_causal_lm=_load_causal_lm,
        inject_system_into_base_model=inject_system_into_base_model,
        set_model_router_centroids=set_model_router_centroids,
        train_experts_by_clusters=train_experts_by_clusters,
        clear_cuda_cache=clear_cuda_cache,
        evaluate_model_for_task=evaluate_model_for_task,
    )


def train_raie_baseline_model(train_dataset, features, cfg, device, logger, tokenizer, model_id):
    return train_raie(
        train_dataset,
        features,
        cfg,
        device,
        logger,
        tokenizer,
        model_id,
        load_causal_lm=_load_causal_lm,
        build_baseline_sft_trainer=_build_baseline_sft_trainer,
        clear_cuda_cache=clear_cuda_cache,
    )


def evaluate_raie_baseline(train_dataset, test_dataset, features, cfg, device, logger, tokenizer, model_id):
    return evaluate_raie(
        train_dataset,
        test_dataset,
        features,
        cfg,
        device,
        logger,
        tokenizer,
        model_id,
        load_causal_lm=_load_causal_lm,
        build_baseline_sft_trainer=_build_baseline_sft_trainer,
        clear_cuda_cache=clear_cuda_cache,
        collect_features_with_cache=collect_features_with_cache,
        build_feature_cache_path=build_feature_cache_path,
        evaluate_model_for_task_fn=evaluate_model_for_task,
    )


def continue_raie_baseline_model(
    state,
    adaptation_dataset,
    adaptation_labels,
    new_centroids,
    cfg,
    device,
    logger,
    tokenizer,
    inherit_map=None,
):
    return continue_raie(
        state,
        adaptation_dataset,
        adaptation_labels,
        new_centroids,
        cfg,
        device,
        logger,
        tokenizer,
        inherit_map=inherit_map,
        build_baseline_sft_trainer=_build_baseline_sft_trainer,
        clear_cuda_cache=clear_cuda_cache,
    )


def evaluate_raie_baseline_state(state, test_dataset, cfg, device, logger, tokenizer, eval_tag="raie", cache_suffix="stage1"):
    return evaluate_raie_state(
        state,
        test_dataset,
        cfg,
        device,
        logger,
        tokenizer,
        load_causal_lm=_load_causal_lm,
        collect_features_with_cache=collect_features_with_cache,
        build_feature_cache_path=build_feature_cache_path,
        clear_cuda_cache=clear_cuda_cache,
        evaluate_model_for_task_fn=evaluate_model_for_task,
        eval_tag=eval_tag,
        cache_suffix=cache_suffix,
    )


def train_hydralora_baseline_model(train_dataset, cfg, device, logger, tokenizer, model_id):
    return train_hydralora(
        train_dataset,
        cfg,
        device,
        logger,
        tokenizer,
        model_id,
        load_causal_lm=_load_causal_lm,
        inject_learned_moe_system=inject_learned_moe_system,
        set_trainable_learned_moe=set_trainable_learned_moe,
        build_baseline_sft_trainer=_build_baseline_sft_trainer,
        clear_cuda_cache=clear_cuda_cache,
    )


def evaluate_hydralora_baseline(train_dataset, test_dataset, cfg, device, logger, tokenizer, model_id):
    return evaluate_hydralora(
        train_dataset,
        test_dataset,
        cfg,
        device,
        logger,
        tokenizer,
        model_id,
        load_causal_lm=_load_causal_lm,
        inject_learned_moe_system=inject_learned_moe_system,
        set_trainable_learned_moe=set_trainable_learned_moe,
        build_baseline_sft_trainer=_build_baseline_sft_trainer,
        clear_cuda_cache=clear_cuda_cache,
        evaluate_model_for_task=evaluate_model_for_task,
    )


def train_moe_lora_stage1_model(train_dataset, features, cfg, device, logger, tokenizer, model_id):
    del tokenizer
    return train_stage1_moe(
        train_dataset,
        features,
        cfg,
        device,
        logger,
        model_id,
        load_causal_lm=_load_causal_lm,
        build_rank_layout=build_rank_layout,
        inject_system_into_base_model=inject_system_into_base_model,
        set_model_router_centroids=set_model_router_centroids,
        train_experts_by_clusters=train_experts_by_clusters,
        clear_cuda_cache=clear_cuda_cache,
    )


def evaluate_moe_lora_stage1_method(train_dataset, test_dataset, features, cfg, device, logger, tokenizer, model_id):
    return evaluate_stage1_moe(
        train_dataset,
        test_dataset,
        features,
        cfg,
        device,
        logger,
        tokenizer,
        model_id,
        load_causal_lm=_load_causal_lm,
        build_rank_layout=build_rank_layout,
        inject_system_into_base_model=inject_system_into_base_model,
        set_model_router_centroids=set_model_router_centroids,
        train_experts_by_clusters=train_experts_by_clusters,
        clear_cuda_cache=clear_cuda_cache,
        evaluate_model_for_task=evaluate_model_for_task,
    )


def evaluate_stage1_baselines(cfg, device, logger, train_dataset, test_dataset, features, tokenizer):
    if not cfg.get("evaluation", {}).get("compare_reference_models", True):
        return

    raw_model_id = cfg["model"]["base_model_path"]
    train_base_model_id = cfg.get("_personalization_model_id", raw_model_id)

    raw_model, _ = evaluate_raw_base(
        cfg,
        device,
        AutoTokenizer.from_pretrained,
        build_dataset,
        _load_causal_lm,
        evaluate_model_for_task,
        logger,
    )
    del raw_model
    clear_cuda_cache()

    evaluate_single_lora_baseline(train_dataset, test_dataset, cfg, device, logger, tokenizer, train_base_model_id)
    evaluate_mocle_baseline(train_dataset, test_dataset, features, cfg, device, logger, tokenizer, train_base_model_id)
    evaluate_hydralora_baseline(train_dataset, test_dataset, cfg, device, logger, tokenizer, train_base_model_id)
    evaluate_raie_baseline(train_dataset, test_dataset, features, cfg, device, logger, tokenizer, train_base_model_id)


def evaluate_stage2_baselines(
    cfg,
    device,
    logger,
    *,
    stage1_dataset,
    stage1_features,
    stage2_dataset,
    stage2_test_dataset,
    tokenizer,
    stage2_adaptation_dataset,
    stage2_adaptation_indices,
):
    """
    评估第二阶段参考基线。
    Evaluate Stage-2 reference baselines.
    """
    if not cfg.get("evaluation", {}).get("compare_reference_models", True):
        return

    raw_model_id = cfg["model"]["base_model_path"]
    train_base_model_id = cfg.get("_personalization_model_id", raw_model_id)

    raw_model = _load_causal_lm(raw_model_id, device)
    raw_metrics = evaluate_model_for_task(raw_model, stage2_test_dataset, cfg, device, tokenizer)
    logger.info(f"[Baseline][stage2_raw_base] {raw_metrics}")
    del raw_model
    clear_cuda_cache()

    single_model = train_single_lora_baseline_model(stage1_dataset, cfg, device, logger, tokenizer, train_base_model_id)
    single_model = continue_single_lora_baseline_model(single_model, stage2_adaptation_dataset, cfg, device, logger, tokenizer)
    single_metrics = evaluate_model_for_task(single_model, stage2_test_dataset, cfg, device, tokenizer)
    logger.info(f"[Baseline][stage2_single_lora_r32] {single_metrics}")
    del single_model
    clear_cuda_cache()

    raie_state = train_raie_baseline_model(stage1_dataset, stage1_features, cfg, device, logger, tokenizer, train_base_model_id)
    raie_feature_model = _load_causal_lm(train_base_model_id, device)
    raie_stage2_cache = build_feature_cache_path(cfg, split_name="baseline_raie_stage2_full", dataset_len=len(stage2_dataset))
    raie_stage2_features = collect_features_with_cache(
        raie_feature_model,
        stage2_dataset,
        batch_size=cfg["data"]["feature_batch_size"],
        device=device,
        feature_strategy=cfg["clustering"].get("feature_strategy", "sentence_mean"),
        cache_path=raie_stage2_cache,
        logger=logger,
    )
    del raie_feature_model
    clear_cuda_cache()
    selector = DBCHKSelector(
        k_min=cfg["clustering"]["k_min"],
        k_max=cfg["clustering"]["k_max"],
        random_state=cfg["clustering"].get("random_state", 42),
    )
    _, raie_stage2_labels_all, raie_stage2_centroids = selector.select(raie_stage2_features)
    raie_stage2_labels = np.asarray(raie_stage2_labels_all)[stage2_adaptation_indices]
    raie_overlap = ClusterOverlapManager(
        high_overlap_threshold=cfg["stage2"]["high_overlap_threshold"],
        mid_overlap_threshold=cfg["stage2"]["mid_overlap_threshold"],
    )
    raie_inherit_map, _ = raie_overlap.nearest_parent_map(raie_state.centroids, raie_stage2_centroids)
    raie_state = continue_raie_baseline_model(
        raie_state,
        stage2_adaptation_dataset,
        raie_stage2_labels,
        raie_stage2_centroids,
        cfg,
        device,
        logger,
        tokenizer,
        inherit_map=raie_inherit_map,
    )
    evaluate_raie_baseline_state(
        raie_state,
        stage2_test_dataset,
        cfg,
        device,
        logger,
        tokenizer,
        eval_tag="stage2_raie",
        cache_suffix="stage2",
    )
    del raie_state.model
    clear_cuda_cache()


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
    返回逐 rank 的谱能量占比。
    Return per-rank spectral energy shares.

    gain[r] 对应新增第 (r+1) 个 rank 分量时的边际收益。
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


def normalize_cluster_scores(values):
    x = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.ones_like(x, dtype=np.float32)
    fill = float(np.nanmean(x[finite]))
    x = np.where(finite, x, fill).astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if abs(mx - mn) < 1e-8:
        return np.ones_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def build_cluster_rank_utility_curves(
    svd_curves,
    min_rank: int,
    max_rank: int,
    svd_weight: float,
):
    svd_curves = [np.asarray(curve, dtype=np.float32) for curve in svd_curves]

    if len(svd_curves) == 0:
        return [], []

    utility_curves = []
    structure_scores = []
    extra_slots = max(0, max_rank - min_rank)

    for cluster_id, svd_curve in enumerate(svd_curves):
        if extra_slots == 0:
            utility_curves.append(np.zeros(0, dtype=np.float32))
            structure_scores.append(float(svd_curve[:max_rank].sum()))
            continue

        tail = svd_curve[min_rank:max_rank]
        if tail.shape[0] < extra_slots:
            tail = np.pad(tail, (0, extra_slots - tail.shape[0]))

        marginal = svd_weight * tail
        residual = float(tail.sum()) / max(1, extra_slots)
        marginal = marginal + residual * 1e-3

        utility_curves.append(marginal.astype(np.float32))
        structure_scores.append(float(svd_curve[:max_rank].sum()))

    return utility_curves, normalize_cluster_scores(structure_scores).tolist()


def build_rank_layout(features, labels, cfg):
    k = int(labels.max()) + 1
    svd_values = []
    svd_curves = []

    for cluster_id in range(k):
        idx = np.where(labels == cluster_id)[0].tolist()
        print(f"[RankLayout] cluster={cluster_id}, samples={len(idx)}")
        cluster_features = features[idx]
        svd_values.append(svd_complexity(cluster_features))
        svd_curves.append(
            svd_energy_curve(
                cluster_features,
                max_rank=int(cfg["rank_allocation"]["max_rank"]),
            )
        )
        clear_cuda_cache()

    complexity = normalize_cluster_scores(svd_values).tolist()

    allocator = BudgetedRankAllocator(
        total_rank_budget=cfg["rank_allocation"]["total_budget"],
        min_rank=cfg["rank_allocation"]["min_rank"],
        max_rank=cfg["rank_allocation"]["max_rank"],
    )
    utility_curves, cluster_scores = build_cluster_rank_utility_curves(
        svd_curves=svd_curves,
        min_rank=int(cfg["rank_allocation"]["min_rank"]),
        max_rank=int(cfg["rank_allocation"]["max_rank"]),
        svd_weight=float(cfg["rank_allocation"].get("svd_weight", 1.0)),
    )
    ranks = allocator.allocate(utility_curves, cluster_scores=cluster_scores)
    return ranks, svd_values, complexity


def train_experts_by_clusters(
    model,
    dataset,
    labels,
    cfg,
    device,
    logger,
    stage_tag="stage1",
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
                stage=f"{stage_tag}_expert",
                track=f"expert_{cluster_id}",
                logger=logger,
                target_device=device,
                curve_tracker=curve_tracker,
                global_step_state=global_step_state,
            )
            logger.info(f"[{stage_tag}][Expert {cluster_id}] trl_loss={loss:.4f}")
            if not np.isfinite(loss):
                logger.warning(f"[{stage_tag}][Expert {cluster_id}] TRL produced no finite loss.")
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
                    curve_tracker.log(stage=f"{stage_tag}_expert", track=f"expert_{cluster_id}", step=gstep, loss=step_loss)

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
            logger.info(f"[{stage_tag}][Expert {cluster_id}] epoch={epoch + 1} loss={loss:.4f}")
            if not np.isfinite(loss):
                logger.warning(f"[{stage_tag}][Expert {cluster_id}] epoch={epoch + 1} produced no finite steps.")

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
    del model, dataset, cfg, device, curve_tracker, global_step_state
    logger.info(f"[{stage_tag}][Router] centroid routing is deterministic; skip router training.")


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

    # 保证签名索引在当前 k 范围内有效。
    # Keep signature indices valid under the current k.
    max_idx = max(layout.k - 1, 0)
    safe_requests = []
    for r in requests:
        sig = tuple(min(s, max_idx) for s in r.activated_loras)
        safe_requests.append(InferenceRequest(r.request_id, sig, r.token_length))

    plan = scheduler.schedule(safe_requests)
    logger.info(f"[Stage-3] GPU groups={len(plan['gpu'])}, CPU groups={len(plan['cpu'])}")


def generate_auto_plots(logger):
    """
    基于当前运行日志自动生成关键对比图。
    Automatically generate the key comparison figures from the current run log.
    """
    try:
        stage1_summary = generate_summary_plot(logger.log_file, stage="stage1")
        logger.info(f"[Plot] stage1_summary={stage1_summary['output_path']}")
    except Exception as exc:
        logger.warning(f"[Plot] failed to generate stage1 summary plot: {exc}")

    try:
        stage2_compare = generate_stage2_baseline_plot(logger.log_file)
        logger.info(f"[Plot] stage2_compare={stage2_compare['output_path']}")
    except Exception as exc:
        logger.warning(f"[Plot] failed to generate stage2 baseline plot: {exc}")


def parse_args():
    """Parse command-line arguments. / 解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run the 3-stage MoE-LoRA research pipeline.")
    parser.add_argument(
        "--config",
        default="config_unified.yaml",
        help="Path to the YAML config file. / YAML 配置文件路径。",
    )
    parser.add_argument(
        "--profile",
        default="",
        help="Optional profile inside a unified config file. / 综合配置文件中的可选 profile。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, profile=args.profile or None)
    logger = ExperimentLogger(exp_name="MoE_LoRA_ResearchPipeline")
    device = resolve_device(cfg)
    set_global_seed(int(cfg["data"].get("seed", 42)))

    logger.info("=== Starting 3-Stage Privacy MoE-LoRA Pipeline ===")
    logger.info(f"[Runtime] config={args.config}")
    if cfg.get("_profile"):
        logger.info(f"[Runtime] profile={cfg['_profile']}")
    logger.info(f"[Runtime] Using device={device}")
    curve_tracker = TrainingCurveTracker(
        out_dir=cfg["training"].get("curve_dir", "eval/figures"),
        run_name=cfg["training"].get("curve_run_name", "moe_lora"),
    )

    model_id = run_stage0_cloud_pretraining(cfg, device=device, logger=logger)
    cfg["_personalization_model_id"] = model_id
    logger.info(f"[Model] personalization base={model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    cfg["_tokenizer_obj"] = tokenizer
    global_step_state = {"step": 0}

    # 第 0 步：加载未注入 LoRA 的基础模型，用于特征提取和簇复杂度估计。
    # Step 0: load the base model without LoRA injection for feature extraction and cluster-complexity estimation.
    base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    full_dataset = build_dataset(cfg, tokenizer=tokenizer, split="train")
    full_test_dataset = build_dataset(cfg, tokenizer=tokenizer, split="test")
    task_type = get_dataset_task_type(full_dataset)

    historical_dataset, combined_stage2_dataset, new_data_indices = split_historical_and_new(
        full_dataset,
        cfg["stage2"].get("drift_sample_size", 1000),
    )
    historical_test_dataset, stage2_test_dataset, stage2_test_new_indices = split_historical_and_new(
        full_test_dataset,
        cfg["stage2"].get("drift_sample_size", 1000),
    )
    stage1_dataset = maybe_subset(historical_dataset, cfg["data"].get("stage1_samples", 50000))
    logger.info(
        f"[Data] task_type={task_type}, full={len(full_dataset)}, historical={len(historical_dataset)}, "
        f"new_arrivals={len(new_data_indices)}, stage1_subset={len(stage1_dataset)}"
    )
    logger.info(
        f"[EvalData] stage1_test={len(historical_test_dataset)}, "
        f"stage2_test_full={len(stage2_test_dataset)}, stage2_test_new={len(stage2_test_new_indices)}"
    )

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
    clear_cuda_cache()

    selector = DBCHKSelector(
        k_min=cfg["clustering"]["k_min"],
        k_max=cfg["clustering"]["k_max"],
        random_state=cfg["clustering"].get("random_state", 42),
    )
    best_k, labels, centroids = selector.select(features)
    logger.info(f"[Stage-1] DB/CH selected k={best_k}")
    maybe_log_cluster_style_mix(cfg, logger, stage1_dataset, labels, task_type, stage_tag="Stage-1")
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

    ranks, svd_vals, complexity = build_rank_layout(features, labels, cfg)
    logger.info(f"[Stage-1] per-cluster svd_complexity={svd_vals}")
    logger.info(f"[Stage-1] structure_complexity={complexity}")
    logger.info(f"[Stage-1] allocated ranks={ranks}")
    del base_model
    clear_cuda_cache()

    evaluate_stage1_baselines(
        cfg,
        device=device,
        logger=logger,
        train_dataset=stage1_dataset,
        test_dataset=historical_test_dataset,
        features=features,
        tokenizer=tokenizer,
    )

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
        stage_tag="stage1",
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
        stage1_metrics = evaluate_model_for_task(model, historical_test_dataset, cfg, device, tokenizer)
        logger.info(f"[Stage-1] {stage1_metrics}")
        clear_cuda_cache()
    save_model_checkpoint(
        model,
        cfg,
        stage_name="stage1",
        logger=logger,
        extra_meta={"k": best_k, "ranks": ranks},
    )

    current_layout = ClusterLayout(k=best_k, labels=labels, centroids=centroids, ranks=ranks)

    # Stage-2：历史私有数据和新到达数据切片共同定义新的布局。
    # Stage-2: historical private data and the newly arrived slice jointly define the new layout.
    logger.info("[Stage-2] Running incremental re-clustering and overlap-driven update...")
    stage2_dataset = combined_stage2_dataset
    new_arrival_dataset = Subset(full_dataset, new_data_indices)
    logger.info(
        f"[Stage-2] recluster_on=current_private={len(stage2_dataset)}, "
        f"new_arrivals_only={len(new_arrival_dataset)}"
    )

    stage2_cache_path = build_feature_cache_path(cfg, split_name="stage2_full", dataset_len=len(stage2_dataset))
    stage2_features = collect_features_with_cache(
        model,
        stage2_dataset,
        batch_size=cfg["data"]["feature_batch_size"],
        device=device,
        feature_strategy=feature_strategy,
        cache_path=stage2_cache_path,
        logger=logger,
    )
    clear_cuda_cache()

    new_k, new_labels, new_centroids = selector.select(stage2_features)
    maybe_log_cluster_style_mix(cfg, logger, stage2_dataset, new_labels, task_type, stage_tag="Stage-2")
    if viz_method != "none":
        paths = visualize_clusters(
            stage2_features,
            new_labels,
            out_dir=cfg["clustering"].get("visualization_dir", "eval/figures"),
            prefix="stage2",
            method=viz_method,
            max_points=cfg["clustering"].get("visualization_max_points", 3000),
            random_state=cfg["clustering"].get("random_state", 42),
            reference_features=features,
            reference_labels=labels,
        )
        logger.info(f"[Stage-2] cluster viz saved: {paths}")

    # 基于漂移后的统计重新估计新的 rank 需求。
    # Re-estimate new rank requirements using drift-aware statistics.
    new_ranks, _, _ = build_rank_layout(stage2_features, new_labels, cfg)

    overlap_manager = ClusterOverlapManager(
        high_overlap_threshold=cfg["stage2"]["high_overlap_threshold"],
        mid_overlap_threshold=cfg["stage2"]["mid_overlap_threshold"],
    )

    inherit_map_unique, pairs, avg_overlap = overlap_manager.greedy_match(current_layout.centroids, new_centroids)
    inherit_map_nearest, nearest_pairs = overlap_manager.nearest_parent_map(current_layout.centroids, new_centroids)
    strategy = overlap_manager.choose_strategy(avg_overlap, optimize_for=cfg["stage2"]["optimize_for"])

    logger.info(f"[Stage-2] overlap pairs={pairs}")
    logger.info(f"[Stage-2] nearest inherit pairs={nearest_pairs}")
    logger.info(f"[Stage-2] avg_overlap={avg_overlap:.4f}, strategy={strategy}")

    if strategy == "full_retrain":
        inherit_map = {}
    else:
        inherit_map = inherit_map_nearest

    stage2_adaptation_dataset, stage2_adaptation_indices, replay_meta = build_stage2_adaptation_subset(
        current_private_dataset=stage2_dataset,
        new_arrival_indices=new_data_indices,
        historical_stage1_dataset=stage1_dataset,
        historical_labels=labels,
        cfg=cfg,
    )
    stage2_adaptation_labels = np.asarray(new_labels)[stage2_adaptation_indices]
    logger.info(f"[Stage-2] adaptation_data={replay_meta}")

    evaluate_stage2_baselines(
        cfg,
        device,
        logger,
        stage1_dataset=stage1_dataset,
        stage1_features=features,
        stage2_dataset=stage2_dataset,
        stage2_test_dataset=stage2_test_dataset,
        tokenizer=tokenizer,
        stage2_adaptation_dataset=stage2_adaptation_dataset,
        stage2_adaptation_indices=stage2_adaptation_indices,
    )

    rebuild_for_new_layout(model, new_ranks, inherit_map=inherit_map, device=device)
    set_model_router_centroids(model, new_centroids)
    train_experts_by_clusters(
        model,
        stage2_adaptation_dataset,
        stage2_adaptation_labels,
        cfg,
        device,
        logger,
        stage_tag="stage2",
        curve_tracker=curve_tracker,
        global_step_state=global_step_state,
    )
    train_router(
        model,
        stage2_adaptation_dataset,
        cfg,
        device,
        logger,
        curve_tracker=curve_tracker,
        stage_tag="stage2",
        global_step_state=global_step_state,
    )
    if cfg.get("evaluation", {}).get("enable", True):
        stage2_metrics = evaluate_model_for_task(model, stage2_test_dataset, cfg, device, tokenizer)
        logger.info(f"[Stage-2] {stage2_metrics}")
        clear_cuda_cache()
    save_model_checkpoint(
        model,
        cfg,
        stage_name="stage2",
        logger=logger,
        extra_meta={"k": new_k, "ranks": new_ranks, "avg_overlap": avg_overlap, "strategy": strategy},
    )

    # Stage-3：系统级调度演示。
    # Stage-3: system-level scheduling demo.
    stage3_demo_scheduler(ClusterLayout(new_k, new_labels, new_centroids, new_ranks), logger)
    generate_auto_plots(logger)
    logger.info("=== Pipeline completed ===")


if __name__ == "__main__":
    main()
