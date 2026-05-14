import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from peft import LoraConfig, get_peft_model
from torch.utils.data import Subset

from src.clustering.spherical_cluster import DBCHKSelector
from src.eval.task_eval import evaluate_model_for_task
from src.pipeline.common import clone_config_with_overrides, maybe_subset


@dataclass
class RAIEState:
    """
    RAIE baseline 的运行时状态。
    Runtime state for the RAIE baseline.
    """

    model: object
    centroids: np.ndarray
    adapter_names: List[str]
    ranks: List[int]
    routing_model_id: str


def _nearest_centroid_labels(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    基于余弦相似度分配最近区域标签。
    Assign the nearest region label using cosine similarity.
    """
    x = np.asarray(features, dtype=np.float32)
    c = np.asarray(centroids, dtype=np.float32)
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    c = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-8, None)
    return np.argmax(x @ c.T, axis=1).astype(int)


def _equal_rank_layout(k: int, cfg) -> List[int]:
    """
    在固定总预算下均匀分配 rank。
    Evenly distribute rank under a fixed total budget.
    """
    total_budget = int(cfg["rank_allocation"]["total_budget"])
    min_rank = int(cfg["rank_allocation"]["min_rank"])
    max_rank = int(cfg["rank_allocation"]["max_rank"])
    if k <= 0:
        return []

    base = [min_rank] * k
    current = sum(base)
    if current >= total_budget:
        per = max(1, total_budget // k)
        ranks = [per] * k
        for i in range(total_budget - per * k):
            ranks[i % k] += 1
        return [min(max_rank, r) for r in ranks]

    remain = total_budget - current
    idx = 0
    while remain > 0:
        if base[idx] < max_rank:
            base[idx] += 1
            remain -= 1
        idx = (idx + 1) % k
        if idx == 0 and all(r >= max_rank for r in base):
            break
    return base


def _build_lora_config(rank: int, cfg) -> LoraConfig:
    return LoraConfig(
        r=int(rank),
        lora_alpha=int(cfg["model"].get("alpha", 16)),
        lora_dropout=float(cfg["model"].get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["model"]["target_modules"],
    )


def _set_trainable_adapter_only(model, adapter_name: str):
    """
    仅训练指定 adapter 的 LoRA 参数。
    Train only the LoRA parameters belonging to the selected adapter.
    """
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)

    for param in model.parameters():
        param.requires_grad = False

    trainable = 0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        if f".{adapter_name}." in name or name.endswith(f".{adapter_name}"):
            param.requires_grad = True
            trainable += int(param.numel())

    if trainable <= 0:
        raise RuntimeError(f"Failed to find trainable parameters for adapter={adapter_name}")


def _train_single_adapter(
    model,
    dataset,
    cfg,
    tokenizer,
    out_dir: str,
    adapter_name: str,
    *,
    build_baseline_sft_trainer,
    clear_cuda_cache,
):
    _set_trainable_adapter_only(model, adapter_name)
    trainer = build_baseline_sft_trainer(
        model=model,
        dataset=dataset,
        cfg=cfg,
        lr=cfg["training"]["lora_lr"],
        epochs=cfg["training"]["cluster_epochs"],
        tokenizer=tokenizer,
        out_dir=out_dir,
        peft_config=None,
    )
    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False
    trainer.train()
    if old_use_cache is not None:
        model.config.use_cache = old_use_cache
    del trainer
    clear_cuda_cache()


def _weighted_merge_text_metrics(weighted_parts):
    total = sum(item["weight"] for item in weighted_parts)
    if total <= 0:
        return {"text_generation_metrics": {"bleu1": 0.0, "rougeL_f1": 0.0, "eval_samples": 0}}

    bleu = sum(item["metrics"]["text_generation_metrics"]["bleu1"] * item["weight"] for item in weighted_parts) / total
    rouge = sum(item["metrics"]["text_generation_metrics"]["rougeL_f1"] * item["weight"] for item in weighted_parts) / total
    return {
        "text_generation_metrics": {
            "bleu1": float(bleu),
            "rougeL_f1": float(rouge),
            "eval_samples": int(total),
        }
    }


def train_raie(
    train_dataset,
    features,
    cfg,
    device,
    logger,
    tokenizer,
    model_id,
    *,
    load_causal_lm,
    build_baseline_sft_trainer,
    clear_cuda_cache,
):
    logger.info("[Baseline] training raie")

    selector = DBCHKSelector(
        k_min=cfg["clustering"]["k_min"],
        k_max=cfg["clustering"]["k_max"],
        random_state=cfg["clustering"].get("random_state", 42),
    )
    best_k, labels, centroids = selector.select(features)
    ranks = _equal_rank_layout(best_k, cfg)
    logger.info(f"[Baseline][raie] selected_k={best_k}, ranks={ranks}")

    base_model = load_causal_lm(model_id, device)
    default_out_dir = os.path.join(
        cfg.get("outputs", {}).get("trl_run_root", "eval/trl_runs"),
        "baseline_raie_default_r32",
    )
    default_trainer = build_baseline_sft_trainer(
        model=base_model,
        dataset=train_dataset,
        cfg=cfg,
        lr=cfg["training"]["lora_lr"],
        epochs=cfg["training"]["cluster_epochs"],
        tokenizer=tokenizer,
        out_dir=default_out_dir,
        peft_config=_build_lora_config(32, cfg),
    )
    old_use_cache = getattr(base_model.config, "use_cache", None)
    if old_use_cache is not None:
        base_model.config.use_cache = False
    default_trainer.train()
    if old_use_cache is not None:
        base_model.config.use_cache = old_use_cache

    default_model = default_trainer.model
    del default_trainer
    clear_cuda_cache()

    merged_backbone = default_model.merge_and_unload()
    del default_model
    clear_cuda_cache()

    first_name = "region_0"
    region_model = get_peft_model(merged_backbone, _build_lora_config(ranks[0], cfg), adapter_name=first_name)
    adapter_names = [first_name]
    for region_id in range(1, best_k):
        adapter_name = f"region_{region_id}"
        region_model.add_adapter(adapter_name, _build_lora_config(ranks[region_id], cfg))
        adapter_names.append(adapter_name)
    region_model.to(device)

    run_root = cfg.get("outputs", {}).get("trl_run_root", "eval/trl_runs")
    for region_id in range(best_k):
        region_indices = np.where(labels == region_id)[0].tolist()
        if not region_indices:
            continue
        logger.info(f"[Baseline][raie] region={region_id}, samples={len(region_indices)}")
        region_subset = Subset(train_dataset, region_indices)
        _train_single_adapter(
            region_model,
            region_subset,
            cfg,
            tokenizer,
            out_dir=os.path.join(run_root, f"baseline_raie_region_{region_id}"),
            adapter_name=adapter_names[region_id],
            build_baseline_sft_trainer=build_baseline_sft_trainer,
            clear_cuda_cache=clear_cuda_cache,
        )

    return RAIEState(
        model=region_model,
        centroids=np.asarray(centroids, dtype=np.float32),
        adapter_names=adapter_names,
        ranks=ranks,
        routing_model_id=model_id,
    )


def continue_raie(
    state: RAIEState,
    adaptation_dataset,
    adaptation_labels,
    new_centroids,
    cfg,
    device,
    logger,
    tokenizer,
    *,
    inherit_map: Optional[Dict[int, int]] = None,
    build_baseline_sft_trainer,
    clear_cuda_cache,
):
    logger.info("[Baseline] updating raie for stage2")
    new_centroids = np.asarray(new_centroids, dtype=np.float32)
    new_labels = np.asarray(adaptation_labels, dtype=int)
    new_k = int(new_centroids.shape[0])
    target_ranks = _equal_rank_layout(new_k, cfg)
    new_adapter_names = [None] * new_k
    effective_ranks = [0] * new_k

    for new_idx in range(new_k):
        old_idx = None if inherit_map is None else inherit_map.get(new_idx)
        if old_idx is not None and 0 <= old_idx < len(state.adapter_names):
            new_adapter_names[new_idx] = state.adapter_names[old_idx]
            effective_ranks[new_idx] = int(state.ranks[old_idx])
        else:
            adapter_name = f"region_s2_{new_idx}"
            if adapter_name not in getattr(state.model, "peft_config", {}):
                state.model.add_adapter(adapter_name, _build_lora_config(target_ranks[new_idx], cfg))
            new_adapter_names[new_idx] = adapter_name
            effective_ranks[new_idx] = int(target_ranks[new_idx])

    run_root = cfg.get("outputs", {}).get("trl_run_root", "eval/trl_runs")
    for region_id in range(new_k):
        region_indices = np.where(new_labels == region_id)[0].tolist()
        if not region_indices:
            continue
        logger.info(f"[Baseline][stage2_raie] region={region_id}, samples={len(region_indices)}")
        region_subset = Subset(adaptation_dataset, region_indices)
        _train_single_adapter(
            state.model,
            region_subset,
            cfg,
            tokenizer,
            out_dir=os.path.join(run_root, f"baseline_stage2_raie_region_{region_id}"),
            adapter_name=new_adapter_names[region_id],
            build_baseline_sft_trainer=build_baseline_sft_trainer,
            clear_cuda_cache=clear_cuda_cache,
        )

    state.centroids = new_centroids
    state.adapter_names = list(new_adapter_names)
    state.ranks = list(effective_ranks)
    state.model.to(device)
    clear_cuda_cache()
    return state


def evaluate_raie_state(
    state: RAIEState,
    test_dataset,
    cfg,
    device,
    logger,
    tokenizer,
    *,
    load_causal_lm,
    collect_features_with_cache,
    build_feature_cache_path,
    clear_cuda_cache,
    evaluate_model_for_task_fn=evaluate_model_for_task,
    eval_tag: str = "raie",
    cache_suffix: str = "stage1",
):
    eval_cap = int(cfg.get("evaluation", {}).get("max_samples", 500))
    eval_dataset = maybe_subset(test_dataset, eval_cap)
    routing_model = load_causal_lm(state.routing_model_id, device)
    test_cache = build_feature_cache_path(cfg, split_name=f"baseline_{eval_tag}_{cache_suffix}_test", dataset_len=len(eval_dataset))
    test_features = collect_features_with_cache(
        routing_model,
        eval_dataset,
        batch_size=cfg["data"]["feature_batch_size"],
        device=device,
        feature_strategy=cfg["clustering"].get("feature_strategy", "sentence_mean"),
        cache_path=test_cache,
        logger=logger,
    )
    del routing_model
    clear_cuda_cache()

    routed_labels = _nearest_centroid_labels(test_features, state.centroids)
    eval_cfg = clone_config_with_overrides(cfg, {"evaluation": {"max_samples": 0}})
    weighted_parts = []
    for region_id, adapter_name in enumerate(state.adapter_names):
        region_indices = np.where(routed_labels == region_id)[0].tolist()
        if not region_indices:
            continue
        if hasattr(state.model, "set_adapter"):
            state.model.set_adapter(adapter_name)
        subset = Subset(eval_dataset, region_indices)
        metrics = evaluate_model_for_task_fn(state.model, subset, eval_cfg, device, tokenizer)
        weighted_parts.append({"weight": len(region_indices), "metrics": metrics})

    merged = _weighted_merge_text_metrics(weighted_parts)
    logger.info(f"[Baseline][{eval_tag}] {merged}")
    return merged


def evaluate_raie(
    train_dataset,
    test_dataset,
    features,
    cfg,
    device,
    logger,
    tokenizer,
    model_id,
    *,
    load_causal_lm,
    build_baseline_sft_trainer,
    clear_cuda_cache,
    collect_features_with_cache,
    build_feature_cache_path,
    evaluate_model_for_task_fn=evaluate_model_for_task,
):
    state = train_raie(
        train_dataset,
        features,
        cfg,
        device,
        logger,
        tokenizer,
        model_id,
        load_causal_lm=load_causal_lm,
        build_baseline_sft_trainer=build_baseline_sft_trainer,
        clear_cuda_cache=clear_cuda_cache,
    )
    metrics = evaluate_raie_state(
        state,
        test_dataset,
        cfg,
        device,
        logger,
        tokenizer,
        load_causal_lm=load_causal_lm,
        collect_features_with_cache=collect_features_with_cache,
        build_feature_cache_path=build_feature_cache_path,
        clear_cuda_cache=clear_cuda_cache,
        evaluate_model_for_task_fn=evaluate_model_for_task_fn,
        eval_tag="raie",
        cache_suffix="stage1",
    )
    del state.model
    clear_cuda_cache()
    return metrics
