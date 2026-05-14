import os
import random
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from src.utils.cluster_viz import plot_cluster_style_mix


def _deep_merge_dict(base, overrides):
    merged = dict(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path="config_unified.yaml", profile=None):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    profiles = cfg.pop("profiles", {}) or {}
    selected_profile = profile or cfg.pop("active_profile", None)
    if selected_profile:
        if selected_profile not in profiles:
            available = ", ".join(sorted(profiles.keys()))
            raise ValueError(f"Unknown config profile={selected_profile}. Available profiles: {available}")
        cfg = _deep_merge_dict(cfg, profiles[selected_profile])
        cfg["_profile"] = selected_profile
    return cfg


def resolve_device(cfg):
    preferred = cfg.get("runtime", {}).get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if preferred.startswith("cuda") and torch.cuda.is_available():
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
    idx = np.arange(len(dataset))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    return Subset(dataset, idx[:max_samples].tolist())


def get_cfg_batch_size(cfg, key, fallback_key="batch_size", default=4):
    data_cfg = cfg.get("data", {})
    return int(data_cfg.get(key, data_cfg.get(fallback_key, default)))


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def split_historical_and_new(dataset, drift_sample_size):
    n = len(dataset)
    if n <= 1:
        return dataset, dataset, list(range(n))

    requested = int(drift_sample_size)
    if requested <= 0 or requested >= n:
        requested = max(1, n // 4)
    historical_end = max(1, n - requested)
    new_indices = list(range(historical_end, n))
    historical_indices = list(range(historical_end))
    return Subset(dataset, historical_indices), dataset, new_indices


def clone_config_with_overrides(cfg, overrides):
    cloned = dict(cfg)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cloned.get(key), dict):
            merged = dict(cloned[key])
            merged.update(value)
            cloned[key] = merged
        else:
            cloned[key] = value
    return cloned


def get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def get_dataset_task_type(dataset):
    base = dataset
    while isinstance(base, Subset):
        base = base.dataset
    return getattr(base, "task_type", "multiple_choice")


def get_dataset_base_indices(dataset):
    base = dataset
    indices = np.arange(len(dataset))
    while isinstance(base, Subset):
        parent = base.dataset
        parent_indices = np.asarray(base.indices)
        indices = parent_indices[indices]
        base = parent
    return base, indices.astype(int)


def summarize_cluster_style_mix(dataset, labels, top_k=None):
    base, indices = get_dataset_base_indices(dataset)
    if not hasattr(base, "samples"):
        return {}

    labels = np.asarray(labels, dtype=int)
    if len(labels) != len(indices):
        raise ValueError("labels must align with dataset length for style-mix summarization")

    style_by_cluster = {}
    for cluster_id in np.unique(labels):
        member_indices = indices[labels == cluster_id]
        counter = Counter()
        for idx in member_indices:
            sample = base.samples[int(idx)]
            counter[str(sample.get("style_label", "unknown"))] += 1

        total = sum(counter.values())
        ranked = counter.most_common()
        if top_k is not None and top_k > 0:
            ranked = ranked[:top_k]
        style_by_cluster[int(cluster_id)] = {
            "total": int(total),
            "styles": [
                {"style": style, "count": int(count), "ratio": round(float(count / max(total, 1)), 4)}
                for style, count in ranked
            ],
        }
    return style_by_cluster


def save_cluster_style_mix_figure(cfg, stage_tag, style_mix_summary):
    out_dir = cfg.get("clustering", {}).get("visualization_dir", "eval/figures")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_slug = stage_tag.lower().replace(" ", "_").replace("-", "_")
    save_path = os.path.join(out_dir, f"{stage_slug}_style_mix_{ts}.png")
    plot_cluster_style_mix(
        style_mix_summary,
        save_path=save_path,
        title=f"{stage_tag} Cluster Style Composition",
        subtitle="Stacked ratios of style labels inside each discovered cluster",
    )
    return save_path


def maybe_log_cluster_style_mix(cfg, logger, dataset, labels, task_type, stage_tag):
    if task_type != "text_generation":
        return
    summary = summarize_cluster_style_mix(dataset, labels)
    logger.info(f"[{stage_tag}] cluster_style_mix={summary}")
    fig_path = save_cluster_style_mix_figure(cfg, stage_tag, summary)
    logger.info(f"[{stage_tag}] cluster_style_mix_fig={fig_path}")


def build_stage2_adaptation_subset(
    current_private_dataset,
    new_arrival_indices,
    historical_stage1_dataset,
    historical_labels,
    cfg,
):
    stage2_cfg = cfg.get("stage2", {})
    replay_enable = bool(stage2_cfg.get("replay_enable", True))
    if not replay_enable or len(new_arrival_indices) == 0:
        return (
            Subset(current_private_dataset, new_arrival_indices),
            np.asarray(new_arrival_indices, dtype=int),
            {
                "replay_enabled": replay_enable,
                "replay_samples": 0,
                "new_samples": len(new_arrival_indices),
                "adaptation_total": len(new_arrival_indices),
            },
        )

    replay_ratio = float(stage2_cfg.get("replay_ratio", 0.5))
    replay_strategy = str(stage2_cfg.get("replay_strategy", "stratified")).lower()
    replay_min_per_cluster = int(stage2_cfg.get("replay_min_per_cluster", 0))
    target_replay = int(round(len(new_arrival_indices) * max(replay_ratio, 0.0)))
    if target_replay <= 0:
        return (
            Subset(current_private_dataset, new_arrival_indices),
            np.asarray(new_arrival_indices, dtype=int),
            {
                "replay_enabled": replay_enable,
                "replay_samples": 0,
                "new_samples": len(new_arrival_indices),
                "adaptation_total": len(new_arrival_indices),
            },
        )

    _, historical_base_indices = get_dataset_base_indices(historical_stage1_dataset)
    historical_labels = np.asarray(historical_labels, dtype=int)
    if historical_labels.shape[0] != historical_base_indices.shape[0]:
        raise ValueError("historical_labels must align with historical_stage1_dataset")

    rng = np.random.default_rng(int(cfg.get("data", {}).get("seed", 42)))
    replay_indices = []

    if replay_strategy == "stratified":
        unique_clusters, counts = np.unique(historical_labels, return_counts=True)
        remaining_target = min(target_replay, len(historical_base_indices))
        allocations = {}
        total_count = int(counts.sum())

        for cluster_id, count in zip(unique_clusters, counts):
            share = count / max(total_count, 1)
            alloc = int(round(remaining_target * share))
            if replay_min_per_cluster > 0:
                alloc = max(alloc, replay_min_per_cluster)
            allocations[int(cluster_id)] = min(int(count), alloc)

        allocated_total = sum(allocations.values())
        if allocated_total > remaining_target:
            overflow = allocated_total - remaining_target
            for cluster_id in sorted(allocations, key=lambda x: allocations[x], reverse=True):
                if overflow <= 0:
                    break
                reducible = max(0, allocations[cluster_id] - replay_min_per_cluster)
                delta = min(reducible, overflow)
                allocations[cluster_id] -= delta
                overflow -= delta
        elif allocated_total < remaining_target:
            deficit = remaining_target - allocated_total
            for cluster_id, count in sorted(zip(unique_clusters, counts), key=lambda x: x[1], reverse=True):
                cid = int(cluster_id)
                spare = int(count) - allocations[cid]
                if spare <= 0:
                    continue
                delta = min(spare, deficit)
                allocations[cid] += delta
                deficit -= delta
                if deficit <= 0:
                    break

        for cluster_id in unique_clusters:
            member_positions = np.where(historical_labels == cluster_id)[0]
            take = min(len(member_positions), allocations[int(cluster_id)])
            if take <= 0:
                continue
            chosen = rng.choice(member_positions, size=take, replace=False)
            replay_indices.extend(historical_base_indices[chosen].tolist())
    else:
        replay_take = min(target_replay, len(historical_base_indices))
        chosen = rng.choice(np.arange(len(historical_base_indices)), size=replay_take, replace=False)
        replay_indices.extend(historical_base_indices[chosen].tolist())

    combined_indices = sorted(set(replay_indices + list(new_arrival_indices)))
    return (
        Subset(current_private_dataset, combined_indices),
        np.asarray(combined_indices, dtype=int),
        {
            "replay_enabled": replay_enable,
            "replay_strategy": replay_strategy,
            "replay_ratio": replay_ratio,
            "replay_samples": len(set(replay_indices)),
            "new_samples": len(new_arrival_indices),
            "adaptation_total": len(combined_indices),
        },
    )


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
