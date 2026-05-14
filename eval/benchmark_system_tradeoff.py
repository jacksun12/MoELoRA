import argparse
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark task quality vs system cost for CDS baselines.")
    parser.add_argument("--config", default="config_unified.yaml")
    parser.add_argument("--profile", default="", help="Optional profile inside a unified config file.")
    parser.add_argument(
        "--methods",
        default="raw_base,single_lora_r32,mocle_4x8,hydralora_4x8,raie,moe_lora_stage1",
        help="Comma-separated methods to benchmark.",
    )
    parser.add_argument("--output_json", default="")
    parser.add_argument("--stage1_samples", type=int, default=0, help="Override stage1 sample cap; 0 keeps config.")
    return parser.parse_args()


def main():
    args = parse_args()

    from transformers import AutoTokenizer
    from eval.plot_system_tradeoff import generate_tradeoff_plot

    from data.data_loader import build_dataset
    from main_server_sim import (
        build_rank_layout,
        clear_cuda_cache,
        ExperimentLogger,
        collect_features_with_cache,
        DBCHKSelector,
        evaluate_model_for_task,
        get_dataset_task_type,
        inject_system_into_base_model,
        load_config,
        maybe_subset,
        resolve_device,
        run_stage0_cloud_pretraining,
        set_model_router_centroids,
        set_global_seed,
        split_historical_and_new,
        train_hydralora_baseline_model,
        train_mocle_baseline_model,
        evaluate_raie_baseline_state,
        train_raie_baseline_model,
        train_single_lora_baseline_model,
        train_experts_by_clusters,
        _load_causal_lm,
        build_feature_cache_path,
    )
    cfg = load_config(args.config, profile=args.profile or None)
    set_global_seed(int(cfg.get("data", {}).get("seed", 42)))
    device = resolve_device(cfg)
    logger = ExperimentLogger(log_dir="eval/logs", exp_name="system_tradeoff_benchmark")

    model_id = run_stage0_cloud_pretraining(cfg, device=device, logger=logger)
    cfg["_personalization_model_id"] = model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    cfg["_tokenizer_obj"] = tokenizer

    full_train = build_dataset(cfg, tokenizer=tokenizer, split="train")
    full_test = build_dataset(cfg, tokenizer=tokenizer, split="test")
    stage1_cap = args.stage1_samples if args.stage1_samples > 0 else int(cfg["data"].get("stage1_samples", 50000))
    historical_train, _, _ = split_historical_and_new(
        full_train,
        cfg["stage2"].get("drift_sample_size", 1000),
    )
    historical_test, _, _ = split_historical_and_new(
        full_test,
        cfg["stage2"].get("drift_sample_size", 1000),
    )
    train_dataset = maybe_subset(historical_train, stage1_cap)
    test_dataset = historical_test

    task_type = get_dataset_task_type(train_dataset)
    if task_type != "text_generation":
        raise ValueError(f"This benchmark script currently expects CDS text-generation data, got task_type={task_type}")

    feature_cache_path = build_feature_cache_path(cfg, split_name="benchmark_stage1", dataset_len=len(train_dataset))
    base_model = _load_causal_lm(model_id, device)
    features = collect_features_with_cache(
        base_model,
        train_dataset,
        batch_size=cfg["data"]["feature_batch_size"],
        device=device,
        feature_strategy=cfg["clustering"].get("feature_strategy", "sentence_mean"),
        cache_path=feature_cache_path,
        logger=logger,
    )
    del base_model
    clear_cuda_cache()

    def _begin_measure():
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device=device)
            torch.cuda.synchronize(device=device)

    def _end_measure():
        import time

        if device.startswith("cuda"):
            import torch

            torch.cuda.synchronize(device=device)
            peak_alloc_gb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)
            peak_reserved_gb = torch.cuda.max_memory_reserved(device=device) / (1024 ** 3)
        else:
            peak_alloc_gb = None
            peak_reserved_gb = None
        return peak_alloc_gb, peak_reserved_gb, time.perf_counter()

    def _measure_train(name, train_fn):
        import time

        _begin_measure()
        start = time.perf_counter()
        model = train_fn()
        peak_alloc_gb, peak_reserved_gb, end = _end_measure()
        result = {
            "train_elapsed_sec": float(end - start),
            "train_peak_alloc_gb": None if peak_alloc_gb is None else float(peak_alloc_gb),
            "train_peak_reserved_gb": None if peak_reserved_gb is None else float(peak_reserved_gb),
        }
        logger.info(f"[Benchmark][{name}][train] {result}")
        return model, result

    def _measure_step(name, step_name, fn):
        import time

        _begin_measure()
        start = time.perf_counter()
        payload = fn()
        peak_alloc_gb, peak_reserved_gb, end = _end_measure()
        result = {
            f"{step_name}_elapsed_sec": float(end - start),
            f"{step_name}_peak_alloc_gb": None if peak_alloc_gb is None else float(peak_alloc_gb),
            f"{step_name}_peak_reserved_gb": None if peak_reserved_gb is None else float(peak_reserved_gb),
        }
        logger.info(f"[Benchmark][{name}][{step_name}] {result}")
        return payload, result

    def _train_moe_lora_stage1_with_breakdown():
        selector = DBCHKSelector(
            k_min=cfg["clustering"]["k_min"],
            k_max=cfg["clustering"]["k_max"],
            random_state=cfg["clustering"].get("random_state", 42),
        )

        def _select_clusters():
            return selector.select(features)

        (best_k, labels, centroids), cluster_result = _measure_step("moe_lora_stage1", "cluster_select", _select_clusters)
        logger.info(f"[Benchmark][moe_lora_stage1] selected_k={best_k}")

        def _build_rank_payload():
            ranks, svd_values, complexity = build_rank_layout(
                features,
                labels,
                cfg,
            )
            return {
                "ranks": ranks,
                "svd_values": svd_values,
                "complexity": complexity,
            }

        rank_payload, rank_result = _measure_step("moe_lora_stage1", "rank_layout", _build_rank_payload)
        ranks = rank_payload["ranks"]

        def _train_experts_payload():
            model = _load_causal_lm(model_id, device)
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
                train_dataset,
                labels,
                cfg,
                device,
                logger,
                stage_tag="benchmark_stage1",
            )
            return model

        model, expert_result = _measure_step("moe_lora_stage1", "expert_train", _train_experts_payload)
        train_result = {
            "train_elapsed_sec": (
                cluster_result["cluster_select_elapsed_sec"]
                + rank_result["rank_layout_elapsed_sec"]
                + expert_result["expert_train_elapsed_sec"]
            ),
            "train_peak_alloc_gb": max(
                v
                for v in [
                    cluster_result["cluster_select_peak_alloc_gb"],
                    rank_result["rank_layout_peak_alloc_gb"],
                    expert_result["expert_train_peak_alloc_gb"],
                ]
                if v is not None
            ),
            "train_peak_reserved_gb": max(
                v
                for v in [
                    cluster_result["cluster_select_peak_reserved_gb"],
                    rank_result["rank_layout_peak_reserved_gb"],
                    expert_result["expert_train_peak_reserved_gb"],
                ]
                if v is not None
            ),
            **cluster_result,
            **rank_result,
            **expert_result,
        }
        logger.info(f"[Benchmark][moe_lora_stage1][train] {train_result}")
        return model, train_result

    def _measure_eval(name, model, eval_dataset, eval_tokenizer):
        import time

        _begin_measure()
        start = time.perf_counter()
        metrics = evaluate_model_for_task(model, eval_dataset, cfg, device, eval_tokenizer)
        peak_alloc_gb, peak_reserved_gb, end = _end_measure()
        task_metrics = metrics.get("text_generation_metrics", metrics)
        result = {
            "bleu1": float(task_metrics.get("bleu1", 0.0)),
            "rougeL_f1": float(task_metrics.get("rougeL_f1", 0.0)),
            "eval_samples": int(task_metrics.get("eval_samples", 0)),
            "eval_elapsed_sec": float(end - start),
            "eval_peak_alloc_gb": None if peak_alloc_gb is None else float(peak_alloc_gb),
            "eval_peak_reserved_gb": None if peak_reserved_gb is None else float(peak_reserved_gb),
        }
        logger.info(f"[Benchmark][{name}][eval] {result}")
        return result

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    results = []

    for method in methods:
        if method == "raw_base":
            raw_tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model_path"])
            raw_tokenizer.pad_token = raw_tokenizer.eos_token
            raw_tokenizer.padding_side = "left"
            raw_full_test = build_dataset(cfg, tokenizer=raw_tokenizer, split="test")
            raw_test_dataset, _, _ = split_historical_and_new(
                raw_full_test,
                cfg["stage2"].get("drift_sample_size", 1000),
            )
            model, train_result = _measure_train(method, lambda: _load_causal_lm(cfg["model"]["base_model_path"], device))
            eval_result = _measure_eval(method, model, raw_test_dataset, raw_tokenizer)
            del model
            clear_cuda_cache()
            results.append({"method": method, **train_result, **eval_result})
        elif method == "single_lora_r32":
            model, train_result = _measure_train(
                method,
                lambda: train_single_lora_baseline_model(train_dataset, cfg, device, logger, tokenizer, model_id),
            )
            eval_result = _measure_eval(method, model, test_dataset, tokenizer)
            del model
            clear_cuda_cache()
            results.append({"method": method, **train_result, **eval_result})
        elif method == "mocle_4x8":
            model, train_result = _measure_train(
                method,
                lambda: train_mocle_baseline_model(train_dataset, features, cfg, device, logger, model_id),
            )
            eval_result = _measure_eval(method, model, test_dataset, tokenizer)
            del model
            clear_cuda_cache()
            results.append({"method": method, **train_result, **eval_result})
        elif method == "hydralora_4x8":
            model, train_result = _measure_train(
                method,
                lambda: train_hydralora_baseline_model(train_dataset, cfg, device, logger, tokenizer, model_id),
            )
            eval_result = _measure_eval(method, model, test_dataset, tokenizer)
            del model
            clear_cuda_cache()
            results.append({"method": method, **train_result, **eval_result})
        elif method == "raie":
            state, train_result = _measure_train(
                method,
                lambda: train_raie_baseline_model(train_dataset, features, cfg, device, logger, tokenizer, model_id),
            )
            _begin_measure()
            import time
            start = time.perf_counter()
            metrics = evaluate_raie_baseline_state(
                state,
                test_dataset,
                cfg,
                device,
                logger,
                tokenizer,
                eval_tag="raie",
                cache_suffix="benchmark",
            )
            peak_alloc_gb, peak_reserved_gb, end = _end_measure()
            task_metrics = metrics.get("text_generation_metrics", metrics)
            eval_result = {
                "bleu1": float(task_metrics.get("bleu1", 0.0)),
                "rougeL_f1": float(task_metrics.get("rougeL_f1", 0.0)),
                "eval_samples": int(task_metrics.get("eval_samples", 0)),
                "eval_elapsed_sec": float(end - start),
                "eval_peak_alloc_gb": None if peak_alloc_gb is None else float(peak_alloc_gb),
                "eval_peak_reserved_gb": None if peak_reserved_gb is None else float(peak_reserved_gb),
            }
            logger.info(f"[Benchmark][{method}][eval] {eval_result}")
            del state.model
            clear_cuda_cache()
            results.append({"method": method, **train_result, **eval_result})
        elif method == "moe_lora_stage1":
            model, train_result = _train_moe_lora_stage1_with_breakdown()
            eval_result = _measure_eval(method, model, test_dataset, tokenizer)
            del model
            clear_cuda_cache()
            results.append({"method": method, **train_result, **eval_result})
        else:
            raise ValueError(f"Unknown method: {method}")

    output_json = args.output_json
    if not output_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json = os.path.join("eval", "results", f"system_tradeoff_{ts}.json")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"config": args.config, "methods": methods, "results": results}, f, ensure_ascii=False, indent=2)

    auto_plots = []
    default_plot_specs = [
        ("rougeL_f1", "train_peak_alloc_gb"),
        ("rougeL_f1", "train_elapsed_sec"),
        ("rougeL_f1", "eval_peak_alloc_gb"),
        ("rougeL_f1", "eval_elapsed_sec"),
    ]
    for metric, x_axis in default_plot_specs:
        try:
            plot_result = generate_tradeoff_plot(output_json, metric=metric, x_axis=x_axis)
            auto_plots.append(plot_result)
            logger.info(f"[Benchmark][Plot] {plot_result['output_path']}")
        except Exception as exc:
            logger.warning(f"[Benchmark][Plot] failed for {metric} vs {x_axis}: {exc}")

    print(json.dumps({"output_json": output_json, "results": results, "plots": auto_plots}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
