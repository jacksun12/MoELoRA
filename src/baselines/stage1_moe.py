from src.clustering.spherical_cluster import DBCHKSelector


def train_stage1_moe(
    train_dataset,
    features,
    cfg,
    device,
    logger,
    model_id,
    *,
    load_causal_lm,
    build_rank_layout,
    inject_system_into_base_model,
    set_model_router_centroids,
    train_experts_by_clusters,
    clear_cuda_cache,
):
    logger.info("[Benchmark] training moe_lora_stage1")
    selector = DBCHKSelector(
        k_min=cfg["clustering"]["k_min"],
        k_max=cfg["clustering"]["k_max"],
        random_state=cfg["clustering"].get("random_state", 42),
    )
    best_k, labels, centroids = selector.select(features)
    logger.info(f"[Benchmark][moe_lora_stage1] selected_k={best_k}")

    ranks, _, _ = build_rank_layout(features, labels, cfg)

    model = load_causal_lm(model_id, device)
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
    clear_cuda_cache()
    return model


def evaluate_stage1_moe(
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
    build_rank_layout,
    inject_system_into_base_model,
    set_model_router_centroids,
    train_experts_by_clusters,
    clear_cuda_cache,
    evaluate_model_for_task,
):
    model = train_stage1_moe(
        train_dataset,
        features,
        cfg,
        device,
        logger,
        model_id,
        load_causal_lm=load_causal_lm,
        build_rank_layout=build_rank_layout,
        inject_system_into_base_model=inject_system_into_base_model,
        set_model_router_centroids=set_model_router_centroids,
        train_experts_by_clusters=train_experts_by_clusters,
        clear_cuda_cache=clear_cuda_cache,
    )
    metrics = evaluate_model_for_task(model, test_dataset, cfg, device, tokenizer)
    logger.info(f"[Benchmark][moe_lora_stage1] {metrics}")
    del model
    clear_cuda_cache()
    return metrics
