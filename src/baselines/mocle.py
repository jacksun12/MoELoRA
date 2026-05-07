from src.clustering.spherical_cluster import SphericalKMeans


def train_mocle(
    train_dataset,
    features,
    cfg,
    device,
    logger,
    model_id,
    *,
    load_causal_lm,
    inject_system_into_base_model,
    set_model_router_centroids,
    train_experts_by_clusters,
    clear_cuda_cache,
):
    logger.info("[Baseline] training mocle_4x8")
    n_experts = 4
    clusterer = SphericalKMeans(
        n_clusters=min(n_experts, max(1, len(train_dataset) - 1)),
        random_state=cfg["clustering"].get("random_state", 42),
    )
    mocle_labels = clusterer.fit_predict(features)
    mocle_centroids = clusterer.centroids_

    model = load_causal_lm(model_id, device)
    inject_system_into_base_model(
        model,
        target_modules=cfg["model"]["target_modules"],
        initial_ranks=[8] * int(mocle_centroids.shape[0]),
        top_k=1,
        alpha=cfg["model"]["alpha"],
        dropout=cfg["model"]["dropout"],
    )
    set_model_router_centroids(model, mocle_centroids)
    train_experts_by_clusters(
        model,
        train_dataset,
        mocle_labels,
        cfg,
        device,
        logger,
        stage_tag="baseline_mocle",
    )
    clear_cuda_cache()
    return model


def evaluate_mocle(
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
    inject_system_into_base_model,
    set_model_router_centroids,
    train_experts_by_clusters,
    clear_cuda_cache,
    evaluate_model_for_task,
):
    model = train_mocle(
        train_dataset,
        features,
        cfg,
        device,
        logger,
        model_id,
        load_causal_lm=load_causal_lm,
        inject_system_into_base_model=inject_system_into_base_model,
        set_model_router_centroids=set_model_router_centroids,
        train_experts_by_clusters=train_experts_by_clusters,
        clear_cuda_cache=clear_cuda_cache,
    )
    metrics = evaluate_model_for_task(model, test_dataset, cfg, device, tokenizer)
    logger.info(f"[Baseline][mocle_4x8] {metrics}")
    del model
    clear_cuda_cache()
    return metrics
