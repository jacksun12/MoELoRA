import os


def train_hydralora(
    train_dataset,
    cfg,
    device,
    logger,
    tokenizer,
    model_id,
    *,
    load_causal_lm,
    inject_learned_moe_system,
    set_trainable_learned_moe,
    build_baseline_sft_trainer,
    clear_cuda_cache,
):
    logger.info("[Baseline] training hydralora_4x8")
    model = load_causal_lm(model_id, device)
    inject_learned_moe_system(
        model,
        target_modules=cfg["model"]["target_modules"],
        initial_ranks=[8, 8, 8, 8],
        top_k=1,
        alpha=cfg["model"]["alpha"],
        dropout=cfg["model"]["dropout"],
    )
    set_trainable_learned_moe(model)
    out_dir = os.path.join(cfg.get("outputs", {}).get("trl_run_root", "eval/trl_runs"), "baseline_hydralora_4x8")
    trainer = build_baseline_sft_trainer(
        model=model,
        dataset=train_dataset,
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
    return model


def evaluate_hydralora(
    train_dataset,
    test_dataset,
    cfg,
    device,
    logger,
    tokenizer,
    model_id,
    *,
    load_causal_lm,
    inject_learned_moe_system,
    set_trainable_learned_moe,
    build_baseline_sft_trainer,
    clear_cuda_cache,
    evaluate_model_for_task,
):
    model = train_hydralora(
        train_dataset,
        cfg,
        device,
        logger,
        tokenizer,
        model_id,
        load_causal_lm=load_causal_lm,
        inject_learned_moe_system=inject_learned_moe_system,
        set_trainable_learned_moe=set_trainable_learned_moe,
        build_baseline_sft_trainer=build_baseline_sft_trainer,
        clear_cuda_cache=clear_cuda_cache,
    )
    metrics = evaluate_model_for_task(model, test_dataset, cfg, device, tokenizer)
    logger.info(f"[Baseline][hydralora_4x8] {metrics}")
    del model
    clear_cuda_cache()
    return metrics
