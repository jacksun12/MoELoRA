import os

from peft import LoraConfig


def train_single_lora(
    train_dataset,
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
    logger.info("[Baseline] training single_lora_r32")
    model = load_causal_lm(model_id, device)
    out_dir = os.path.join(cfg.get("outputs", {}).get("trl_run_root", "eval/trl_runs"), "baseline_single_lora_r32")
    peft_cfg = LoraConfig(
        r=32,
        lora_alpha=int(cfg["model"].get("alpha", 16)),
        lora_dropout=float(cfg["model"].get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["model"]["target_modules"],
    )
    trainer = build_baseline_sft_trainer(
        model=model,
        dataset=train_dataset,
        cfg=cfg,
        lr=cfg["training"]["lora_lr"],
        epochs=cfg["training"]["cluster_epochs"],
        tokenizer=tokenizer,
        out_dir=out_dir,
        peft_config=peft_cfg,
    )
    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False
    trainer.train()
    if old_use_cache is not None:
        model.config.use_cache = old_use_cache
    trained_model = trainer.model
    del trainer
    clear_cuda_cache()
    return trained_model


def evaluate_single_lora(
    train_dataset,
    test_dataset,
    cfg,
    device,
    logger,
    tokenizer,
    model_id,
    *,
    load_causal_lm,
    build_baseline_sft_trainer,
    clear_cuda_cache,
    evaluate_model_for_task,
):
    model = train_single_lora(
        train_dataset,
        cfg,
        device,
        logger,
        tokenizer,
        model_id,
        load_causal_lm=load_causal_lm,
        build_baseline_sft_trainer=build_baseline_sft_trainer,
        clear_cuda_cache=clear_cuda_cache,
    )
    metrics = evaluate_model_for_task(model, test_dataset, cfg, device, tokenizer)
    logger.info(f"[Baseline][single_lora_r32] {metrics}")
    del model
    clear_cuda_cache()
    return metrics
