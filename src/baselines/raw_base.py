def evaluate_raw_base(cfg, device, tokenizer_builder, dataset_builder, load_causal_lm, evaluate_model_for_task, logger):
    raw_model_id = cfg["model"]["base_model_path"]
    logger.info(f"[Baseline] evaluating raw_base={raw_model_id}")
    raw_tokenizer = tokenizer_builder(raw_model_id)
    raw_tokenizer.pad_token = raw_tokenizer.eos_token
    raw_tokenizer.padding_side = "left"
    raw_test_dataset = dataset_builder(cfg, tokenizer=raw_tokenizer, split="test")
    raw_model = load_causal_lm(raw_model_id, device)
    raw_metrics = evaluate_model_for_task(raw_model, raw_test_dataset, cfg, device, raw_tokenizer)
    logger.info(f"[Baseline][raw_base] {raw_metrics}")
    return raw_model, raw_metrics
