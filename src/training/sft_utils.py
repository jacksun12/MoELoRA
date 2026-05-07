import os
from datetime import datetime

import torch
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer


def rows_from_dataset_for_sft(dataset):
    import numpy as np
    from torch.utils.data import Subset

    base = dataset
    indices = np.arange(len(dataset))
    while isinstance(base, Subset):
        parent = base.dataset
        parent_indices = np.asarray(base.indices)
        indices = parent_indices[indices]
        base = parent

    if not hasattr(base, "samples") or not hasattr(base, "_format_prompt"):
        raise AttributeError("Base dataset for TRL conversion must expose .samples and ._format_prompt().")

    rows = []
    for i in indices:
        s = base.samples[int(i)]
        target = base._format_target(s)
        rows.append({"text": base._format_prompt(s) + target})
    return rows


class CurveLogCallback(TrainerCallback):
    def __init__(self, curve_tracker, stage, track, global_step_state=None):
        self.curve_tracker = curve_tracker
        self.stage = stage
        self.track = track
        self.global_step_state = global_step_state
        self.offset = 0 if global_step_state is None else int(global_step_state.get("step", 0))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.curve_tracker is None or logs is None or "loss" not in logs:
            return
        gstep = self.offset + int(state.global_step)
        self.curve_tracker.log(stage=self.stage, track=self.track, step=gstep, loss=float(logs["loss"]))


def run_trl_sft(
    model,
    dataset,
    cfg,
    lr,
    epochs,
    stage,
    track,
    logger,
    target_device=None,
    curve_tracker=None,
    global_step_state=None,
    clear_cuda_cache_fn=None,
):
    rows = rows_from_dataset_for_sft(dataset)
    if len(rows) == 0:
        return float("nan")

    trl_ds = Dataset.from_list(rows)
    run_root = cfg.get("outputs", {}).get("trl_run_root", "eval/trl_runs")
    os.makedirs(run_root, exist_ok=True)
    out_dir = os.path.join(run_root, f"{stage}_{track}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    trl_bs = int(cfg["training"].get("trl_per_device_batch_size", cfg["data"].get("batch_size", 2)))
    trl_accum = int(cfg["training"].get("trl_gradient_accumulation_steps", cfg["training"].get("gradient_accumulation_steps", 1)))
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

    trainer = SFTTrainer(
        model=model,
        args=train_cfg,
        train_dataset=trl_ds,
        processing_class=cfg["_tokenizer_obj"],
        callbacks=[CurveLogCallback(curve_tracker, stage, track, global_step_state)],
    )

    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False
    result = trainer.train()
    if old_use_cache is not None:
        model.config.use_cache = old_use_cache
    if target_device is not None:
        model.to(target_device)
        if clear_cuda_cache_fn is not None:
            clear_cuda_cache_fn()
    if global_step_state is not None:
        global_step_state["step"] = int(global_step_state.get("step", 0)) + int(trainer.state.global_step)
    loss = float(result.training_loss) if hasattr(result, "training_loss") else float("nan")
    logger.info(f"[{stage}][{track}] trl_steps={trainer.state.global_step}, trl_loss={loss:.4f}")
    return loss


def build_baseline_sft_trainer(model, dataset, cfg, lr, epochs, tokenizer, out_dir, peft_config=None):
    rows = rows_from_dataset_for_sft(dataset)
    if len(rows) == 0:
        raise ValueError("Baseline SFT dataset is empty.")
    trl_ds = Dataset.from_list(rows)
    trl_bs = int(cfg["training"].get("trl_per_device_batch_size", cfg["data"].get("batch_size", 2)))
    trl_accum = int(cfg["training"].get("trl_gradient_accumulation_steps", cfg["training"].get("gradient_accumulation_steps", 1)))
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

    trainer_kwargs = {
        "model": model,
        "args": train_cfg,
        "train_dataset": trl_ds,
        "processing_class": tokenizer,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    return SFTTrainer(**trainer_kwargs)
