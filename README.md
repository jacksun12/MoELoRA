# MoE-LoRA System

This repository implements a multi-stage system for efficient on-device personalized LLM adaptation with LoRA experts. The system is designed for settings where user-local data are scarce, heterogeneous, privacy-sensitive, and subject to distribution drift.

The main experimental pipeline is organized around CDS style-transfer personalization and its more complex multi-style and multi-application variants. The repository also includes baselines, Stage-2 continual adaptation, and Stage-3 serving-side scheduling experiments.

## Overview

MoE-LoRA decomposes a user's local data into latent behavior or style regions, assigns LoRA capacity under a fixed rank budget, and updates experts incrementally as the private distribution changes over time.

The system consists of four stages:

- `Stage-0`: cloud-side warmup on non-private pooled data.
- `Stage-1`: user-local clustering, structure-aware rank allocation, expert training, and centroid routing.
- `Stage-2`: inheritance-based incremental update under user distribution drift.
- `Stage-3`: serving-side analysis for LoRA-signature-aware batching and heterogeneous placement.

The key design goals are:

- Avoid overfitting when user-local data are limited.
- Reduce mode interference caused by heterogeneous local behavior.
- Allocate a fixed LoRA rank budget according to local structural complexity.
- Support continual updates without full retraining from scratch.
- Study serving strategies for mixed LoRA-signature request streams.

## Pipeline

### Stage-0: Cloud Warmup

Stage-0 optionally trains a lightweight adapter on pooled non-private data and merges it into the base model. This produces a warm-start backbone for downstream user-local adaptation.

Default Stage-0 data and outputs:

- Data: `data/cds_cloud_pretrain`
- Adapter output: `outputs/stage0_cloud_adapter`
- Merged model output: `outputs/stage0_cloud_merged`

Relevant code:

- [main_server_sim.py](/root/aggLLMv3/MoE-LoRA-System/main_server_sim.py:1)
- [scripts/train_trl_peft_baseline.py](/root/aggLLMv3/MoE-LoRA-System/scripts/train_trl_peft_baseline.py:1)

### Stage-1: Initial Private Personalization

Stage-1 performs the initial on-device personalization procedure:

1. Extract hidden-state features from the private training split.
2. Select the number of local modes with DB/CH-based spherical clustering.
3. Estimate cluster structural complexity from the representation spectrum.
4. Allocate the total LoRA rank budget across clusters.
5. Train one LoRA expert per discovered cluster.
6. Route future requests with nearest-centroid routing.

Relevant code:

- [src/clustering/spherical_cluster.py](/root/aggLLMv3/MoE-LoRA-System/src/clustering/spherical_cluster.py:1)
- [src/system/rank_allocator.py](/root/aggLLMv3/MoE-LoRA-System/src/system/rank_allocator.py:1)
- [src/models/moe_layer.py](/root/aggLLMv3/MoE-LoRA-System/src/models/moe_layer.py:1)
- [src/models/dynamic_router.py](/root/aggLLMv3/MoE-LoRA-System/src/models/dynamic_router.py:1)
- [src/baselines/stage1_moe.py](/root/aggLLMv3/MoE-LoRA-System/src/baselines/stage1_moe.py:1)

The current rank allocator is purely structure-aware. It does not use PPL or cluster loss estimation.

### Stage-2: Incremental Update Under Drift

Stage-2 adapts the personalized model when new private data arrive. The update procedure:

1. Splits the local stream into historical data and new arrivals.
2. Recomputes the cluster layout on the updated private distribution.
3. Measures overlap between old and new cluster layouts.
4. Assigns each new cluster to its nearest previous parent cluster.
5. Builds an adaptation set from new arrivals and replayed historical samples.
6. Inherits old expert structure and continues training.

The inheritance logic supports many-to-one parent mapping, so multiple new clusters may inherit from the same old cluster when a previous mode splits.

Relevant code:

- [src/system/overlap_manager.py](/root/aggLLMv3/MoE-LoRA-System/src/system/overlap_manager.py:1)
- [main_server_sim.py](/root/aggLLMv3/MoE-LoRA-System/main_server_sim.py:1)

### Stage-3: Serving-Side Scheduling

Stage-3 studies serving behavior for personalized multi-expert models. Different requests may activate different LoRA signatures, which changes batching and placement decisions.

The serving-side experiments cover:

- Signature-aware batching.
- GPU execution for hot, batchable LoRA signatures.
- CPU offloading for sparse long-tail signatures.
- NPU-style execution assumptions for a static shared backbone.

Relevant code:

- [src/system/serving_sim.py](/root/aggLLMv3/MoE-LoRA-System/src/system/serving_sim.py:1)
- [eval/benchmark_stage3_serving.py](/root/aggLLMv3/MoE-LoRA-System/eval/benchmark_stage3_serving.py:1)
- [eval/calibrate_stage3_hardware.py](/root/aggLLMv3/MoE-LoRA-System/eval/calibrate_stage3_hardware.py:1)
- [eval/calibrate_stage3_adapter_switching.py](/root/aggLLMv3/MoE-LoRA-System/eval/calibrate_stage3_adapter_switching.py:1)

## Evaluation Protocol

The repository uses a stage-aware evaluation protocol.

For Stage-1:

- Train on `historical_train`.
- Evaluate on `historical_test`.

For Stage-2:

- Treat `historical + new arrivals` as the updated target distribution.
- Adapt incrementally using new arrivals plus optional replay.
- Evaluate on `stage2_test`, which follows the updated distribution.

This avoids mixing Stage-1 and Stage-2 test distributions.

## Baselines

The current baseline implementations are organized under [src/baselines](/root/aggLLMv3/MoE-LoRA-System/src/baselines).

Stage-1 baselines:

- `raw_base`
- `single_lora_r32`
- `mocle_4x8`
- `hydralora_4x8`
- `raie`
- `moe_lora_stage1`

Stage-2 baselines:

- `stage2_raw_base`
- `stage2_single_lora_r32`
- `stage2_raie`
- `our Stage-2`

## Repository Layout

Main entrypoints:

- [main_server_sim.py](/root/aggLLMv3/MoE-LoRA-System/main_server_sim.py:1): end-to-end Stage-0 to Stage-3 pipeline.
- [eval/benchmark_system_tradeoff.py](/root/aggLLMv3/MoE-LoRA-System/eval/benchmark_system_tradeoff.py:1): Stage-1 quality and system-cost benchmark.
- [eval/benchmark_stage3_serving.py](/root/aggLLMv3/MoE-LoRA-System/eval/benchmark_stage3_serving.py:1): synthetic Stage-3 serving benchmark.

Configuration:

- [config_unified.yaml](/root/aggLLMv3/MoE-LoRA-System/config_unified.yaml:1): unified configuration file with all experiment profiles.

Data builders:

- [data/build_cds_style_transfer.py](/root/aggLLMv3/MoE-LoRA-System/data/build_cds_style_transfer.py:1): builds the default CDS cloud/private style-transfer data.
- [data/build_local_style_transfer_variant.py](/root/aggLLMv3/MoE-LoRA-System/data/build_local_style_transfer_variant.py:1): builds the 7-style private style-transfer variant.
- [data/build_local_style_transfer_drift.py](/root/aggLLMv3/MoE-LoRA-System/data/build_local_style_transfer_drift.py:1): builds the 7-style drift variant.
- [data/build_local_multiapp_style_transfer.py](/root/aggLLMv3/MoE-LoRA-System/data/build_local_multiapp_style_transfer.py:1): builds a multi-application style-transfer proxy while preserving the CDS-style task format.
- [data/build_composite_stream.py](/root/aggLLMv3/MoE-LoRA-System/data/build_composite_stream.py:1): merges multiple event sources into one device-level stream.
- [data/prepare_multi_app_composite.py](/root/aggLLMv3/MoE-LoRA-System/data/prepare_multi_app_composite.py:1): prepares a multi-application composite stream from MovieLens, Amazon, Goodreads, Yelp, and optional music events.

## Configuration Profiles

All experiments are configured through [config_unified.yaml](/root/aggLLMv3/MoE-LoRA-System/config_unified.yaml:1).

Available profiles:

- `default_3style`: default CDS private style-transfer setting.
- `llama32_1b`: default CDS setting with `meta-llama/Llama-3.2-1B-Instruct`.
- `7style`: more complex 7-style private data.
- `7style_drift`: 7-style private data with explicit distribution drift.
- `7style_drift_llama32_1b`: 7-style drift setting with Llama 3.2 1B.
- `multiapp_style`: multi-application private style-transfer proxy.
- `multiapp_style_drift`: multi-application private style-transfer proxy with explicit drift.
- `composite_multiapp`: cross-application device-event stream.

Example:

```bash
python main_server_sim.py --config config_unified.yaml --profile 7style_drift
```

## Quick Start

Activate the environment:

```bash
conda activate aggLLM
cd /root/aggLLMv3/MoE-LoRA-System
```

Run the default CDS pipeline:

```bash
CUDA_VISIBLE_DEVICES=4 python main_server_sim.py \
  --config config_unified.yaml \
  --profile default_3style
```

Run the 7-style drift setting:

```bash
CUDA_VISIBLE_DEVICES=4 python main_server_sim.py \
  --config config_unified.yaml \
  --profile 7style_drift
```

Run the Llama 3.2 1B setting:

```bash
CUDA_VISIBLE_DEVICES=4 python main_server_sim.py \
  --config config_unified.yaml \
  --profile llama32_1b
```

Run the multi-application style-transfer drift setting:

```bash
CUDA_VISIBLE_DEVICES=4 python main_server_sim.py \
  --config config_unified.yaml \
  --profile multiapp_style_drift
```

## Data Preparation

Build the multi-application style-transfer proxy:

```bash
python data/build_local_multiapp_style_transfer.py \
  --mode plain \
  --output_dir data/cds_private_device_multiapp

python data/build_local_multiapp_style_transfer.py \
  --mode drift \
  --output_dir data/cds_private_device_multiapp_drift
```

Build a composite device stream from multiple application event sources:

```bash
python data/prepare_multi_app_composite.py \
  --output_root data/auto_composite_multiapp \
  --movielens_dir data/ml_1m \
  --amazon_events_path data/prepared_multiapp/amazon_events.jsonl \
  --goodreads_events_path data/prepared_multiapp/goodreads_events.jsonl \
  --yelp_events_path data/prepared_multiapp/yelp_events.jsonl
```

The generic event JSONL files should contain at least:

- `user_id`
- `timestamp`
- `item_id`
- `title`
- `meta`

## Stage-1 System Trade-off Benchmark

Run the Stage-1 benchmark:

```bash
CUDA_VISIBLE_DEVICES=4 python eval/benchmark_system_tradeoff.py \
  --config config_unified.yaml \
  --profile 7style_drift \
  --methods raw_base,single_lora_r32,mocle_4x8,hydralora_4x8,raie,moe_lora_stage1
```

Run the Llama 3.2 1B benchmark:

```bash
CUDA_VISIBLE_DEVICES=4 python eval/benchmark_system_tradeoff.py \
  --config config_unified.yaml \
  --profile 7style_drift_llama32_1b \
  --methods raw_base,single_lora_r32,mocle_4x8,hydralora_4x8,raie,moe_lora_stage1
```

Outputs:

- `eval/results/system_tradeoff_*.json`
- `eval/figures/system_tradeoff_*_rougeL_f1_vs_*.png`

The Llama 3.2 1B profiles may require Hugging Face access to `meta-llama/Llama-3.2-1B-Instruct`.

## Stage-3 Serving Benchmark

Run the synthetic serving benchmark:

```bash
python eval/benchmark_stage3_serving.py --workload bursty_hotspot
```

Supported workloads:

- `simple`
- `complex`
- `fragmented`
- `bursty_hotspot`

Supported strategies:

- `all_cpu`
- `fifo_gpu`
- `signature_gpu_cpu`
- `signature_length_gpu_cpu`
- `npu_hybrid`

## Real GPU Calibration

Measure base, warm, and adapter latency:

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate aggLLM
CUDA_VISIBLE_DEVICES=4 python eval/calibrate_stage3_hardware.py \
  --config config_unified.yaml \
  --profile 7style_drift \
  --device cuda:0 \
  --batch_sizes 1,4 \
  --prompt_lengths 64,256 \
  --decode_lengths 32
```

Measure adapter sequence overhead:

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate aggLLM
CUDA_VISIBLE_DEVICES=4 python eval/calibrate_stage3_adapter_switching.py \
  --config config_unified.yaml \
  --profile 7style_drift \
  --device cuda:0 \
  --prompt_length 128 \
  --decode_length 32 \
  --num_requests 16
```

The Stage-3 serving benchmark is a synthetic scheduler benchmark. The calibration scripts measure real GPU behavior on the current server, but they are not mobile SoC measurements.

## Outputs

The main pipeline and benchmark scripts write artifacts to:

- `eval/logs/`
- `eval/results/`
- `eval/figures/`
- `eval/trl_runs/`
- `outputs/`

Typical generated artifacts include:

- Stage-1 summary figures.
- Stage-2 baseline comparison figures.
- Cluster visualizations.
- Cluster style-mix figures.
- Training curves.
- System trade-off JSON files and plots.

## Notes

- Stage-1 and Stage-2 use separate test distributions.
- Rank allocation is structure-aware and does not use PPL.
- Stage-2 inheritance allows multiple new clusters to inherit from the same previous cluster.
- Stage-3 focuses on LoRA-signature-aware batching and heterogeneous placement.
