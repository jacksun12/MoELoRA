# MoE-LoRA System

面向端侧个性化的多阶段 LoRA 系统。  
Multi-stage LoRA system for on-device personalization.

当前仓库的主线任务是 **CDS style-transfer personalization**：

- `Stage-0`: cloud warmup
- `Stage-1`: private clustering + structure-aware rank budgeting + expert training
- `Stage-2`: incremental update under distribution drift
- `Stage-3`: serving-side scheduling and hardware-aware analysis

项目当前默认配置和实验脚本，都已经围绕这条主线整理完成。

## What This Repo Does

这个系统解决的问题是：

- 私有数据很少，直接端侧微调容易过拟合
- 私有分布不是单峰的，不同兴趣区域/风格区域复杂度不同
- 新数据会逐步到来，不能每次 full retrain
- 推理时不同请求会激活不同 LoRA / expert，服务策略也要考虑

对应地，仓库实现了四件事：

1. 用云侧公共数据先把 base model warm up
2. 在私有数据上做球面聚类，发现兴趣区域
3. 在固定 rank budget 下按结构复杂度分配 LoRA 容量
4. 在 serving 侧研究按 LoRA signature 聚批和异构放置

## Current Pipeline

### Stage-0: Cloud Warmup

先在云侧 pooled 数据上训练一个通用 adapter，再 merge 成 warm base。

当前默认用的是：

- `data/cds_cloud_pretrain`
- 输出到 `outputs/stage0_cloud_adapter`
- merge 后输出到 `outputs/stage0_cloud_merged`

入口：

- [`main_server_sim.py`](/root/aggLLMv3/MoE-LoRA-System/main_server_sim.py:79)
- [`scripts/train_trl_peft_baseline.py`](/root/aggLLMv3/MoE-LoRA-System/scripts/train_trl_peft_baseline.py:1)

### Stage-1: Initial Private Personalization

Stage-1 的核心链路是：

1. 从私有 train split 提取 hidden-state features
2. 用 `DB/CH + spherical clustering` 选择最优 `k`
3. 用 `SVD spectrum` 估计 cluster 结构复杂度
4. 在固定总预算下做 `rank allocation`
5. 每个 cluster 训练一个 LoRA expert
6. 用 centroid routing 做确定性路由

关键实现：

- [`src/clustering/spherical_cluster.py`](/root/aggLLMv3/MoE-LoRA-System/src/clustering/spherical_cluster.py:1)
- [`src/system/rank_allocator.py`](/root/aggLLMv3/MoE-LoRA-System/src/system/rank_allocator.py:1)
- [`src/models/moe_layer.py`](/root/aggLLMv3/MoE-LoRA-System/src/models/moe_layer.py:1)
- [`src/models/dynamic_router.py`](/root/aggLLMv3/MoE-LoRA-System/src/models/dynamic_router.py:1)
- [`src/baselines/stage1_moe.py`](/root/aggLLMv3/MoE-LoRA-System/src/baselines/stage1_moe.py:1)

当前 rank allocation 已经是 **pure structure-aware** 版本：

- 不再使用 PPL / cluster loss proxy
- 只依赖 `SVD complexity + SVD tail utility`

### Stage-2: Incremental Update Under Drift

当新私有数据到来时：

1. 将数据划分为 `historical` 和 `new arrivals`
2. 在 `historical + new` 上重新发现 cluster layout
3. 比较 old/new centroid overlap
4. 为每个新 cluster 找最近旧父簇
5. 用 replay + new arrivals 构造 adaptation set
6. 继承旧 expert 并继续训练

关键点：

- 现在的继承逻辑允许 **多个新簇继承同一个旧簇**
- 这比一对一 greedy matching 更符合 “旧簇 split 成多个新簇” 的情况

关键实现：

- [`src/system/overlap_manager.py`](/root/aggLLMv3/MoE-LoRA-System/src/system/overlap_manager.py:1)
- [`main_server_sim.py`](/root/aggLLMv3/MoE-LoRA-System/main_server_sim.py:963)

### Stage-3: Serving-Side Scheduling

Stage-3 关注的问题不是训练，而是：

- 不同 request 会激活不同 LoRA signature
- 同 signature 是否应该 batch 到一起
- 热门大批是否应该走 GPU
- 长尾碎片是否应该 offload 到 CPU
- backbone 是否更适合固定到 NPU

当前 Stage-3 分成两层：

1. **Synthetic serving simulator**
2. **Real GPU calibration on the current server**

关键实现：

- [`src/system/serving_sim.py`](/root/aggLLMv3/MoE-LoRA-System/src/system/serving_sim.py:1)
- [`eval/benchmark_stage3_serving.py`](/root/aggLLMv3/MoE-LoRA-System/eval/benchmark_stage3_serving.py:1)
- [`eval/calibrate_stage3_hardware.py`](/root/aggLLMv3/MoE-LoRA-System/eval/calibrate_stage3_hardware.py:1)
- [`eval/calibrate_stage3_adapter_switching.py`](/root/aggLLMv3/MoE-LoRA-System/eval/calibrate_stage3_adapter_switching.py:1)

## Evaluation Protocol

这是当前仓库里非常重要的一点。

### Stage-1

Stage-1 使用：

- `historical_train`
- `historical_test`

也就是：

- 第一阶段只在第一阶段的训练分布上训练
- 只在第一阶段对应的测试分布上评估

### Stage-2

Stage-2 使用：

- 目标训练分布：`historical + new arrivals`
- 训练方式：incremental adaptation，不是从头 full retrain
- 测试：`stage2_test`

也就是：

- Stage-2 的测试集会随分布变化而变化
- 不会再混用 Stage-1 的测试口径

当前这套协议在 [`main_server_sim.py`](/root/aggLLMv3/MoE-LoRA-System/main_server_sim.py:1027) 和 [`eval/benchmark_system_tradeoff.py`](/root/aggLLMv3/MoE-LoRA-System/eval/benchmark_system_tradeoff.py:22) 里已经统一。

## Baselines

当前已整理为独立模块的 baseline：

- `raw_base`
- `single_lora_r32`
- `mocle_4x8`
- `hydralora_4x8`
- `raie`
- `moe_lora_stage1`

代码位置：

- [`src/baselines/raw_base.py`](/root/aggLLMv3/MoE-LoRA-System/src/baselines/raw_base.py:1)
- [`src/baselines/single_lora.py`](/root/aggLLMv3/MoE-LoRA-System/src/baselines/single_lora.py:1)
- [`src/baselines/mocle.py`](/root/aggLLMv3/MoE-LoRA-System/src/baselines/mocle.py:1)
- [`src/baselines/hydralora.py`](/root/aggLLMv3/MoE-LoRA-System/src/baselines/hydralora.py:1)
- [`src/baselines/raie.py`](/root/aggLLMv3/MoE-LoRA-System/src/baselines/raie.py:1)
- [`src/baselines/stage1_moe.py`](/root/aggLLMv3/MoE-LoRA-System/src/baselines/stage1_moe.py:1)

Stage-2 baseline 当前主流程支持：

- `stage2_raw_base`
- `stage2_single_lora_r32`
- `stage2_raie`
- `our Stage-2`

## Repo Layout

### Main Entrypoints

- [`main_server_sim.py`](/root/aggLLMv3/MoE-LoRA-System/main_server_sim.py:1)
  端到端主流程 / end-to-end pipeline

- [`eval/benchmark_system_tradeoff.py`](/root/aggLLMv3/MoE-LoRA-System/eval/benchmark_system_tradeoff.py:1)
  Stage-1 质量-系统开销对比 / Stage-1 quality-vs-cost benchmark

- [`eval/benchmark_stage3_serving.py`](/root/aggLLMv3/MoE-LoRA-System/eval/benchmark_stage3_serving.py:1)
  Stage-3 synthetic serving benchmark

### Configs

- [`config_unified.yaml`](/root/aggLLMv3/MoE-LoRA-System/config_unified.yaml:1)
  唯一配置入口。包含当前所有实验 profile，并在字段旁标注了常用可选参数。

`config_unified.yaml` 当前支持这些 profile：

- `default_3style`
- `llama32_1b`
- `7style`
- `7style_drift`
- `7style_drift_llama32_1b`
- `multiapp_style`
- `multiapp_style_drift`
- `composite_multiapp`

### Data Builders

- [`data/build_cds_style_transfer.py`](/root/aggLLMv3/MoE-LoRA-System/data/build_cds_style_transfer.py:1)
  默认 CDS cloud/private 数据构建

- [`data/build_local_style_transfer_variant.py`](/root/aggLLMv3/MoE-LoRA-System/data/build_local_style_transfer_variant.py:1)
  构建更复杂的本地 style-transfer 私有数据

- [`data/build_local_style_transfer_drift.py`](/root/aggLLMv3/MoE-LoRA-System/data/build_local_style_transfer_drift.py:1)
  构建带显式 drift 的私有数据

- [`data/build_composite_stream.py`](/root/aggLLMv3/MoE-LoRA-System/data/build_composite_stream.py:1)
  按 manifest 将多个来源 merge 成单个设备用户流

- [`data/prepare_multi_app_composite.py`](/root/aggLLMv3/MoE-LoRA-System/data/prepare_multi_app_composite.py:1)
  从 `MovieLens + Amazon + Goodreads + Yelp + Music(可选)` 的预处理事件文件中构建多应用设备流

- [`data/build_local_multiapp_style_transfer.py`](/root/aggLLMv3/MoE-LoRA-System/data/build_local_multiapp_style_transfer.py:1)
  将当前 CDS 风格池重映射成设备侧 `app/context` 标签，构建与现有主线同构的多应用私有生成式数据

## Quick Start

### 1. Environment

推荐环境：

```bash
conda activate aggLLM
```

### 2. Default Pipeline

运行默认 3-style CDS 流程：

```bash
cd /root/aggLLMv3/MoE-LoRA-System
CUDA_VISIBLE_DEVICES=4 python main_server_sim.py \
  --config config_unified.yaml \
  --profile default_3style
```

### 3. More Complex 7-style Setting

```bash
cd /root/aggLLMv3/MoE-LoRA-System
CUDA_VISIBLE_DEVICES=4 python main_server_sim.py \
  --config config_unified.yaml \
  --profile 7style
```

### 4. Default Llama 3.2 1B Setting

```bash
cd /root/aggLLMv3/MoE-LoRA-System
CUDA_VISIBLE_DEVICES=4 python main_server_sim.py \
  --config config_unified.yaml \
  --profile llama32_1b
```

### 5. 7-style Drift Setting

```bash
cd /root/aggLLMv3/MoE-LoRA-System
CUDA_VISIBLE_DEVICES=4 python main_server_sim.py \
  --config config_unified.yaml \
  --profile 7style_drift
```

### 6. 7-style Drift with Llama 3.2 1B

```bash
cd /root/aggLLMv3/MoE-LoRA-System
CUDA_VISIBLE_DEVICES=4 python main_server_sim.py \
  --config config_unified.yaml \
  --profile 7style_drift_llama32_1b
```

### 7. Multi-App Composite Device Stream

先准备好多应用事件 JSONL。每一行建议至少包含：

- `user_id`
- `timestamp`
- `item_id`
- `title`
- `meta`

然后用下面的脚本把它们 merge 成一个设备级用户流：

```bash
cd /root/aggLLMv3/MoE-LoRA-System
python data/prepare_multi_app_composite.py \
  --output_root data/auto_composite_multiapp \
  --movielens_dir data/ml_1m \
  --amazon_events_path data/prepared_multiapp/amazon_events.jsonl \
  --goodreads_events_path data/prepared_multiapp/goodreads_events.jsonl \
  --yelp_events_path data/prepared_multiapp/yelp_events.jsonl
```

生成后可以直接跑：

```bash
cd /root/aggLLMv3/MoE-LoRA-System
CUDA_VISIBLE_DEVICES=4 python main_server_sim.py \
  --config config_unified.yaml \
  --profile composite_multiapp
```

也可以先参考这个 manifest 模板：

- [`data/composite_manifest_multiapp_example.json`](/root/aggLLMv3/MoE-LoRA-System/data/composite_manifest_multiapp_example.json:1)

### 8. Multi-App Private Style-Transfer Proxy

如果你想保持和当前 `CDS style-transfer` 完全一致的任务形态，只把私有数据改成更像端侧多应用隐私流，可以直接先构建这两套：

```bash
cd /root/aggLLMv3/MoE-LoRA-System
python data/build_local_multiapp_style_transfer.py \
  --mode plain \
  --output_dir data/cds_private_device_multiapp

python data/build_local_multiapp_style_transfer.py \
  --mode drift \
  --output_dir data/cds_private_device_multiapp_drift
```

对应运行配置：

```bash
cd /root/aggLLMv3/MoE-LoRA-System
CUDA_VISIBLE_DEVICES=4 python main_server_sim.py \
  --config config_unified.yaml \
  --profile multiapp_style

CUDA_VISIBLE_DEVICES=4 python main_server_sim.py \
  --config config_unified.yaml \
  --profile multiapp_style_drift
```

## Stage-1 System Trade-off Benchmark

比较 Stage-1 baseline 与我们的方法：

```bash
cd /root/aggLLMv3/MoE-LoRA-System
CUDA_VISIBLE_DEVICES=4 python ./eval/benchmark_system_tradeoff.py \
  --config config_unified.yaml \
  --profile 7style_drift \
  --methods raw_base,single_lora_r32,mocle_4x8,hydralora_4x8,raie,moe_lora_stage1
```

如果要切到 `Llama 3.2 1B`，直接把配置改成：

```bash
cd /root/aggLLMv3/MoE-LoRA-System
CUDA_VISIBLE_DEVICES=4 python ./eval/benchmark_system_tradeoff.py \
  --config config_unified.yaml \
  --profile 7style_drift_llama32_1b \
  --methods raw_base,single_lora_r32,mocle_4x8,hydralora_4x8,raie,moe_lora_stage1
```

运行结束后会自动生成：

- `eval/results/system_tradeoff_*.json`
- `eval/figures/system_tradeoff_*_rougeL_f1_vs_*.png`

> 注：`meta-llama/Llama-3.2-1B-Instruct` 可能需要 Hugging Face 访问权限；同时我已经把 Stage-0 输出目录单独拆开，避免误复用之前 `Qwen/Qwen2.5-0.5B` 的 warmup 结果。

## Stage-3 Serving Benchmark

### Synthetic Scheduling Benchmark

```bash
cd /root/aggLLMv3/MoE-LoRA-System
python ./eval/benchmark_stage3_serving.py --workload bursty_hotspot
```

当前支持 workload：

- `simple`
- `complex`
- `fragmented`
- `bursty_hotspot`

当前支持策略：

- `all_cpu`
- `fifo_gpu`
- `signature_gpu_cpu`
- `signature_length_gpu_cpu`
- `npu_hybrid`

### Real GPU Calibration

当前这部分是在服务器 GPU 上做实测，不是纯模拟。

#### Base / Warm / Adapter Latency

```bash
cd /root/aggLLMv3/MoE-LoRA-System
source /root/anaconda3/etc/profile.d/conda.sh
conda activate aggLLM
CUDA_VISIBLE_DEVICES=4 python ./eval/calibrate_stage3_hardware.py \
  --config config_unified.yaml \
  --profile 7style_drift \
  --device cuda:0 \
  --batch_sizes 1,4 \
  --prompt_lengths 64,256 \
  --decode_lengths 32
```

#### Adapter Sequence Switching Probe

```bash
cd /root/aggLLMv3/MoE-LoRA-System
source /root/anaconda3/etc/profile.d/conda.sh
conda activate aggLLM
CUDA_VISIBLE_DEVICES=4 python ./eval/calibrate_stage3_adapter_switching.py \
  --config config_unified.yaml \
  --profile 7style_drift \
  --device cuda:0 \
  --prompt_length 128 \
  --decode_length 32 \
  --num_requests 16
```

## Figures and Outputs

主流程会自动生成：

- Stage-1 summary 图
- Stage-2 baseline compare 图
- cluster visualization
- style mix 图
- training curve 图

位置：

- `eval/figures/`
- `eval/results/`
- `eval/logs/`

## Important Notes

### 1. Stage-1 和 Stage-2 的测试口径是分开的

当前仓库不会再把两个阶段的测试集混在一起。

### 2. Stage-3 目前是“模拟 + 真实 GPU 校准”

也就是说：

- `benchmark_stage3_serving.py` 是 synthetic scheduler benchmark
- `calibrate_stage3_hardware.py` 和 `calibrate_stage3_adapter_switching.py` 是真实 GPU 校准

目前还不是手机真机 benchmark。

### 3. 当前最自然的 Stage-3 叙事

第三阶段现在最适合讲的是：

- 不 merge backbone
- 请求先看激活哪个 LoRA / expert signature
- 同 signature 尽量聚成 batch
- 热门大批给 GPU
- 零散长尾给 CPU
- 静态 backbone 适合 NPU

而不是把重点放在 `adapter switch API` 本身。

## Current Best Use Case

目前这套系统最适合的实验风格是：

- 私有数据不大
- 分布逐步漂移
- 风格/兴趣区域明显多峰
- 需要同时讨论训练、更新、推理三层 trade-off

如果你想继续推进论文，当前仓库最值得优先补强的是：

1. 多 seed 稳定性
2. Stage-3 更完整的真实硬件验证
3. 手机端 CPU/GPU/NPU 实测
