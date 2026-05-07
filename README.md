# MoE-LoRA Privacy Personalization System

This project implements a **4-stage privacy personalization pipeline** for private LLM adaptation, with support for:

- recommendation-style data (`movielens_1m`, `composite_device`),
- style-transfer text generation (`style_transfer_jsonl`).

The current default configuration is the **CDS style-transfer setting**, which is a better fit for on-device private language personalization experiments.

The system objective is:
0. Pre-adapt the base model on large-scale cloud interaction data to reduce overfitting under small private composite data.
1. Partition private data into interest regions with **DB/CH-driven k selection + spherical clustering**.
2. Measure each region structural complexity by **SVD spectrum statistics**, then allocate LoRA rank under a **fixed total budget**.
3. Train one LoRA expert per cluster and route requests by nearest cluster centroid.
4. When new private data arrives, update cluster layout with **overlap-aware inheritance** instead of full retraining.
5. At serving time, group requests by activated LoRA signatures and schedule to **GPU/CPU heterogeneously**.

---

## Stage-0: Cloud Warmup (Optional)

Before private personalization, you can first warm up the base model on a large cross-user cloud dataset.
This stage is designed for the "small private data" setting:

- reduce overfitting on one device user,
- inject generic cross-user patterns into the base,
- then run clustering / rank allocation / expert adaptation on top of the warmed base.

Two cloud builders are supported:

### CDS cloud warmup (default text-generation setting)

```bash
python data/build_cds_style_transfer.py \
  --cloud_output_dir data/cds_cloud_pretrain \
  --private_output_dir data/cds_private_device \
  --private_total_samples 1200 \
  --max_cloud_samples_per_style 5000 \
  --private_style_mix 'tweets:0.5,switchboard:0.3,poetry:0.2'
```

### Yelp cloud warmup (recommendation setting)

```bash
python data/build_yelp_cloud_pretrain.py \
  --review_path "/path/to/yelp_academic_dataset_review.json" \
  --business_path "/path/to/yelp_academic_dataset_business.json" \
  --output_dir data/yelp_cloud_pretrain \
  --max_users 2000
```

Then enable `stage0.enable: true` in `config.yaml`. The main pipeline will:

- train a PEFT adapter on `stage0.data_dir`,
- merge that adapter into a warm base model,
- use the merged local checkpoint as the personalization base for later stages.

---

## Research Problem

Classical LoRA personalization typically uses fixed rank and static task assumptions. In privacy scenarios:
- user preference regions are heterogeneous (different complexity),
- private distribution drifts over time,
- full retraining every update is expensive.

We target a dynamic but budget-controlled solution:
- fixed rank budget,
- adaptive rank allocation to high-complexity clusters,
- overlap-based warm-start for continuous updates,
- practical system scheduling for mixed request parallelism.

---

## Stage-1: Initial Personalization (Offline)

### 1) Feature Extraction
- Build task-formatted private samples from the selected dataset.
- Extract hidden-state features per sample.

For the current CDS default:
- each sample is a style-transfer example,
- the model sees a source sentence and rewrites it into a target style,
- clustering is performed over private user samples to discover heterogeneous style modes.

Code:
- `data/data_loader.py`
- `main_server_sim.py::collect_features`

### 2) Best-k Selection + Spherical Clustering
- Search `k in [k_min, k_max]`.
- Feature source is configurable by `clustering.feature_strategy` (default `sentence_mean`, optional `last_token`).
- For each `k`, run spherical k-means (cosine-based clustering).
- Evaluate cluster quality using DB/CH.
- Select the `k` maximizing combined normalized score.
- Optional visualization is supported with PCA / UMAP / t-SNE (`clustering.visualization_method`) and saved to `eval/figures`.
- You can force Stage-1 visualization to use all points via `clustering.visualization_use_all_stage1: true`.

Code:
- `src/clustering/spherical_cluster.py`
- `src/utils/cluster_viz.py`

Visualization note:
- `PCA` and `UMAP` support strict Stage-1-fit / Stage-2-transform overlays, so new data can be shown in the original Stage-1 coordinate system.
- `t-SNE` is kept only as a joint refit view for intuition and is labeled `Non-Strict` in overlay plots.

### 3) Complexity Estimation and Rank Allocation
For each cluster:
- `SVD complexity(cluster)`: entropy on singular-spectrum energy.
- `SVD tail utility(cluster)`: post-min-rank spectral energy used as the marginal gain curve.

Allocate rank with a fixed total budget:
- structurally richer clusters receive larger rank,
- total sum is constrained by `rank_allocation.total_budget`,
- extra rank is assigned greedily using per-cluster marginal utility curves derived from SVD tail energy.

Code:
- `src/system/rank_allocator.py`
- `main_server_sim.py::build_rank_layout`

### 4) Expert Training + Centroid Routing
- Inject MoE-LoRA layers into target modules (`q_proj`, `v_proj`).
- One expert per cluster with allocated rank.
- Train each expert on its own cluster data (`force_expert` mode).
- Route requests to experts by cosine similarity against cluster centroids.

Code:
- `src/models/moe_layer.py`
- `main_server_sim.py::train_experts_by_clusters`
- `src/models/dynamic_router.py`

---

## Stage-2: Incremental Update Under Drift

When new private data arrives:
1. Split private data into `historical` and `new arrivals`.
2. Recompute the best cluster layout on the **current full private set** (`historical + new`).
3. Compute overlap between old/new centroids (cosine similarity).
4. Select strategy by overlap and objective (`time` vs `performance`):
   - `fast_rebuild`: high overlap, inherit nearest old experts and continue.
   - `hybrid`: mixed overlap, partial inheritance + partial fresh training.
   - `full_retrain`: low overlap with performance priority.
5. Build the Stage-2 adaptation set using:
   - all newly arrived private samples,
   - plus a sampled replay subset from historical Stage-1 clusters.
6. Rebuild experts by overlap-aware inheritance, resize inherited LoRA weights by SVD if ranks change, and continue adaptation on the replay-augmented Stage-2 data.

Code:
- `src/system/overlap_manager.py`
- `main_server_sim.py::rebuild_for_new_layout`

This provides a smoother transition between update cost and final quality while preserving previously learned private modes.

---

## Stage-3: Serving-Side System Adaptation

Different app requests may activate different LoRA expert signatures.

Scheduler policy:
- group requests by activated-LoRA signature,
- send high-parallel/high-token groups to GPU (batch benefit),
- send sparse groups to CPU (avoid GPU under-utilization).

Code:
- `src/system/hetero_batcher.py`
- `main_server_sim.py::stage3_demo_scheduler`

---

## Project Structure

- `main_server_sim.py`: end-to-end 3-stage pipeline
- `config.yaml`: experiment and system settings
- `src/clustering/spherical_cluster.py`: DB/CH k-selection + spherical k-means
- `src/system/rank_allocator.py`: fixed-budget rank allocation
- `src/system/overlap_manager.py`: old/new cluster overlap and strategy
- `src/system/hetero_batcher.py`: CPU/GPU heterogeneous request scheduler
- `src/models/moe_layer.py`: MoE-LoRA layer with expert force mode and rebuild
- `src/models/evolving_expert.py`: LoRA expert + SVD-based resize/inheritance

---

## Configuration Highlights (`config.yaml`)

- `clustering.k_min/k_max`: DB/CH search range for best k
- `rank_allocation.total_budget`: global rank budget cap
- `rank_allocation.svd_weight`: structure-aware rank utility scaling
- `stage2.optimize_for`: `time` or `performance`
- `stage2.high_overlap_threshold`: warm-start confidence threshold
- `stage2.replay_ratio`: replay size relative to newly arrived private data
- `stage2.replay_strategy`: `stratified` or `random`

---

## Run

### Default CDS pipeline

The default `config.yaml` is already set to:

- `stage0.data_dir: data/cds_cloud_pretrain`
- `data.data_dir: data/cds_private_device`
- `data.dataset_type: style_transfer_jsonl`

So after building CDS data, you can directly run:

```bash
CUDA_VISIBLE_DEVICES=0 python main_server_sim.py
```

The pipeline will automatically report:

- `raw_base`
- `stage0_warm_base`
- `Stage-1`
- `Stage-2`

for `BLEU1` and `ROUGE-L`.

For `style_transfer_jsonl` experiments, the log will also report `cluster_style_mix`,
showing the style-label composition ratio inside each discovered cluster.

### Generic run

```bash
python main_server_sim.py
```

To enable cloud warmup first, set in `config.yaml`:

```yaml
stage0:
  enable: true
  data_dir: "data/cds_cloud_pretrain"
```

If `stage0.enable` is `false`, the pipeline starts directly from the original HF base model.

For cross-app unified-device experiments, first build a composite stream directory and then switch
`config.yaml -> data.dataset_type` to `composite_device`.

Example builder usage:

```bash
python data/build_composite_stream.py \
  --manifest data/composite_manifest_example.json \
  --output_dir data/composite_device
```

One-shot auto preparation is also available:

```bash
python data/prepare_composite_experiment.py \
  --output_root data/auto_composite \
  --movielens_dir data/ml_1m \
  --yelp_review_path /path/to/yelp_academic_dataset_review.json \
  --yelp_business_path /path/to/yelp_academic_dataset_business.json
```

This helper will:
- auto-select active source users unless explicit user ids are provided,
- export a filtered Yelp user stream,
- generate `composite_manifest.json`,
- build `composite_device/train.jsonl`, `val.jsonl`, and `test.jsonl`.

By default, composite-device negatives are sampled from the same source dataset as the target item,
so the history can remain cross-app while the prediction candidates stay in-domain.

For cloud-stage pretraining on Yelp, you can build a large multi-user next-item dataset with:

```bash
python data/build_yelp_cloud_pretrain.py \
  --review_path "/remote-home/share/sunyansong/yelp/yelp/Yelp JSON/yelp_academic_dataset_review.json" \
  --business_path "/remote-home/share/sunyansong/yelp/yelp/Yelp JSON/yelp_academic_dataset_business.json" \
  --output_dir data/yelp_cloud_pretrain \
  --max_users 2000
```

### Stage-2 replay behavior

By default, Stage-2 does **not** retrain on all historical private data.
Instead it:

1. splits the private stream into `historical` and `new arrivals`,
2. recomputes the new cluster layout on the **current full private set**,
3. adapts experts on:
   - all new-arrival samples,
   - plus a replay subset sampled from historical Stage-1 clusters.

Relevant config:

```yaml
stage2:
  replay_enable: true
  replay_ratio: 0.5
  replay_strategy: "stratified"  # stratified | random
  replay_min_per_cluster: 0
```

This keeps Stage-1 capability while avoiding the cost of full historical retraining.

Example manifest:

```json
{
  "composite_user_id": "device_user_001",
  "sources": [
    {
      "type": "movielens_1m",
      "data_dir": "data/ml_1m",
      "user_id": 123,
      "source_dataset": "movielens_1m",
      "app": "movie",
      "domain": "movie",
      "item_prefix": "ml"
    },
    {
      "type": "jsonl",
      "path": "data/yelp_user_events.jsonl",
      "user_field": "user_id",
      "user_id": "abc123",
      "timestamp_field": "timestamp",
      "item_field": "business_id",
      "title_field": "title",
      "meta_field": "categories",
      "source_dataset": "yelp",
      "app": "local",
      "domain": "poi",
      "item_prefix": "yelp"
    }
  ]
}
```

Training curves are updated during run and saved to `training.curve_dir` as:
- `<curve_run_name>_training_curves.jsonl`
- `<curve_run_name>_training_curves.png`

---

## Current Notes

- This repository is a research prototype focused on methodology validation.
- Routing is centroid-based by design in the current pipeline; there is no separate learned-router training stage.
- Stage-2 now reclusters on the full current private data and adapts on `new arrivals + historical replay`; production can replace the synthetic split with real private update slices.
- Stage-1/Stage-2 checkpoints are saved to `outputs.checkpoint_root`.
- Ranking-style quality can be logged with `evaluation.enable` (`HR@K`/`NDCG@K`).

---

## Suggested Experiments

1. **Budget sweep**: vary `total_budget` and report quality/latency trade-off.
2. **Overlap-time curve**: correlate centroid overlap and update training time.
3. **Time-vs-performance policy**: compare `optimize_for=time` vs `performance` in stage-2.
4. **Serving policy**: evaluate heterogeneous scheduler under mixed request concurrency.
