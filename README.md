# MoE-LoRA Privacy Personalization System

This project implements a **3-stage privacy personalization pipeline** for recommendation-oriented LLM adaptation on MovieLens-1M simulation data.

The system objective is:
1. Partition private data into interest regions with **DB/CH-driven k selection + spherical clustering**.
2. Measure each region difficulty by **SVD complexity + PPL**, then allocate LoRA rank under a **fixed total budget**.
3. Train one LoRA expert per cluster and a router to activate experts.
4. When new private data arrives, update cluster layout with **overlap-aware inheritance** instead of full retraining.
5. At serving time, group requests by activated LoRA signatures and schedule to **GPU/CPU heterogeneously**.

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
- Build sequence recommendation text samples from MovieLens-1M.
- Extract hidden-state features per sample.

Code:
- `data/data_loader.py`
- `main_server_sim.py::collect_features`

### 2) Best-k Selection + Spherical Clustering
- Search `k in [k_min, k_max]`.
- Feature source is configurable by `clustering.feature_strategy` (default `sentence_mean`, optional `last_token`).
- For each `k`, run spherical k-means (cosine-based clustering).
- Evaluate cluster quality using DB/CH.
- Select the `k` maximizing combined normalized score.
- Optional visualization is supported with PCA/t-SNE (`clustering.visualization_method`) and saved to `eval/figures`.
- You can force Stage-1 visualization to use all points via `clustering.visualization_use_all_stage1: true`.

Code:
- `src/clustering/spherical_cluster.py`

### 3) Complexity Estimation and Rank Allocation
For each cluster:
- `PPL(cluster)`: LM loss-based perplexity on cluster samples.
- `SVD complexity(cluster)`: entropy on singular-spectrum energy.

Combine both into normalized complexity score, then allocate rank with fixed total budget:
- higher score -> higher rank,
- total sum constrained by `rank_allocation.total_budget`.
- PPL is estimated by per-cluster sampling (`rank_allocation.ppl_max_samples_per_cluster`) to reduce overhead.
- Rank allocation uses smooth softmax weighting (`smooth_temperature`, `min_extra_share`) to avoid extreme winner-takes-all splits.

Code:
- `src/system/rank_allocator.py`
- `main_server_sim.py::build_rank_layout`

### 4) Expert Training + Router Training
- Inject MoE-LoRA layers into target modules (`q_proj`, `v_proj`).
- One expert per cluster with allocated rank.
- Train each expert on its own cluster data (`force_expert` mode).
- Train router to dispatch requests across experts.

Code:
- `src/models/moe_layer.py`
- `main_server_sim.py::train_experts_by_clusters`
- `main_server_sim.py::train_router`

---

## Stage-2: Incremental Update Under Drift

When new private data arrives:
1. Recompute best cluster layout using DB/CH + spherical clustering.
2. Compute overlap between old/new centroids (cosine similarity).
3. Select strategy by overlap and objective (`time` vs `performance`):
   - `fast_rebuild`: high overlap, inherit nearest old experts and continue.
   - `hybrid`: mixed overlap, partial inheritance + partial fresh training.
   - `full_retrain`: low overlap with performance priority.

Code:
- `src/system/overlap_manager.py`
- `main_server_sim.py::rebuild_for_new_layout`

This provides smooth transition between update cost and final quality.

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
- `rank_allocation.ppl_weight/svd_weight`: complexity fusion weights
- `stage2.optimize_for`: `time` or `performance`
- `stage2.high_overlap_threshold`: warm-start confidence threshold

---

## Run

```bash
python main_server_sim.py
```

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
- `router` training currently uses LM objective only; you can further add explicit cluster-supervision losses.
- Stage-2 currently demonstrates drift update on a sampled recent subset; production can replace this with real streaming slices.
- Stage-1/Stage-2 checkpoints are saved to `outputs.checkpoint_root`.
- Ranking-style quality can be logged with `evaluation.enable` (`HR@K`/`NDCG@K`).

---

## Suggested Experiments

1. **Budget sweep**: vary `total_budget` and report quality/latency trade-off.
2. **Overlap-time curve**: correlate centroid overlap and update training time.
3. **Time-vs-performance policy**: compare `optimize_for=time` vs `performance` in stage-2.
4. **Serving policy**: evaluate heterogeneous scheduler under mixed request concurrency.
