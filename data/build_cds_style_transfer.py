import argparse
import json
import os
import random
from collections import Counter, defaultdict

from datasets import load_dataset
from tqdm import tqdm


def save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_style_mix(spec):
    weights = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        style, weight = chunk.split(":")
        weights[style.strip()] = float(weight)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("style mix must have positive total weight")
    return {k: v / total for k, v in weights.items()}


def split_rows(rows, train_ratio=0.8, val_ratio=0.1):
    n = len(rows)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def make_sample(row, sample_id, user_id, timestamp):
    return {
        "sample_id": sample_id,
        "user_id": user_id,
        "timestamp": timestamp,
        "style_label": row["label"],
        "source_text": row["paraphrase"].strip(),
        "target_text": row["text"].strip(),
        "source_dataset": "cds",
        "task_type": "style_transfer",
    }


def rows_to_samples(rows, user_fn, split_name, ts_start=0):
    samples = []
    for i, row in enumerate(rows):
        user_id = user_fn(row, i)
        samples.append(
            make_sample(
                row=row,
                sample_id=f"{split_name}_{i+1:07d}",
                user_id=user_id,
                timestamp=ts_start + i,
            )
        )
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="billray110/corpus-of-diverse-styles")
    parser.add_argument("--cloud_output_dir", type=str, default="data/cds_cloud_pretrain")
    parser.add_argument("--private_output_dir", type=str, default="data/cds_private_device")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_cloud_samples_per_style", type=int, default=30000)
    parser.add_argument("--cloud_users_per_style", type=int, default=20)
    parser.add_argument("--private_user_id", type=str, default="device_user_001")
    parser.add_argument("--private_total_samples", type=int, default=6000)
    parser.add_argument("--private_style_mix", type=str, default="tweets:0.5,switchboard:0.3,poetry:0.2")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--min_chars", type=int, default=8)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    style_mix = parse_style_mix(args.private_style_mix)

    print(f"[CDS Builder] loading dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name)["train"]
    print(ds)

    style_rows = defaultdict(list)
    for row in tqdm(ds, desc="Group CDS rows", dynamic_ncols=True):
        source_text = str(row["paraphrase"]).strip()
        target_text = str(row["text"]).strip()
        if len(source_text) < args.min_chars or len(target_text) < args.min_chars:
            continue
        style_rows[str(row["label"])].append(
            {
                "label": str(row["label"]),
                "paraphrase": source_text,
                "text": target_text,
            }
        )

    counts = {style: len(rows) for style, rows in style_rows.items()}
    print("[CDS Builder] style counts:", json.dumps(counts, indent=2, ensure_ascii=False))

    for style, rows in style_rows.items():
        rng.shuffle(rows)

    private_rows = []
    private_counts = Counter()
    requested_private = {}
    allocated_by_style = defaultdict(int)

    remaining = args.private_total_samples
    styles = list(style_mix.items())
    for idx, (style, weight) in enumerate(styles):
        if style not in style_rows:
            raise ValueError(f"Unknown style in private_style_mix: {style}")
        take = int(round(args.private_total_samples * weight))
        if idx == len(styles) - 1:
            take = remaining
        take = min(take, len(style_rows[style]))
        remaining -= take
        requested_private[style] = take
        private_rows.extend(style_rows[style][:take])
        allocated_by_style[style] += take
        private_counts[style] += take

    print("[CDS Builder] private style allocation:", dict(private_counts))

    for style, rows in style_rows.items():
        style_rows[style] = rows[allocated_by_style[style]:]

    rng.shuffle(private_rows)
    private_train, private_val, private_test = split_rows(
        private_rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    private_train_samples = rows_to_samples(
        private_train,
        user_fn=lambda _row, _i: args.private_user_id,
        split_name="train",
        ts_start=0,
    )
    private_val_samples = rows_to_samples(
        private_val,
        user_fn=lambda _row, _i: args.private_user_id,
        split_name="val",
        ts_start=len(private_train_samples),
    )
    private_test_samples = rows_to_samples(
        private_test,
        user_fn=lambda _row, _i: args.private_user_id,
        split_name="test",
        ts_start=len(private_train_samples) + len(private_val_samples),
    )

    cloud_pool = []
    for style, rows in style_rows.items():
        cap = len(rows)
        if args.max_cloud_samples_per_style > 0:
            cap = min(cap, args.max_cloud_samples_per_style)
        cloud_pool.extend(rows[:cap])
    rng.shuffle(cloud_pool)
    cloud_train, cloud_val, cloud_test = split_rows(
        cloud_pool,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    cloud_user_mod = max(1, int(args.cloud_users_per_style))
    cloud_train_samples = rows_to_samples(
        cloud_train,
        user_fn=lambda row, i: f"cloud_{row['label']}_{i % cloud_user_mod:03d}",
        split_name="train",
        ts_start=0,
    )
    cloud_val_samples = rows_to_samples(
        cloud_val,
        user_fn=lambda row, i: f"cloud_{row['label']}_{i % cloud_user_mod:03d}",
        split_name="val",
        ts_start=len(cloud_train_samples),
    )
    cloud_test_samples = rows_to_samples(
        cloud_test,
        user_fn=lambda row, i: f"cloud_{row['label']}_{i % cloud_user_mod:03d}",
        split_name="test",
        ts_start=len(cloud_train_samples) + len(cloud_val_samples),
    )

    os.makedirs(args.cloud_output_dir, exist_ok=True)
    save_jsonl(os.path.join(args.cloud_output_dir, "train.jsonl"), cloud_train_samples)
    save_jsonl(os.path.join(args.cloud_output_dir, "val.jsonl"), cloud_val_samples)
    save_jsonl(os.path.join(args.cloud_output_dir, "test.jsonl"), cloud_test_samples)
    with open(os.path.join(args.cloud_output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_type": "style_transfer_jsonl",
                "source_dataset": "cds",
                "sample_counts": {
                    "train": len(cloud_train_samples),
                    "val": len(cloud_val_samples),
                    "test": len(cloud_test_samples),
                },
                "config": vars(args),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.makedirs(args.private_output_dir, exist_ok=True)
    save_jsonl(os.path.join(args.private_output_dir, "train.jsonl"), private_train_samples)
    save_jsonl(os.path.join(args.private_output_dir, "val.jsonl"), private_val_samples)
    save_jsonl(os.path.join(args.private_output_dir, "test.jsonl"), private_test_samples)
    with open(os.path.join(args.private_output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_type": "style_transfer_jsonl",
                "source_dataset": "cds",
                "private_style_mix": style_mix,
                "sample_counts": {
                    "train": len(private_train_samples),
                    "val": len(private_val_samples),
                    "test": len(private_test_samples),
                },
                "config": vars(args),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        json.dumps(
            {
                "cloud_counts": {
                    "train": len(cloud_train_samples),
                    "val": len(cloud_val_samples),
                    "test": len(cloud_test_samples),
                },
                "private_counts": {
                    "train": len(private_train_samples),
                    "val": len(private_val_samples),
                    "test": len(private_test_samples),
                },
                "private_style_mix": style_mix,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
