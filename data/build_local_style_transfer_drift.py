import argparse
import json
import os
import random
from collections import defaultdict

from style_transfer_build_utils import load_jsonl, parse_mix, rebuild_samples, save_jsonl, take_weighted_rows


def main():
    parser = argparse.ArgumentParser(description="Build a local CDS private dataset with explicit temporal distribution drift.")
    parser.add_argument("--source_dir", default="data/cds_cloud_pretrain")
    parser.add_argument("--output_dir", default="data/cds_private_device_7style_drift")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--private_user_id", default="device_user_drift")
    parser.add_argument("--train_total", type=int, default=1920)
    parser.add_argument("--val_total", type=int, default=240)
    parser.add_argument("--test_total", type=int, default=240)
    parser.add_argument("--train_stage1_ratio", type=float, default=0.75)
    parser.add_argument("--test_stage1_ratio", type=float, default=0.75)
    parser.add_argument(
        "--stage1_style_mix",
        default="tweets:0.34,aae:0.20,switchboard:0.18,poetry:0.10,bible:0.05,shakespeare:0.03,coha_1890:0.10",
    )
    parser.add_argument(
        "--stage2_style_mix",
        default="tweets:0.06,aae:0.04,switchboard:0.06,poetry:0.18,bible:0.20,shakespeare:0.18,coha_1890:0.28",
    )
    parser.add_argument(
        "--val_style_mix",
        default="tweets:0.20,aae:0.12,switchboard:0.12,poetry:0.14,bible:0.14,shakespeare:0.12,coha_1890:0.16",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    stage1_mix = parse_mix(args.stage1_style_mix)
    stage2_mix = parse_mix(args.stage2_style_mix)
    val_mix = parse_mix(args.val_style_mix)

    pool = []
    for split in ["train", "val", "test"]:
        path = os.path.join(args.source_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        pool.extend(load_jsonl(path))

    by_style = defaultdict(list)
    for row in pool:
        by_style[str(row.get("style_label", "unknown"))].append(row)

    for style, rows in by_style.items():
        rng.shuffle(rows)

    cursor = defaultdict(int)

    train_stage1_n = int(round(args.train_total * args.train_stage1_ratio))
    train_stage2_n = int(args.train_total - train_stage1_n)
    test_stage1_n = int(round(args.test_total * args.test_stage1_ratio))
    test_stage2_n = int(args.test_total - test_stage1_n)

    train_stage1_rows, train_stage1_counts = take_weighted_rows(stage1_mix, by_style, cursor, train_stage1_n, "style")
    train_stage2_rows, train_stage2_counts = take_weighted_rows(stage2_mix, by_style, cursor, train_stage2_n, "style")
    val_rows, val_counts = take_weighted_rows(val_mix, by_style, cursor, args.val_total, "style")
    test_stage1_rows, test_stage1_counts = take_weighted_rows(stage1_mix, by_style, cursor, test_stage1_n, "style")
    test_stage2_rows, test_stage2_counts = take_weighted_rows(stage2_mix, by_style, cursor, test_stage2_n, "style")

    rng.shuffle(train_stage1_rows)
    rng.shuffle(train_stage2_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_stage1_rows)
    rng.shuffle(test_stage2_rows)

    train_rows = train_stage1_rows + train_stage2_rows
    test_rows = test_stage1_rows + test_stage2_rows

    train_samples = rebuild_samples(train_rows, "train", args.private_user_id, 0)
    val_samples = rebuild_samples(val_rows, "val", args.private_user_id, len(train_samples))
    test_samples = rebuild_samples(test_rows, "test", args.private_user_id, len(train_samples) + len(val_samples))

    os.makedirs(args.output_dir, exist_ok=True)
    save_jsonl(os.path.join(args.output_dir, "train.jsonl"), train_samples)
    save_jsonl(os.path.join(args.output_dir, "val.jsonl"), val_samples)
    save_jsonl(os.path.join(args.output_dir, "test.jsonl"), test_samples)

    meta = {
        "dataset_type": "style_transfer_jsonl",
        "source_dataset": "cds_local_pool",
        "sample_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "train_stage1_count": train_stage1_n,
        "train_stage2_count": train_stage2_n,
        "test_stage1_count": test_stage1_n,
        "test_stage2_count": test_stage2_n,
        "stage1_style_mix": stage1_mix,
        "stage2_style_mix": stage2_mix,
        "val_style_mix": val_mix,
        "train_stage1_style_counts": dict(train_stage1_counts),
        "train_stage2_style_counts": dict(train_stage2_counts),
        "val_style_counts": dict(val_counts),
        "test_stage1_style_counts": dict(test_stage1_counts),
        "test_stage2_style_counts": dict(test_stage2_counts),
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps({"output_dir": args.output_dir, **meta}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
