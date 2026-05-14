import argparse
import json
import os
import random
from collections import Counter, defaultdict

from style_transfer_build_utils import load_jsonl, parse_mix, rebuild_samples, save_jsonl, split_rows


def main():
    parser = argparse.ArgumentParser(description="Build a more complex private CDS style-transfer dataset from local pooled data.")
    parser.add_argument("--source_dir", default="data/cds_cloud_pretrain")
    parser.add_argument("--output_dir", default="data/cds_private_device_7style")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--private_user_id", default="device_user_007style")
    parser.add_argument("--private_total_samples", type=int, default=2100)
    parser.add_argument(
        "--private_style_mix",
        default="tweets:0.18,aae:0.14,switchboard:0.12,poetry:0.12,bible:0.14,shakespeare:0.14,coha_1890:0.16",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    style_mix = parse_mix(args.private_style_mix)

    pool = []
    for split in ["train", "val", "test"]:
        path = os.path.join(args.source_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        pool.extend(load_jsonl(path))

    by_style = defaultdict(list)
    for row in pool:
        by_style[str(row.get("style_label", "unknown"))].append(row)

    available = {style: len(rows) for style, rows in by_style.items()}
    print("[Local CDS Builder] available counts:", json.dumps(available, ensure_ascii=False, indent=2))

    for style, rows in by_style.items():
        rng.shuffle(rows)

    selected = []
    selected_counts = Counter()
    remaining = int(args.private_total_samples)
    style_items = list(style_mix.items())
    for idx, (style, weight) in enumerate(style_items):
        if style not in by_style:
            raise ValueError(f"Unknown style in private_style_mix: {style}")
        take = int(round(args.private_total_samples * weight))
        if idx == len(style_items) - 1:
            take = remaining
        take = min(take, len(by_style[style]))
        remaining -= take
        chosen = by_style[style][:take]
        selected.extend(chosen)
        selected_counts[style] += len(chosen)

    rng.shuffle(selected)
    train_rows, val_rows, test_rows = split_rows(
        selected,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    train_samples = rebuild_samples(train_rows, "train", args.private_user_id, 0)
    val_samples = rebuild_samples(val_rows, "val", args.private_user_id, len(train_samples))
    test_samples = rebuild_samples(test_rows, "test", args.private_user_id, len(train_samples) + len(val_samples))

    os.makedirs(args.output_dir, exist_ok=True)
    save_jsonl(os.path.join(args.output_dir, "train.jsonl"), train_samples)
    save_jsonl(os.path.join(args.output_dir, "val.jsonl"), val_samples)
    save_jsonl(os.path.join(args.output_dir, "test.jsonl"), test_samples)

    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_type": "style_transfer_jsonl",
                "source_dataset": "cds_local_pool",
                "private_style_mix": style_mix,
                "selected_style_counts": dict(selected_counts),
                "sample_counts": {
                    "train": len(train_samples),
                    "val": len(val_samples),
                    "test": len(test_samples),
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
                "output_dir": args.output_dir,
                "selected_style_counts": dict(selected_counts),
                "sample_counts": {
                    "train": len(train_samples),
                    "val": len(val_samples),
                    "test": len(test_samples),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
