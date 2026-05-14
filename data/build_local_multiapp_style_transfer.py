import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List

from style_transfer_build_utils import load_jsonl, parse_mix, rebuild_samples, save_jsonl, take_weighted_rows


DEFAULT_STYLE_TO_APP = {
    "tweets": "social_post",
    "aae": "chat_reply",
    "switchboard": "voice_message",
    "poetry": "diary_note",
    "bible": "reading_note",
    "shakespeare": "movie_comment",
    "coha_1890": "shopping_review",
    "lyrics": "music_lyric",
    "coha_1810": "reading_note",
    "coha_1990": "shopping_review",
    "joyce": "diary_note",
}


def parse_style_to_app(spec: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        style, app_label = chunk.split(":")
        mapping[style.strip()] = app_label.strip()
    return mapping


def remap_row(row: Dict, style_to_app: Dict[str, str]) -> Dict:
    sample = dict(row)
    raw_style = str(row.get("style_label", "unknown"))
    if raw_style not in style_to_app:
        raise ValueError(f"Unknown raw style label: {raw_style}")
    sample["raw_style_label"] = raw_style
    sample["style_label"] = style_to_app[raw_style]
    sample["app_context_label"] = style_to_app[raw_style]
    return sample

def main():
    parser = argparse.ArgumentParser(description="Build a multi-app private style-transfer dataset from the CDS local pool.")
    parser.add_argument("--source_dir", default="data/cds_cloud_pretrain")
    parser.add_argument("--output_dir", default="data/cds_private_device_multiapp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--private_user_id", default="device_user_multiapp")
    parser.add_argument("--mode", choices=["plain", "drift"], default="plain")
    parser.add_argument("--style_to_app", default=",".join(f"{k}:{v}" for k, v in DEFAULT_STYLE_TO_APP.items()))

    parser.add_argument("--private_total_samples", type=int, default=2100)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--private_app_mix",
        default="social_post:0.16,chat_reply:0.12,voice_message:0.10,diary_note:0.10,reading_note:0.12,movie_comment:0.12,shopping_review:0.14,music_lyric:0.14",
    )

    parser.add_argument("--train_total", type=int, default=1920)
    parser.add_argument("--val_total", type=int, default=240)
    parser.add_argument("--test_total", type=int, default=240)
    parser.add_argument("--train_stage1_ratio", type=float, default=0.75)
    parser.add_argument("--test_stage1_ratio", type=float, default=0.75)
    parser.add_argument(
        "--stage1_app_mix",
        default="social_post:0.28,chat_reply:0.16,voice_message:0.16,diary_note:0.08,reading_note:0.05,movie_comment:0.03,shopping_review:0.08,music_lyric:0.16",
    )
    parser.add_argument(
        "--stage2_app_mix",
        default="social_post:0.05,chat_reply:0.03,voice_message:0.04,diary_note:0.16,reading_note:0.18,movie_comment:0.16,shopping_review:0.24,music_lyric:0.14",
    )
    parser.add_argument(
        "--val_app_mix",
        default="social_post:0.16,chat_reply:0.10,voice_message:0.10,diary_note:0.12,reading_note:0.12,movie_comment:0.12,shopping_review:0.14,music_lyric:0.14",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    style_to_app = parse_style_to_app(args.style_to_app)

    pool: List[Dict] = []
    for split in ["train", "val", "test"]:
        path = os.path.join(args.source_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        pool.extend(load_jsonl(path))

    remapped_pool = [remap_row(row, style_to_app) for row in pool]
    by_app: Dict[str, List[Dict]] = defaultdict(list)
    by_raw_style: Counter = Counter()
    for row in remapped_pool:
        by_app[str(row["style_label"])].append(row)
        by_raw_style[str(row["raw_style_label"])] += 1

    for app_label, rows in by_app.items():
        rng.shuffle(rows)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "plain":
        app_mix = parse_mix(args.private_app_mix)
        selected: List[Dict] = []
        selected_counts: Counter = Counter()
        remaining = int(args.private_total_samples)
        items = list(app_mix.items())
        for idx, (app_label, weight) in enumerate(items):
            want = int(round(args.private_total_samples * weight))
            if idx == len(items) - 1:
                want = remaining
            want = min(want, len(by_app[app_label]))
            remaining -= want
            chosen = by_app[app_label][:want]
            selected.extend(chosen)
            selected_counts[app_label] += len(chosen)

        rng.shuffle(selected)
        n = len(selected)
        train_end = int(n * args.train_ratio)
        val_end = train_end + int(n * args.val_ratio)
        train_rows = selected[:train_end]
        val_rows = selected[train_end:val_end]
        test_rows = selected[val_end:]

        train_samples = rebuild_samples(train_rows, "train", args.private_user_id, 0)
        val_samples = rebuild_samples(val_rows, "val", args.private_user_id, len(train_samples))
        test_samples = rebuild_samples(test_rows, "test", args.private_user_id, len(train_samples) + len(val_samples))

        save_jsonl(os.path.join(args.output_dir, "train.jsonl"), train_samples)
        save_jsonl(os.path.join(args.output_dir, "val.jsonl"), val_samples)
        save_jsonl(os.path.join(args.output_dir, "test.jsonl"), test_samples)

        meta = {
            "dataset_type": "style_transfer_jsonl",
            "source_dataset": "cds_local_pool_multiapp_proxy",
            "mode": "plain",
            "style_to_app": style_to_app,
            "raw_style_counts": dict(by_raw_style),
            "selected_app_counts": dict(selected_counts),
            "sample_counts": {
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples),
            },
            "config": vars(args),
        }
    else:
        stage1_mix = parse_mix(args.stage1_app_mix)
        stage2_mix = parse_mix(args.stage2_app_mix)
        val_mix = parse_mix(args.val_app_mix)
        cursor: Dict[str, int] = defaultdict(int)

        train_stage1_n = int(round(args.train_total * args.train_stage1_ratio))
        train_stage2_n = int(args.train_total - train_stage1_n)
        test_stage1_n = int(round(args.test_total * args.test_stage1_ratio))
        test_stage2_n = int(args.test_total - test_stage1_n)

        train_stage1_rows, train_stage1_counts = take_weighted_rows(stage1_mix, by_app, cursor, train_stage1_n, "app_label")
        train_stage2_rows, train_stage2_counts = take_weighted_rows(stage2_mix, by_app, cursor, train_stage2_n, "app_label")
        val_rows, val_counts = take_weighted_rows(val_mix, by_app, cursor, args.val_total, "app_label")
        test_stage1_rows, test_stage1_counts = take_weighted_rows(stage1_mix, by_app, cursor, test_stage1_n, "app_label")
        test_stage2_rows, test_stage2_counts = take_weighted_rows(stage2_mix, by_app, cursor, test_stage2_n, "app_label")

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

        save_jsonl(os.path.join(args.output_dir, "train.jsonl"), train_samples)
        save_jsonl(os.path.join(args.output_dir, "val.jsonl"), val_samples)
        save_jsonl(os.path.join(args.output_dir, "test.jsonl"), test_samples)

        meta = {
            "dataset_type": "style_transfer_jsonl",
            "source_dataset": "cds_local_pool_multiapp_proxy",
            "mode": "drift",
            "style_to_app": style_to_app,
            "raw_style_counts": dict(by_raw_style),
            "sample_counts": {
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples),
            },
            "train_stage1_count": train_stage1_n,
            "train_stage2_count": train_stage2_n,
            "test_stage1_count": test_stage1_n,
            "test_stage2_count": test_stage2_n,
            "stage1_app_mix": stage1_mix,
            "stage2_app_mix": stage2_mix,
            "val_app_mix": val_mix,
            "train_stage1_app_counts": dict(train_stage1_counts),
            "train_stage2_app_counts": dict(train_stage2_counts),
            "val_app_counts": dict(val_counts),
            "test_stage1_app_counts": dict(test_stage1_counts),
            "test_stage2_app_counts": dict(test_stage2_counts),
            "config": vars(args),
        }

    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps({"output_dir": args.output_dir, **meta}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
