import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd

from build_composite_stream import (
    build_all_samples,
    build_meta,
    merge_sources,
    save_jsonl,
    split_events,
)


def load_movielens_user_counts(data_dir: str, min_rating: float) -> Counter:
    ratings_file = os.path.join(data_dir, "ratings.dat")
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(ratings_file)

    df = pd.read_csv(
        ratings_file,
        sep="::",
        header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        engine="python",
    )
    df = df[df["Rating"] >= min_rating]
    return Counter(df["UserID"].tolist())


def choose_movielens_user(
    data_dir: str,
    min_rating: float,
    explicit_user_id: Optional[int],
    rank: int,
    min_interactions: int,
) -> Tuple[int, int]:
    counts = load_movielens_user_counts(data_dir, min_rating)
    eligible = [(uid, cnt) for uid, cnt in counts.items() if cnt >= min_interactions]
    eligible.sort(key=lambda x: (-x[1], x[0]))
    if not eligible:
        raise ValueError("No MovieLens users satisfy the min_interactions filter.")

    if explicit_user_id is not None:
        count = counts.get(explicit_user_id, 0)
        if count < min_interactions:
            raise ValueError(
                f"MovieLens user_id={explicit_user_id} has only {count} interactions, "
                f"below min_interactions={min_interactions}."
            )
        return explicit_user_id, count

    idx = max(0, min(rank - 1, len(eligible) - 1))
    return eligible[idx]


def load_yelp_business_map(path: str) -> Dict[str, Dict[str, str]]:
    business_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            bid = row.get("business_id")
            if not bid:
                continue
            business_map[bid] = {
                "title": row.get("name", f"business:{bid}"),
                "categories": row.get("categories") or "",
            }
    return business_map


def load_yelp_user_counts(review_path: str, min_stars: float) -> Counter:
    counts = Counter()
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stars = float(row.get("stars", 0.0))
            if stars < min_stars:
                continue
            uid = row.get("user_id")
            if uid:
                counts[uid] += 1
    return counts


def choose_yelp_user(
    review_path: str,
    min_stars: float,
    explicit_user_id: Optional[str],
    rank: int,
    min_interactions: int,
) -> Tuple[str, int]:
    counts = load_yelp_user_counts(review_path, min_stars)
    eligible = [(uid, cnt) for uid, cnt in counts.items() if cnt >= min_interactions]
    eligible.sort(key=lambda x: (-x[1], x[0]))
    if not eligible:
        raise ValueError("No Yelp users satisfy the min_interactions filter.")

    if explicit_user_id is not None:
        count = counts.get(explicit_user_id, 0)
        if count < min_interactions:
            raise ValueError(
                f"Yelp user_id={explicit_user_id} has only {count} interactions, "
                f"below min_interactions={min_interactions}."
            )
        return explicit_user_id, count

    idx = max(0, min(rank - 1, len(eligible) - 1))
    return eligible[idx]


def export_yelp_user_events(
    review_path: str,
    business_path: str,
    output_path: str,
    user_id: str,
    min_stars: float,
) -> int:
    business_map = load_yelp_business_map(business_path)
    rows: List[Dict] = []
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("user_id")) != str(user_id):
                continue
            stars = float(row.get("stars", 0.0))
            if stars < min_stars:
                continue
            bid = row.get("business_id")
            if not bid:
                continue
            business_meta = business_map.get(bid, {})
            rows.append(
                {
                    "user_id": user_id,
                    "timestamp": int(row.get("date", "1970-01-01 00:00:00").replace("-", "").replace(":", "").replace(" ", "")),
                    "business_id": bid,
                    "title": business_meta.get("title", f"business:{bid}"),
                    "categories": business_meta.get("categories", ""),
                    "stars": stars,
                }
            )
    rows.sort(key=lambda x: (x["timestamp"], x["business_id"]))
    save_jsonl(output_path, rows)
    return len(rows)


def build_manifest(
    composite_user_id: str,
    movielens_dir: str,
    movielens_user_id: Optional[int],
    yelp_events_path: Optional[str],
    yelp_user_id: Optional[str],
) -> Dict:
    sources = []
    if movielens_user_id is not None:
        sources.append(
            {
                "type": "movielens_1m",
                "data_dir": movielens_dir,
                "user_id": movielens_user_id,
                "source_dataset": "movielens_1m",
                "app": "movie",
                "domain": "movie",
                "item_prefix": "ml",
            }
        )
    if yelp_events_path is not None and yelp_user_id is not None:
        sources.append(
            {
                "type": "jsonl",
                "path": yelp_events_path,
                "user_field": "user_id",
                "user_id": yelp_user_id,
                "timestamp_field": "timestamp",
                "item_field": "business_id",
                "title_field": "title",
                "meta_field": "categories",
                "source_dataset": "yelp",
                "app": "local",
                "domain": "poi",
                "item_prefix": "yelp",
            }
        )
    return {
        "composite_user_id": composite_user_id,
        "sources": sources,
    }


def save_json(path: str, payload: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="data/auto_composite")
    parser.add_argument("--composite_user_id", type=str, default="device_user_001")

    parser.add_argument("--movielens_dir", type=str, default="data/ml_1m")
    parser.add_argument("--movielens_user_id", type=int, default=None)
    parser.add_argument("--movielens_user_rank", type=int, default=1)
    parser.add_argument("--movielens_min_rating", type=float, default=4.0)
    parser.add_argument("--movielens_min_interactions", type=int, default=50)

    parser.add_argument("--yelp_review_path", type=str, default=None)
    parser.add_argument("--yelp_business_path", type=str, default=None)
    parser.add_argument("--yelp_user_id", type=str, default=None)
    parser.add_argument("--yelp_user_rank", type=int, default=1)
    parser.add_argument("--yelp_min_stars", type=float, default=4.0)
    parser.add_argument("--yelp_min_interactions", type=int, default=50)
    parser.add_argument("--skip_yelp", action="store_true")

    parser.add_argument("--history_size", type=int, default=10)
    parser.add_argument("--min_history", type=int, default=3)
    parser.add_argument("--neg_sample_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q_original", type=float, default=0.5)
    parser.add_argument("--q_finetune_end", type=float, default=0.8)
    parser.add_argument("--negative_pool_mode", type=str, default="same_source", choices=["same_source", "same_domain", "global"])
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    prepared_dir = os.path.join(args.output_root, "prepared_sources")
    composite_dir = os.path.join(args.output_root, "composite_device")
    os.makedirs(prepared_dir, exist_ok=True)
    os.makedirs(composite_dir, exist_ok=True)

    ml_user_id, ml_count = choose_movielens_user(
        data_dir=args.movielens_dir,
        min_rating=args.movielens_min_rating,
        explicit_user_id=args.movielens_user_id,
        rank=args.movielens_user_rank,
        min_interactions=args.movielens_min_interactions,
    )

    yelp_events_path = None
    yelp_user_id = None
    yelp_count = None
    if not args.skip_yelp:
        if not args.yelp_review_path or not args.yelp_business_path:
            raise ValueError("Yelp is enabled, so both --yelp_review_path and --yelp_business_path are required.")
        yelp_user_id, yelp_count = choose_yelp_user(
            review_path=args.yelp_review_path,
            min_stars=args.yelp_min_stars,
            explicit_user_id=args.yelp_user_id,
            rank=args.yelp_user_rank,
            min_interactions=args.yelp_min_interactions,
        )
        yelp_events_path = os.path.join(prepared_dir, "yelp_user_events.jsonl")
        export_yelp_user_events(
            review_path=args.yelp_review_path,
            business_path=args.yelp_business_path,
            output_path=yelp_events_path,
            user_id=yelp_user_id,
            min_stars=args.yelp_min_stars,
        )

    manifest = build_manifest(
        composite_user_id=args.composite_user_id,
        movielens_dir=args.movielens_dir,
        movielens_user_id=ml_user_id,
        yelp_events_path=yelp_events_path,
        yelp_user_id=yelp_user_id,
    )
    manifest_path = os.path.join(args.output_root, "composite_manifest.json")
    save_json(manifest_path, manifest)

    merged_events = merge_sources(manifest["sources"], composite_user_id=args.composite_user_id)
    events_by_split = split_events(
        merged_events,
        q_original=args.q_original,
        q_finetune_end=args.q_finetune_end,
    )
    samples_by_split = build_all_samples(
        events_by_split=events_by_split,
        history_size=args.history_size,
        min_history=args.min_history,
        neg_sample_size=args.neg_sample_size,
        seed=args.seed,
        negative_pool_mode=args.negative_pool_mode,
    )

    save_jsonl(os.path.join(composite_dir, "events_all.jsonl"), merged_events)
    for split_name, rows in events_by_split.items():
        save_jsonl(os.path.join(composite_dir, f"{split_name}_events.jsonl"), rows)
    for split_name, rows in samples_by_split.items():
        save_jsonl(os.path.join(composite_dir, f"{split_name}.jsonl"), rows)

    meta = build_meta(manifest, events_by_split, samples_by_split)
    meta["selected_users"] = {
        "movielens": {"user_id": ml_user_id, "interactions": ml_count},
        "yelp": (
            {"user_id": yelp_user_id, "interactions": yelp_count}
            if yelp_user_id is not None
            else None
        ),
    }
    save_json(os.path.join(composite_dir, "meta.json"), meta)

    print(json.dumps(meta["selected_users"], ensure_ascii=False, indent=2))
    print(json.dumps(meta["sample_counts"], ensure_ascii=False, indent=2))
    print(f"Manifest: {manifest_path}")
    print(f"Composite data dir: {composite_dir}")


if __name__ == "__main__":
    main()
