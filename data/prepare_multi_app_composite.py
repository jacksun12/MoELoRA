import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

from build_composite_stream import (
    build_all_samples,
    build_meta,
    merge_sources,
    save_jsonl,
    split_events,
)


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_json(path: str, payload: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_movielens_user_counts(data_dir: str, min_rating: float) -> Counter:
    import pandas as pd

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


def choose_ranked_user(
    counts: Counter,
    explicit_user_id: Optional[str],
    rank: int,
    min_interactions: int,
    source_name: str,
) -> Tuple[str, int]:
    eligible = [(uid, cnt) for uid, cnt in counts.items() if cnt >= min_interactions]
    eligible.sort(key=lambda x: (-x[1], str(x[0])))
    if not eligible:
        raise ValueError(f"No {source_name} users satisfy min_interactions={min_interactions}.")

    if explicit_user_id is not None:
        count = counts.get(explicit_user_id, 0)
        if count < min_interactions:
            raise ValueError(
                f"{source_name} user_id={explicit_user_id} has only {count} interactions, "
                f"below min_interactions={min_interactions}."
            )
        return str(explicit_user_id), int(count)

    idx = max(0, min(rank - 1, len(eligible) - 1))
    uid, cnt = eligible[idx]
    return str(uid), int(cnt)


def choose_movielens_user(
    data_dir: str,
    min_rating: float,
    explicit_user_id: Optional[int],
    rank: int,
    min_interactions: int,
) -> Tuple[int, int]:
    counts = load_movielens_user_counts(data_dir, min_rating)
    uid, cnt = choose_ranked_user(
        counts=counts,
        explicit_user_id=str(explicit_user_id) if explicit_user_id is not None else None,
        rank=rank,
        min_interactions=min_interactions,
        source_name="MovieLens",
    )
    return int(uid), cnt


def load_generic_user_counts(path: str, user_field: str = "user_id") -> Counter:
    counts: Counter = Counter()
    for row in load_jsonl(path):
        uid = row.get(user_field)
        if uid is None:
            continue
        counts[str(uid)] += 1
    return counts


def choose_generic_user(
    path: str,
    explicit_user_id: Optional[str],
    rank: int,
    min_interactions: int,
    source_name: str,
    user_field: str = "user_id",
) -> Tuple[str, int]:
    counts = load_generic_user_counts(path, user_field=user_field)
    return choose_ranked_user(
        counts=counts,
        explicit_user_id=explicit_user_id,
        rank=rank,
        min_interactions=min_interactions,
        source_name=source_name,
    )


def build_generic_source(
    path: str,
    source_dataset: str,
    app: str,
    domain: str,
    item_prefix: str,
    user_id: str,
    user_field: str = "user_id",
    timestamp_field: str = "timestamp",
    item_field: str = "item_id",
    title_field: str = "title",
    meta_field: str = "meta",
) -> Dict:
    return {
        "type": "jsonl",
        "path": path,
        "user_field": user_field,
        "user_id": user_id,
        "timestamp_field": timestamp_field,
        "item_field": item_field,
        "title_field": title_field,
        "meta_field": meta_field,
        "source_dataset": source_dataset,
        "app": app,
        "domain": domain,
        "item_prefix": item_prefix,
    }


def maybe_add_generic_source(
    sources: List[Dict],
    selected_users: Dict,
    path: Optional[str],
    explicit_user_id: Optional[str],
    user_rank: int,
    min_interactions: int,
    source_dataset: str,
    app: str,
    domain: str,
    item_prefix: str,
    user_field: str = "user_id",
    timestamp_field: str = "timestamp",
    item_field: str = "item_id",
    title_field: str = "title",
    meta_field: str = "meta",
):
    if not path:
        return
    user_id, count = choose_generic_user(
        path=path,
        explicit_user_id=explicit_user_id,
        rank=user_rank,
        min_interactions=min_interactions,
        source_name=source_dataset,
        user_field=user_field,
    )
    selected_users[source_dataset] = {"user_id": user_id, "interactions": count}
    sources.append(
        build_generic_source(
            path=path,
            source_dataset=source_dataset,
            app=app,
            domain=domain,
            item_prefix=item_prefix,
            user_id=user_id,
            user_field=user_field,
            timestamp_field=timestamp_field,
            item_field=item_field,
            title_field=title_field,
            meta_field=meta_field,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="data/auto_composite_multiapp")
    parser.add_argument("--composite_user_id", type=str, default="device_user_multi_001")

    parser.add_argument("--movielens_dir", type=str, default="data/ml_1m")
    parser.add_argument("--movielens_user_id", type=int, default=None)
    parser.add_argument("--movielens_user_rank", type=int, default=1)
    parser.add_argument("--movielens_min_rating", type=float, default=4.0)
    parser.add_argument("--movielens_min_interactions", type=int, default=50)
    parser.add_argument("--skip_movielens", action="store_true")

    parser.add_argument("--amazon_events_path", type=str, default=None)
    parser.add_argument("--amazon_user_id", type=str, default=None)
    parser.add_argument("--amazon_user_rank", type=int, default=1)
    parser.add_argument("--amazon_min_interactions", type=int, default=30)

    parser.add_argument("--goodreads_events_path", type=str, default=None)
    parser.add_argument("--goodreads_user_id", type=str, default=None)
    parser.add_argument("--goodreads_user_rank", type=int, default=1)
    parser.add_argument("--goodreads_min_interactions", type=int, default=30)

    parser.add_argument("--yelp_events_path", type=str, default=None)
    parser.add_argument("--yelp_user_id", type=str, default=None)
    parser.add_argument("--yelp_user_rank", type=int, default=1)
    parser.add_argument("--yelp_min_interactions", type=int, default=30)

    parser.add_argument("--music_events_path", type=str, default=None)
    parser.add_argument("--music_user_id", type=str, default=None)
    parser.add_argument("--music_user_rank", type=int, default=1)
    parser.add_argument("--music_min_interactions", type=int, default=30)

    parser.add_argument("--history_size", type=int, default=10)
    parser.add_argument("--min_history", type=int, default=3)
    parser.add_argument("--neg_sample_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q_original", type=float, default=0.5)
    parser.add_argument("--q_finetune_end", type=float, default=0.8)
    parser.add_argument(
        "--negative_pool_mode",
        type=str,
        default="same_source",
        choices=["same_source", "same_domain", "global"],
    )
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    composite_dir = os.path.join(args.output_root, "composite_device_multiapp")
    os.makedirs(composite_dir, exist_ok=True)

    sources: List[Dict] = []
    selected_users: Dict[str, Optional[Dict]] = {}

    if not args.skip_movielens:
        ml_user_id, ml_count = choose_movielens_user(
            data_dir=args.movielens_dir,
            min_rating=args.movielens_min_rating,
            explicit_user_id=args.movielens_user_id,
            rank=args.movielens_user_rank,
            min_interactions=args.movielens_min_interactions,
        )
        selected_users["movielens_1m"] = {"user_id": ml_user_id, "interactions": ml_count}
        sources.append(
            {
                "type": "movielens_1m",
                "data_dir": args.movielens_dir,
                "user_id": ml_user_id,
                "source_dataset": "movielens_1m",
                "app": "video",
                "domain": "movie",
                "item_prefix": "ml",
            }
        )

    maybe_add_generic_source(
        sources=sources,
        selected_users=selected_users,
        path=args.amazon_events_path,
        explicit_user_id=args.amazon_user_id,
        user_rank=args.amazon_user_rank,
        min_interactions=args.amazon_min_interactions,
        source_dataset="amazon",
        app="shopping",
        domain="product",
        item_prefix="amazon",
    )
    maybe_add_generic_source(
        sources=sources,
        selected_users=selected_users,
        path=args.goodreads_events_path,
        explicit_user_id=args.goodreads_user_id,
        user_rank=args.goodreads_user_rank,
        min_interactions=args.goodreads_min_interactions,
        source_dataset="goodreads",
        app="reading",
        domain="book",
        item_prefix="goodreads",
    )
    maybe_add_generic_source(
        sources=sources,
        selected_users=selected_users,
        path=args.yelp_events_path,
        explicit_user_id=args.yelp_user_id,
        user_rank=args.yelp_user_rank,
        min_interactions=args.yelp_min_interactions,
        source_dataset="yelp",
        app="local",
        domain="poi",
        item_prefix="yelp",
    )
    maybe_add_generic_source(
        sources=sources,
        selected_users=selected_users,
        path=args.music_events_path,
        explicit_user_id=args.music_user_id,
        user_rank=args.music_user_rank,
        min_interactions=args.music_min_interactions,
        source_dataset="music",
        app="music",
        domain="song",
        item_prefix="music",
    )

    if len(sources) < 2:
        raise ValueError("Need at least two active sources to build a multi-app composite stream.")

    manifest = {
        "composite_user_id": args.composite_user_id,
        "sources": sources,
    }
    manifest_path = os.path.join(args.output_root, "composite_manifest_multiapp.json")
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
    meta["dataset_type"] = "composite_device"
    meta["selected_users"] = selected_users
    save_json(os.path.join(composite_dir, "meta.json"), meta)

    print(json.dumps(meta["selected_users"], ensure_ascii=False, indent=2))
    print(json.dumps(meta["sample_counts"], ensure_ascii=False, indent=2))
    print(f"Manifest: {manifest_path}")
    print(f"Composite data dir: {composite_dir}")


if __name__ == "__main__":
    main()
