import argparse
import json
import os
import random
from typing import Dict, List

import pandas as pd


def save_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_events(events: List[Dict], source_idx: int) -> List[Dict]:
    if not events:
        return []
    events = sorted(events, key=lambda x: (x["timestamp"], x["event_local_idx"]))
    ts_min = events[0]["timestamp"]
    ts_max = events[-1]["timestamp"]
    span = max(ts_max - ts_min, 1)
    for rank, event in enumerate(events):
        if len(events) == 1:
            norm_pos = 0.0
        else:
            norm_pos = (event["timestamp"] - ts_min) / span
        event["normalized_time"] = float(norm_pos)
        event["merge_key"] = float(norm_pos) + source_idx * 1e-6 + rank * 1e-9
    return events


def load_movielens_source(spec: Dict, source_idx: int) -> List[Dict]:
    data_dir = spec["data_dir"]
    user_id = int(spec["user_id"])
    min_rating = float(spec.get("min_rating", 4.0))
    item_prefix = spec.get("item_prefix", "ml")
    source_name = spec.get("source_dataset", "movielens_1m")
    app = spec.get("app", "movie")
    domain = spec.get("domain", "movie")

    ratings_file = os.path.join(data_dir, "ratings.dat")
    movies_file = os.path.join(data_dir, "movies.dat")
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(ratings_file)
    if not os.path.exists(movies_file):
        raise FileNotFoundError(movies_file)

    movies_df = pd.read_csv(
        movies_file,
        sep="::",
        header=None,
        names=["MovieID", "Title", "Genres"],
        engine="python",
        encoding="latin-1",
    )
    movie_map = {
        int(r["MovieID"]): {"title": r["Title"], "meta": r["Genres"]}
        for _, r in movies_df.iterrows()
    }

    ratings_df = pd.read_csv(
        ratings_file,
        sep="::",
        header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        engine="python",
    )
    ratings_df = ratings_df[(ratings_df["UserID"] == user_id) & (ratings_df["Rating"] >= min_rating)]
    ratings_df = ratings_df.sort_values(by="Timestamp")
    if ratings_df.empty:
        raise ValueError(f"No qualifying MovieLens interactions found for user_id={user_id}")

    events = []
    seen = set()
    for local_idx, row in enumerate(ratings_df.itertuples(index=False), start=1):
        movie_id = int(row.MovieID)
        if movie_id in seen and spec.get("deduplicate", True):
            continue
        seen.add(movie_id)
        meta = movie_map.get(movie_id, {})
        events.append(
            {
                "event_local_idx": local_idx,
                "source_dataset": source_name,
                "app": app,
                "domain": domain,
                "timestamp": int(row.Timestamp),
                "item_id": f"{item_prefix}_{movie_id}",
                "title": meta.get("title", f"Movie {movie_id}"),
                "meta": meta.get("meta", "Unknown"),
            }
        )
    return _normalize_events(events, source_idx)


def load_generic_jsonl_source(spec: Dict, source_idx: int) -> List[Dict]:
    path = spec["path"]
    rows = load_jsonl(path)
    user_id = spec.get("user_id")
    user_field = spec.get("user_field")
    ts_field = spec.get("timestamp_field", "timestamp")
    item_field = spec.get("item_field", "item_id")
    title_field = spec.get("title_field")
    meta_field = spec.get("meta_field")
    source_name = spec["source_dataset"]
    app = spec.get("app", source_name)
    domain = spec.get("domain", source_name)
    item_prefix = spec.get("item_prefix", source_name)

    events = []
    for local_idx, row in enumerate(rows, start=1):
        if user_field is not None and user_id is not None and str(row.get(user_field)) != str(user_id):
            continue
        item_raw = row.get(item_field)
        ts_raw = row.get(ts_field)
        if item_raw is None or ts_raw is None:
            continue
        title = row.get(title_field) if title_field else None
        meta = row.get(meta_field) if meta_field else None
        events.append(
            {
                "event_local_idx": local_idx,
                "source_dataset": source_name,
                "app": app,
                "domain": domain,
                "timestamp": int(ts_raw),
                "item_id": f"{item_prefix}_{item_raw}",
                "title": str(title) if title is not None else f"{domain}:{item_raw}",
                "meta": str(meta) if meta is not None else "",
            }
        )

    if not events:
        raise ValueError(f"No qualifying JSONL interactions found for source={source_name}")
    return _normalize_events(events, source_idx)


def load_source_events(spec: Dict, source_idx: int) -> List[Dict]:
    source_type = spec["type"].lower()
    if source_type in {"movielens_1m", "movielens", "ml_1m"}:
        return load_movielens_source(spec, source_idx)
    if source_type in {"jsonl", "generic_jsonl"}:
        return load_generic_jsonl_source(spec, source_idx)
    raise ValueError(f"Unsupported source type: {source_type}")


def merge_sources(sources: List[Dict], composite_user_id: str) -> List[Dict]:
    merged = []
    for source_idx, spec in enumerate(sources):
        merged.extend(load_source_events(spec, source_idx))
    merged.sort(key=lambda x: (x["merge_key"], x["source_dataset"], x["event_local_idx"]))

    timeline_scale = 1_000_000
    final_events = []
    for event_idx, event in enumerate(merged, start=1):
        final_events.append(
            {
                "event_id": f"evt_{event_idx:06d}",
                "user_id": composite_user_id,
                "timestamp": int(event["merge_key"] * timeline_scale),
                "source_dataset": event["source_dataset"],
                "app": event["app"],
                "domain": event["domain"],
                "item_id": event["item_id"],
                "title": event["title"],
                "meta": event["meta"],
            }
        )
    return final_events


def split_events(events: List[Dict], q_original: float, q_finetune_end: float):
    n = len(events)
    if n < 3:
        raise ValueError("Need at least 3 merged events to build original/finetune/test splits.")
    orig_end = max(1, int(n * q_original))
    ft_end = max(orig_end + 1, int(n * q_finetune_end))
    ft_end = min(ft_end, n - 1)
    return {
        "original": events[:orig_end],
        "finetune": events[orig_end:ft_end],
        "test": events[ft_end:],
    }


def _sample_to_item(event: Dict) -> Dict:
    return {
        "item_id": event["item_id"],
        "title": event["title"],
        "domain": event["domain"],
        "source_dataset": event["source_dataset"],
        "app": event["app"],
        "meta": event.get("meta", ""),
    }


def build_samples_for_segment(
    events: List[Dict],
    item_catalog: Dict[str, Dict],
    split_name: str,
    history_size: int,
    min_history: int,
    neg_sample_size: int,
    rng: random.Random,
    negative_pool_mode: str = "same_source",
) -> List[Dict]:
    if len(events) < max(history_size + 1, min_history + 1):
        return []

    samples = []
    all_item_ids = list(item_catalog.keys())
    for i in range(min_history, len(events)):
        history_events = events[max(0, i - history_size) : i]
        target_event = events[i]
        history_ids = [e["item_id"] for e in history_events]
        forbidden = set(history_ids)
        forbidden.add(target_event["item_id"])
        if negative_pool_mode == "same_source":
            negative_pool = [
                item_id
                for item_id in all_item_ids
                if item_id not in forbidden
                and item_catalog[item_id]["source_dataset"] == target_event["source_dataset"]
            ]
        elif negative_pool_mode == "same_domain":
            negative_pool = [
                item_id
                for item_id in all_item_ids
                if item_id not in forbidden
                and item_catalog[item_id]["domain"] == target_event["domain"]
            ]
        else:
            negative_pool = [item_id for item_id in all_item_ids if item_id not in forbidden]
        if len(negative_pool) < neg_sample_size:
            continue

        negative_ids = rng.sample(negative_pool, neg_sample_size)
        candidate_ids = negative_ids + [target_event["item_id"]]
        rng.shuffle(candidate_ids)
        target_option_idx = candidate_ids.index(target_event["item_id"])

        samples.append(
            {
                "sample_id": f"{split_name}_{len(samples) + 1:06d}",
                "user_id": target_event["user_id"],
                "timestamp": target_event["timestamp"],
                "split": split_name,
                "target_domain": target_event["domain"],
                "target_source_dataset": target_event["source_dataset"],
                "history_items": [_sample_to_item(e) for e in history_events],
                "candidate_items": [item_catalog[item_id] for item_id in candidate_ids],
                "target_option_idx": target_option_idx,
            }
        )
    return samples


def build_all_samples(
    events_by_split: Dict[str, List[Dict]],
    history_size: int,
    min_history: int,
    neg_sample_size: int,
    seed: int,
    negative_pool_mode: str = "same_source",
) -> Dict[str, List[Dict]]:
    item_catalog = {}
    for rows in events_by_split.values():
        for event in rows:
            item_catalog[event["item_id"]] = _sample_to_item(event)

    rng = random.Random(seed)
    outputs = {}
    for split_name, rows in events_by_split.items():
        outputs[split_name] = build_samples_for_segment(
            events=rows,
            item_catalog=item_catalog,
            split_name=split_name,
            history_size=history_size,
            min_history=min_history,
            neg_sample_size=neg_sample_size,
            rng=rng,
            negative_pool_mode=negative_pool_mode,
        )

    merged_train = sorted(
        outputs["original"] + outputs["finetune"],
        key=lambda x: (x["timestamp"], x["sample_id"]),
    )
    outputs["train"] = merged_train
    outputs["val"] = list(outputs["finetune"])
    return outputs


def build_meta(manifest: Dict, events_by_split: Dict[str, List[Dict]], samples_by_split: Dict[str, List[Dict]]) -> Dict:
    return {
        "composite_user_id": manifest["composite_user_id"],
        "sources": manifest["sources"],
        "event_counts": {k: len(v) for k, v in events_by_split.items()},
        "sample_counts": {k: len(v) for k, v in samples_by_split.items()},
        "schema": {
            "event": {
                "event_id": "str",
                "user_id": "str",
                "timestamp": "int",
                "source_dataset": "str",
                "app": "str",
                "domain": "str",
                "item_id": "str",
                "title": "str",
                "meta": "str",
            },
            "sample": {
                "sample_id": "str",
                "user_id": "str",
                "timestamp": "int",
                "history_items": "List[Dict]",
                "candidate_items": "List[Dict]",
                "target_option_idx": "int",
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to composite manifest json.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--history_size", type=int, default=10)
    parser.add_argument("--min_history", type=int, default=3)
    parser.add_argument("--neg_sample_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q_original", type=float, default=0.5)
    parser.add_argument("--q_finetune_end", type=float, default=0.8)
    parser.add_argument("--negative_pool_mode", type=str, default="same_source", choices=["same_source", "same_domain", "global"])
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    composite_user_id = manifest["composite_user_id"]
    merged_events = merge_sources(manifest["sources"], composite_user_id=composite_user_id)
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

    os.makedirs(args.output_dir, exist_ok=True)
    save_jsonl(os.path.join(args.output_dir, "events_all.jsonl"), merged_events)
    for split_name, rows in events_by_split.items():
        save_jsonl(os.path.join(args.output_dir, f"{split_name}_events.jsonl"), rows)
    for split_name, rows in samples_by_split.items():
        save_jsonl(os.path.join(args.output_dir, f"{split_name}.jsonl"), rows)

    meta = build_meta(manifest, events_by_split, samples_by_split)
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta["event_counts"], ensure_ascii=False))
    print(json.dumps(meta["sample_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
