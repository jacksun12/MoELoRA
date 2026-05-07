import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm


def save_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def open_with_progress(path: str, desc: str):
    total_bytes = os.path.getsize(path)
    f = open(path, "r", encoding="utf-8")
    bar = tqdm(
        total=total_bytes,
        desc=desc,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    )

    def iterator():
        try:
            for line in f:
                bar.update(len(line.encode("utf-8")))
                yield line
        finally:
            bar.close()
            f.close()

    return iterator()


def load_business_map(path: str) -> Dict[str, Dict[str, str]]:
    business_map = {}
    for line in open_with_progress(path, desc="Load business map"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        bid = row.get("business_id")
        if not bid:
            continue
        business_map[str(bid)] = {
            "item_id": f"yelp_{bid}",
            "title": row.get("name", f"business:{bid}"),
            "domain": "poi",
            "source_dataset": "yelp",
            "app": "local",
            "meta": row.get("categories") or "",
        }
    return business_map


def count_user_interactions(review_path: str, min_stars: float) -> Counter:
    counts = Counter()
    for line in open_with_progress(review_path, desc="Count user interactions"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if float(row.get("stars", 0.0)) < min_stars:
            continue
        uid = row.get("user_id")
        if uid:
            counts[str(uid)] += 1
    return counts


def select_users(
    review_path: str,
    min_stars: float,
    min_interactions: int,
    max_users: int,
    seed: int,
) -> List[str]:
    counts = count_user_interactions(review_path, min_stars)
    eligible = [uid for uid, cnt in counts.items() if cnt >= min_interactions]
    eligible.sort()
    rng = random.Random(seed)
    rng.shuffle(eligible)
    if max_users > 0:
        eligible = eligible[:max_users]
    return eligible


def load_user_sequences(
    review_path: str,
    business_map: Dict[str, Dict[str, str]],
    selected_users: List[str],
    min_stars: float,
    deduplicate: bool = True,
) -> Dict[str, List[Dict]]:
    selected = set(selected_users)
    user_events = defaultdict(list)
    for line in open_with_progress(review_path, desc="Load user sequences"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        uid = str(row.get("user_id"))
        if uid not in selected:
            continue
        if float(row.get("stars", 0.0)) < min_stars:
            continue
        bid = str(row.get("business_id"))
        business = business_map.get(bid)
        if business is None:
            continue
        ts = int(str(row.get("date", "1970-01-01 00:00:00")).replace("-", "").replace(":", "").replace(" ", ""))
        event = dict(business)
        event["timestamp"] = ts
        user_events[uid].append(event)

    final = {}
    for uid, rows in user_events.items():
        rows.sort(key=lambda x: (x["timestamp"], x["item_id"]))
        if deduplicate:
            seen = set()
            dedup = []
            for row in rows:
                if row["item_id"] in seen:
                    continue
                seen.add(row["item_id"])
                dedup.append(row)
            rows = dedup
        final[uid] = rows
    return final


def split_user_rows(rows: List[Dict], min_history: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if len(rows) < min_history + 3:
        return [], [], []
    return rows[:-2], [rows[-2]], [rows[-1]]


def build_samples(
    rows: List[Dict],
    user_id: str,
    item_catalog: Dict[str, Dict],
    split_name: str,
    history_size: int,
    min_history: int,
    neg_sample_size: int,
    rng: random.Random,
) -> List[Dict]:
    if len(rows) < min_history + 1:
        return []
    all_item_ids = list(item_catalog.keys())
    samples = []
    for i in range(min_history, len(rows)):
        history_rows = rows[max(0, i - history_size):i]
        target = rows[i]
        forbidden = {x["item_id"] for x in history_rows}
        forbidden.add(target["item_id"])
        negative_pool = [item_id for item_id in all_item_ids if item_id not in forbidden]
        if len(negative_pool) < neg_sample_size:
            continue
        negatives = rng.sample(negative_pool, neg_sample_size)
        candidate_ids = negatives + [target["item_id"]]
        rng.shuffle(candidate_ids)
        samples.append(
            {
                "sample_id": f"{split_name}_{user_id}_{len(samples)+1:06d}",
                "user_id": user_id,
                "timestamp": int(target["timestamp"]),
                "split": split_name,
                "target_domain": target["domain"],
                "target_source_dataset": target["source_dataset"],
                "history_items": [dict(x) for x in history_rows],
                "candidate_items": [item_catalog[item_id] for item_id in candidate_ids],
                "target_option_idx": int(candidate_ids.index(target["item_id"])),
            }
        )
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review_path", type=str, required=True)
    parser.add_argument("--business_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--min_stars", type=float, default=4.0)
    parser.add_argument("--min_user_interactions", type=int, default=30)
    parser.add_argument("--max_users", type=int, default=2000)
    parser.add_argument("--history_size", type=int, default=10)
    parser.add_argument("--min_history", type=int, default=3)
    parser.add_argument("--neg_sample_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print("[Stage-0 Builder] Loading Yelp business metadata...")
    business_map = load_business_map(args.business_path)
    print("[Stage-0 Builder] Counting eligible Yelp users...")
    selected_users = select_users(
        review_path=args.review_path,
        min_stars=args.min_stars,
        min_interactions=args.min_user_interactions,
        max_users=args.max_users,
        seed=args.seed,
    )
    print(f"[Stage-0 Builder] Selected users: {len(selected_users)}")
    print("[Stage-0 Builder] Loading user interaction sequences...")
    user_sequences = load_user_sequences(
        review_path=args.review_path,
        business_map=business_map,
        selected_users=selected_users,
        min_stars=args.min_stars,
    )

    item_catalog = {item_id: dict(meta) for item_id, meta in business_map.items()}
    for bid, meta in list(item_catalog.items()):
        item_catalog[meta["item_id"]] = meta
        del item_catalog[bid]

    train_samples, val_samples, test_samples = [], [], []
    kept_users = 0
    for uid in tqdm(selected_users, desc="Build per-user samples", dynamic_ncols=True):
        rows = user_sequences.get(uid, [])
        train_rows, val_rows, test_rows = split_user_rows(rows, args.min_history)
        if not train_rows or not val_rows or not test_rows:
            continue
        kept_users += 1
        train_samples.extend(
            build_samples(train_rows, uid, item_catalog, "train", args.history_size, args.min_history, args.neg_sample_size, rng)
        )
        val_samples.extend(
            build_samples(train_rows + val_rows, uid, item_catalog, "val", args.history_size, args.min_history, args.neg_sample_size, rng)
        )
        test_samples.extend(
            build_samples(train_rows + val_rows + test_rows, uid, item_catalog, "test", args.history_size, args.min_history, args.neg_sample_size, rng)
        )

    os.makedirs(args.output_dir, exist_ok=True)
    save_jsonl(os.path.join(args.output_dir, "train.jsonl"), train_samples)
    save_jsonl(os.path.join(args.output_dir, "val.jsonl"), val_samples)
    save_jsonl(os.path.join(args.output_dir, "test.jsonl"), test_samples)
    meta = {
        "dataset_type": "composite_device",
        "source_dataset": "yelp",
        "selected_users": kept_users,
        "sample_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
