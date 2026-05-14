import json
import os
from collections import Counter
from typing import Dict, List, Tuple


def save_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_mix(spec: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, weight = chunk.split(":")
        weights[key.strip()] = float(weight)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("mix must have positive total weight")
    return {k: v / total for k, v in weights.items()}


def rebuild_samples(rows: List[Dict], split_name: str, user_id: str, ts_start: int) -> List[Dict]:
    out: List[Dict] = []
    for i, row in enumerate(rows, start=1):
        sample = dict(row)
        sample["sample_id"] = f"{split_name}_{i:07d}"
        sample["user_id"] = user_id
        sample["timestamp"] = ts_start + (i - 1)
        out.append(sample)
    return out


def split_rows(rows: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1):
    n = len(rows)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def take_weighted_rows(
    mix: Dict[str, float],
    grouped_rows: Dict[str, List[Dict]],
    cursor: Dict[str, int],
    total_samples: int,
    label_name: str = "label",
) -> Tuple[List[Dict], Counter]:
    phase_rows: List[Dict] = []
    phase_counts: Counter = Counter()
    remaining = int(total_samples)
    items = list(mix.items())
    for idx, (label, weight) in enumerate(items):
        if label not in grouped_rows:
            raise ValueError(f"Unknown {label_name} in mix: {label}")
        want = int(round(total_samples * weight))
        if idx == len(items) - 1:
            want = remaining
        available = len(grouped_rows[label]) - cursor[label]
        take = min(want, available)
        if take < want:
            raise ValueError(
                f"Not enough rows for {label_name}={label}: need {want}, available {available}."
            )
        chosen = grouped_rows[label][cursor[label] : cursor[label] + take]
        cursor[label] += take
        remaining -= take
        phase_rows.extend(chosen)
        phase_counts[label] += take
    return phase_rows, phase_counts
