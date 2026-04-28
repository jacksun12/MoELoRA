import json
import os
import random
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset


def _letter_options():
    return ["A", "B", "C", "D"]


class MultipleChoiceSequentialDataset(Dataset):
    """
    Generic multiple-choice next-item recommendation dataset for LLMs.

    Each sample contains:
    - history_items: ordered prior interactions
    - candidate_items: 4 candidates with exactly one positive
    - target_option_idx: index of the positive item in candidate_items
    """

    instruction_text = (
        "Given the device user's recent cross-app activity, choose the most likely next item "
        "from the candidate list. Return only one letter: A, B, C, or D."
    )

    def __init__(self, tokenizer=None, max_seq_len=512, split="train"):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def _format_item_line(self, item: Dict, prefix: str = "- ") -> str:
        title = item.get("title", "Unknown Item")
        domain = item.get("domain", "unknown")
        source = item.get("source_dataset", item.get("app", "unknown"))
        # Keep the cross-app signal but avoid overly verbose prompts for small models.
        return f"{prefix}{title} | Domain: {domain} | Source: {source}"

    def _format_prompt(self, sample: Dict) -> str:
        letters = _letter_options()
        history_lines = [self._format_item_line(x) for x in sample["history_items"]]
        candidate_lines = [
            self._format_item_line(x, prefix=f"{letters[i]}. ")
            for i, x in enumerate(sample["candidate_items"])
        ]
        prompt = (
            "### Instruction:\n"
            f"{self.instruction_text}\n\n"
            "### Input:\n"
            "Recent activity history:\n"
            f"{os.linesep.join(history_lines)}\n\n"
            "Candidate next items:\n"
            f"{os.linesep.join(candidate_lines)}\n\n"
            "### Response:\n"
        )
        return prompt

    def __getitem__(self, idx):
        sample = self.samples[idx]
        letters = _letter_options()
        target_option_idx = int(sample["target_option_idx"])
        target_letter = letters[target_option_idx]

        prompt = self._format_prompt(sample)
        target_ids = self.tokenizer.encode(target_letter, add_special_tokens=False)
        if len(target_ids) == 0:
            target_ids = self.tokenizer.encode("A", add_special_tokens=False)

        max_prompt_len = max(1, self.max_seq_len - len(target_ids))
        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_len,
        )

        full_ids = prompt_ids + target_ids
        full_len = len(full_ids)
        pad_len = max(0, self.max_seq_len - full_len)
        if pad_len > 0:
            full_ids = full_ids + [self.tokenizer.pad_token_id] * pad_len
        else:
            full_ids = full_ids[: self.max_seq_len]
            full_len = self.max_seq_len

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        attention_mask[:full_len] = 1
        labels = torch.full((self.max_seq_len,), -100, dtype=torch.long)
        target_start = min(len(prompt_ids), self.max_seq_len - 1)
        labels[target_start:full_len] = input_ids[target_start:full_len]

        eval_ids = prompt_ids[: self.max_seq_len]
        eval_len = len(eval_ids)
        eval_pad_len = max(0, self.max_seq_len - eval_len)
        eval_ids = eval_ids + [self.tokenizer.pad_token_id] * eval_pad_len
        eval_input_ids = torch.tensor(eval_ids, dtype=torch.long)
        eval_attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        eval_attention_mask[:eval_len] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "eval_input_ids": eval_input_ids,
            "eval_attention_mask": eval_attention_mask,
            "target_option_idx": torch.tensor(target_option_idx, dtype=torch.long),
            "user_id": sample["user_id"],
            "raw_text": f"{prompt}{target_letter}",
        }


class MovieLens1MSequential(MultipleChoiceSequentialDataset):
    """
    Multiple-choice next-item recommendation dataset for MovieLens-1M.
    """

    instruction_text = (
        "Given the user's movie preference history, choose the most likely next movie "
        "from the candidate list. Return only one letter: A, B, C, or D."
    )

    def __init__(
        self,
        data_dir="data/ml_1m",
        tokenizer=None,
        max_seq_len=512,
        history_size=10,
        split="train",
        neg_sample_size=3,
        seed=42,
        min_history=3,
        min_user_samples=3,
    ):
        super().__init__(tokenizer=tokenizer, max_seq_len=max_seq_len, split=split)
        self.data_dir = data_dir
        self.history_size = history_size
        self.neg_sample_size = neg_sample_size
        self.seed = seed
        self.min_history = min_history
        self.min_user_samples = min_user_samples

        self.ratings_file = os.path.join(data_dir, "ratings.dat")
        self.movies_file = os.path.join(data_dir, "movies.dat")
        if not os.path.exists(self.ratings_file):
            raise FileNotFoundError(
                f"Please download ML-1M to {data_dir}. "
                f"'wget https://files.grouplens.org/datasets/movielens/ml-1m.zip'"
            )

        self.samples = self._prepare_data()

    def _prepare_data(self):
        print("Loading and preprocessing MovieLens-1M multiple-choice data...")
        movies_df = pd.read_csv(
            self.movies_file,
            sep="::",
            header=None,
            names=["MovieID", "Title", "Genres"],
            engine="python",
            encoding="latin-1",
        )
        movie_map = {
            int(r["MovieID"]): {
                "title": r["Title"],
                "genres": r["Genres"],
            }
            for _, r in movies_df.iterrows()
        }
        all_movie_ids = list(movie_map.keys())

        ratings_df = pd.read_csv(
            self.ratings_file,
            sep="::",
            header=None,
            names=["UserID", "MovieID", "Rating", "Timestamp"],
            engine="python",
        )
        ratings_df = ratings_df[ratings_df["Rating"] >= 4]
        ratings_df = ratings_df.sort_values(by=["UserID", "Timestamp"])

        all_user_samples = {}
        rng = random.Random(self.seed)

        grouped = ratings_df.groupby("UserID")
        for user_id, group in grouped:
            movie_ids = [int(x) for x in group["MovieID"].tolist()]
            timestamps = [int(x) for x in group["Timestamp"].tolist()]
            dedup_ids, dedup_ts, seen = [], [], set()
            for mid, ts in zip(movie_ids, timestamps):
                if mid in seen:
                    continue
                seen.add(mid)
                dedup_ids.append(mid)
                dedup_ts.append(ts)
            movie_ids, timestamps = dedup_ids, dedup_ts

            if len(movie_ids) < max(self.history_size + 1, self.min_history + 1):
                continue

            user_samples = []
            for i in range(self.min_history, len(movie_ids)):
                history_ids = movie_ids[max(0, i - self.history_size) : i]
                target_id = movie_ids[i]
                target_ts = timestamps[i]

                forbidden = set(history_ids)
                forbidden.add(target_id)
                negative_pool = [m for m in all_movie_ids if m not in forbidden]
                if len(negative_pool) < self.neg_sample_size:
                    continue

                negatives = rng.sample(negative_pool, self.neg_sample_size)
                candidates = negatives + [target_id]
                rng.shuffle(candidates)
                target_option_idx = candidates.index(target_id)

                history_items = [
                    {
                        "item_id": f"ml_{mid}",
                        "title": movie_map.get(mid, {}).get("title", "Unknown Movie"),
                        "domain": "movie",
                        "source_dataset": "movielens_1m",
                        "meta": movie_map.get(mid, {}).get("genres", "Unknown"),
                    }
                    for mid in history_ids
                ]
                candidate_items = [
                    {
                        "item_id": f"ml_{mid}",
                        "title": movie_map.get(mid, {}).get("title", "Unknown Movie"),
                        "domain": "movie",
                        "source_dataset": "movielens_1m",
                        "meta": movie_map.get(mid, {}).get("genres", "Unknown"),
                    }
                    for mid in candidates
                ]

                user_samples.append(
                    {
                        "user_id": int(user_id),
                        "timestamp": target_ts,
                        "history_items": history_items,
                        "candidate_items": candidate_items,
                        "target_option_idx": target_option_idx,
                    }
                )
            if len(user_samples) >= self.min_user_samples:
                all_user_samples[int(user_id)] = user_samples

        train_samples, val_samples, test_samples = [], [], []
        for _uid, rows in all_user_samples.items():
            train_samples.extend(rows[:-2])
            val_samples.append(rows[-2])
            test_samples.append(rows[-1])

        if self.split == "train":
            split_samples = train_samples
        elif self.split == "val":
            split_samples = val_samples
        elif self.split == "test":
            split_samples = test_samples
        else:
            raise ValueError(f"Unsupported split={self.split}, expected train/val/test")

        print(f"Created {len(split_samples)} samples for split={self.split}.")
        return split_samples


class CompositeSequentialDataset(MultipleChoiceSequentialDataset):
    """
    Dataset backed by pre-built JSONL composite-device samples.

    Expected fields per row:
    - user_id
    - timestamp
    - history_items
    - candidate_items
    - target_option_idx
    """

    def __init__(
        self,
        data_dir="data/composite_device",
        tokenizer=None,
        max_seq_len=512,
        split="train",
    ):
        super().__init__(tokenizer=tokenizer, max_seq_len=max_seq_len, split=split)
        self.data_dir = data_dir
        self.samples = self._load_split_samples()

    def _load_jsonl(self, path: str) -> List[Dict]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _load_split_samples(self) -> List[Dict]:
        explicit_path = os.path.join(self.data_dir, f"{self.split}.jsonl")
        if os.path.exists(explicit_path):
            rows = self._load_jsonl(explicit_path)
            print(f"Loaded {len(rows)} composite samples from {explicit_path}")
            return rows

        alias_map = {
            "train": ["original.jsonl", "finetune.jsonl"],
            "val": ["finetune.jsonl"],
            "test": ["test.jsonl"],
        }
        if self.split not in alias_map:
            raise ValueError(f"Unsupported split={self.split}, expected train/val/test")

        rows = []
        for filename in alias_map[self.split]:
            path = os.path.join(self.data_dir, filename)
            if not os.path.exists(path):
                continue
            rows.extend(self._load_jsonl(path))
        rows.sort(key=lambda x: (x.get("timestamp", 0), x.get("sample_id", "")))
        print(f"Loaded {len(rows)} composite samples for split={self.split} from aliases {alias_map[self.split]}")
        return rows


def build_dataset(cfg, tokenizer, split):
    dataset_type = cfg.get("data", {}).get("dataset_type", "movielens_1m").lower()
    max_seq_len = cfg["data"]["max_seq_len"]

    if dataset_type in {"movielens_1m", "movielens", "ml_1m"}:
        return MovieLens1MSequential(
            data_dir=cfg["data"].get("data_dir", "data/ml_1m"),
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            history_size=cfg["data"].get("history_size", 10),
            min_history=cfg["data"].get("min_history", 3),
            min_user_samples=cfg["data"].get("min_user_samples", 3),
            split=split,
            neg_sample_size=cfg["data"].get("neg_sample_size", 3),
            seed=cfg["data"].get("seed", 42),
        )

    if dataset_type in {"composite_device", "composite", "cross_app"}:
        return CompositeSequentialDataset(
            data_dir=cfg["data"].get("data_dir", "data/composite_device"),
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            split=split,
        )

    raise ValueError(f"Unsupported data.dataset_type={dataset_type}")


@torch.no_grad()
def extract_clustering_features(model, batch_inputs, strategy="sentence_mean"):
    outputs = model(
        input_ids=batch_inputs["input_ids"],
        attention_mask=batch_inputs["attention_mask"],
        output_hidden_states=True,
    )
    last_hidden_state = outputs.hidden_states[-1]
    attention_mask = batch_inputs["attention_mask"].unsqueeze(-1).to(last_hidden_state.dtype)

    if strategy == "last_token":
        sequence_lengths = batch_inputs["attention_mask"].sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        return last_hidden_state[torch.arange(batch_size), sequence_lengths]

    pooled = (last_hidden_state * attention_mask).sum(dim=1)
    denom = attention_mask.sum(dim=1).clamp_min(1e-6)
    return pooled / denom
