from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from pawn.index import ShardMeta, build_index

FilterSpec = Callable[[ShardMeta], bool] | dict[str, Any] | None


def _shard_field(m: ShardMeta, key: str) -> Any:
    if hasattr(m, key) and key != "meta":
        return getattr(m, key)
    return m.meta.get(key)


def _matcher(allowed: Any) -> Callable[[Any], bool]:
    if callable(allowed):
        return lambda v: bool(allowed(v))
    if isinstance(allowed, re.Pattern):
        return lambda v: v is not None and bool(allowed.search(str(v)))
    if isinstance(allowed, str) and allowed.startswith("regex:"):
        pat = re.compile(allowed[len("regex:") :])
        return lambda v: v is not None and bool(pat.search(str(v)))
    if isinstance(allowed, (list, tuple, set)):
        s = set(allowed)
        return lambda v: v in s
    return lambda v: v == allowed


def apply_filter(metas: list[ShardMeta], spec: FilterSpec) -> list[ShardMeta]:
    if spec is None:
        return metas
    if callable(spec):
        return [m for m in metas if spec(m)]

    matchers = {k: _matcher(v) for k, v in spec.items()}
    return [
        m
        for m in metas
        if all(match(_shard_field(m, k)) for k, match in matchers.items())
    ]


class CachedFeatureDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        max_len: int = 512,
        index_workers: int = 16,
        filter: FilterSpec = None,
    ):
        all_metas = build_index(Path(root), num_workers=index_workers)
        if not all_metas:
            raise FileNotFoundError(f"no .pt shards under {root}")
        self.metas: list[ShardMeta] = apply_filter(all_metas, filter)
        if not self.metas:
            raise FileNotFoundError(
                f"filter {filter!r} matched 0/{len(all_metas)} shards under {root}"
            )
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.metas)

    @property
    def labels(self) -> list[int]:
        return [m.y for m in self.metas]

    @property
    def sources(self) -> list[str]:
        return [m.src for m in self.metas]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        m = self.metas[idx]
        d = torch.load(m.path, weights_only=True)
        T = min(int(d["length"]), self.max_len)
        return {
            "hs_curr": d["hs_curr"][:T].float(),
            "hs_next": d["hs_next"][:T].float(),
            "metrics": d["metrics"][:T].float(),
            "labels": torch.tensor(d["y"], dtype=torch.long),
        }


@dataclass
class PAWNCollator:
    pad_to_multiple_of: int | None = 8

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        T = max(b["hs_curr"].size(0) for b in batch)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            T = ((T + m - 1) // m) * m

        d = batch[0]["hs_curr"].size(-1)
        M = batch[0]["metrics"].size(-1)
        B = len(batch)

        hs_curr = torch.zeros(B, T, d)
        hs_next = torch.zeros(B, T, d)
        metrics = torch.zeros(B, T, M)
        attention_mask = torch.zeros(B, T, dtype=torch.long)
        labels = torch.zeros(B, dtype=torch.long)

        for i, b in enumerate(batch):
            t = b["hs_curr"].size(0)
            hs_curr[i, :t] = b["hs_curr"]
            hs_next[i, :t] = b["hs_next"]
            metrics[i, :t] = b["metrics"]
            attention_mask[i, :t] = 1
            labels[i] = b["labels"]

        return {
            "hs_curr": hs_curr,
            "hs_next": hs_next,
            "metrics": metrics,
            "attention_mask": attention_mask,
            "labels": labels,
        }
