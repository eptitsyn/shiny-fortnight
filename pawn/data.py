from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from pawn.index import ShardMeta, build_index


class CachedFeatureDataset(Dataset):
    """Loads .pt feature shards. Uses a pre-built index for fast startup."""

    def __init__(
        self,
        root: str | Path,
        max_len: int = 512,
        index_workers: int = 16,
    ):
        self.metas: list[ShardMeta] = build_index(
            Path(root), num_workers=index_workers
        )
        if not self.metas:
            raise FileNotFoundError(f"no .pt shards under {root}")
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.metas)

    @property
    def labels(self) -> list[int]:
        """Cheap accessor used by the balanced sampler."""
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
        attn_mask = torch.zeros(B, T, dtype=torch.long)
        labels = torch.zeros(B, dtype=torch.long)

        for i, b in enumerate(batch):
            t = b["hs_curr"].size(0)
            hs_curr[i, :t] = b["hs_curr"]
            hs_next[i, :t] = b["hs_next"]
            metrics[i, :t] = b["metrics"]
            attn_mask[i, :t] = 1
            labels[i] = b["labels"]

        return {
            "hs_curr": hs_curr,
            "hs_next": hs_next,
            "metrics": metrics,
            "attention_mask": attn_mask,
            "labels": labels,
        }
