"""
Feature extraction.

torchrun --standalone --nproc_per_node=2 extract_features.py
python extract_features.py

CLI overrides:
  torchrun --standalone --nproc_per_node=2 extract_features.py \
      extract.splits='[validation,test]' extract.num_io_workers=8
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

log = logging.getLogger(__name__)

_META_OK = (str, int, float, bool, type(None))

SubmitFn = Callable[[Path, dict[str, Any]], None]


# ---------- DDP helpers ----------


def ddp_info() -> tuple[int, int, int]:
    """(rank, world_size, local_rank). Works under torchrun or solo."""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local_rank


def shard_indices(n: int, rank: int, world: int) -> list[int]:
    """Strided sharding: rank r gets indices r, r+world, r+2*world, ..."""
    return list(range(rank, n, world))


# ---------- Per-token metrics ----------


@torch.no_grad()
def compute_metrics_per_token(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()

    target = target_ids.unsqueeze(-1)
    next_token_log_probs = log_probs.gather(-1, target).squeeze(-1)
    entropy = -(probs * log_probs).sum(-1)
    max_log_probs = log_probs.max(dim=-1).values

    # Tokens with log-prob >= the target token's log-prob (target itself
    # always satisfies this, so quantile is in (0, 1]).
    greater_mask = (log_probs >= next_token_log_probs.unsqueeze(-1)).to(probs.dtype)
    quantile = greater_mask.mean(dim=-1)
    top_p = (probs * greater_mask).sum(-1)

    return torch.stack(
        [entropy, max_log_probs, next_token_log_probs, quantile, top_p], dim=-1
    )


def label_to_y(label: int) -> int:
    return 1 - int(label)


def row_meta(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "text" and isinstance(v, _META_OK)}


def load_backbone(
    backbone: str, device: torch.device
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tok = AutoTokenizer.from_pretrained(backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            backbone,
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
        )
        .to(device)
        .eval()
    )
    return tok, model


def _save_payload(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


@contextmanager
def async_saver(num_workers: int, max_pending: int) -> Iterator[SubmitFn]:
    pool = ThreadPoolExecutor(max_workers=num_workers)
    pending: Queue[Future] = Queue(maxsize=max_pending)

    def submit(path: Path, payload: dict[str, Any]) -> None:
        pending.put(pool.submit(_save_payload, path, payload))
        while not pending.empty() and pending.queue[0].done():
            pending.get().result()

    try:
        yield submit
    finally:
        while not pending.empty():
            pending.get().result()
        pool.shutdown(wait=True)


@torch.no_grad()
def extract_one(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    row: dict[str, Any],
    cfg: DictConfig,
    split: str,
    device: torch.device,
) -> dict[str, Any] | None:
    enc = tok(
        row["text"],
        return_tensors="pt",
        truncation=True,
        max_length=cfg.data.max_len,
        padding=False,
    ).to(device)
    ids = enc["input_ids"]
    if ids.size(1) < cfg.data.min_len:
        return None

    out = model(**enc)
    hs = out.hidden_states[-1][0]
    metrics = compute_metrics_per_token(out.logits[0, :-1], ids[0, 1:])
    hs_curr = hs[:-1]
    hs_next = hs[1:]

    return {
        "hs_curr": hs_curr.to(torch.float16).cpu().contiguous(),
        "hs_next": hs_next.to(torch.float16).cpu().contiguous(),
        "metrics": metrics.to(torch.float16).cpu().contiguous(),
        "length": int(hs_curr.size(0)),
        "y": label_to_y(row["label"]),
        "src": row.get("src", ""),
        "split": split,
        "meta": row_meta(row),
    }


def extract_split_on_rank(cfg: DictConfig, split: str) -> None:
    rank, world, local_rank = ddp_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.data.cache_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_backbone(cfg.data.backbone, device)

    ds = load_dataset(cfg.data.dataset_name, split=split)
    if cfg.extract.limit:
        ds = ds.select(range(cfg.extract.limit))

    my_indices = shard_indices(len(ds), rank, world)
    log.info(
        f"[rank {rank}/{world}] split={split} "
        f"shard={len(my_indices)}/{len(ds)} device={device}"
    )

    max_pending = cfg.extract.num_io_workers * cfg.extract.prefetch_factor
    kept = skipped = 0
    progress = tqdm(
        my_indices, desc=f"r{rank}[{split}]", disable=(rank != 0), mininterval=1.0
    )

    with async_saver(cfg.extract.num_io_workers, max_pending) as submit:
        for idx in progress:
            out_path = out_dir / f"{idx:08d}.pt"
            if out_path.exists():
                skipped += 1
                continue

            payload = extract_one(model, tok, ds[idx], cfg, split, device)
            if payload is None:
                continue
            submit(out_path, payload)
            kept += 1

    log.info(f"[rank {rank}] split={split} kept={kept} skipped={skipped} out={out_dir}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    rank, world, local_rank = ddp_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    cpu_per_rank = max(1, (os.cpu_count() or 32) // max(world, 1))
    torch_threads = max(1, cpu_per_rank - cfg.extract.num_io_workers)
    torch.set_num_threads(torch_threads)
    if rank == 0:
        log.info(
            f"world={world} cpu_per_rank={cpu_per_rank} "
            f"torch_threads={torch_threads} io_workers={cfg.extract.num_io_workers}"
        )

    for split in cfg.extract.splits:
        extract_split_on_rank(cfg, split)


if __name__ == "__main__":
    main()
