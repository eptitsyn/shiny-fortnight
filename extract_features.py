"""
Multi-GPU + multi-threaded feature extraction.

Launch on 2 GPUs:
  torchrun --standalone --nproc_per_node=2 extract_features.py

Single GPU fallback (also works):
  python extract_features.py

CLI overrides:
  torchrun --standalone --nproc_per_node=2 extract_features.py \
      extract.splits='[validation,test]' extract.num_io_workers=8
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


# ---------- DDP helpers ----------

def ddp_info() -> tuple[int, int, int]:
    """Returns (rank, world_size, local_rank). Works under torchrun or solo."""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local_rank


def shard_indices(n: int, rank: int, world: int) -> list[int]:
    """Strided sharding: rank r gets indices r, r+world, r+2*world, ..."""
    return list(range(rank, n, world))


# ---------- Metric computation (unchanged) ----------

@torch.no_grad()
def compute_metrics_per_token(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    V = logits.size(-1)

    target = target_ids.unsqueeze(-1)
    lp_true = log_probs.gather(-1, target).squeeze(-1)
    entropy = -(probs * log_probs).sum(-1)
    max_lp = log_probs.max(dim=-1).values
    rank = (log_probs > lp_true.unsqueeze(-1)).sum(-1).float() / V
    target_p = probs.gather(-1, target)
    top_p_mass = (probs * (probs >= target_p)).sum(-1)

    return torch.stack([lp_true, entropy, max_lp, rank, top_p_mass], dim=-1)


def label_to_y(label: int) -> int:
    return 1 - int(label)


# ---------- Async writer ----------

def save_payload(path: Path, payload: dict[str, Any]) -> None:
    """Runs in worker thread. torch.save releases the GIL for the heavy bytes."""
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic — partial files won't pollute the cache


# ---------- Per-rank extraction ----------

def extract_split_on_rank(cfg: DictConfig, split: str) -> None:
    rank, world, local_rank = ddp_info()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    out_dir = Path(cfg.data.cache_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.data.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.data.backbone,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    ).to(device).eval()

    ds = load_dataset(cfg.data.dataset_name, split=split)
    if cfg.extract.limit:
        ds = ds.select(range(cfg.extract.limit))

    my_indices = shard_indices(len(ds), rank, world)
    log.info(
        f"[rank {rank}/{world}] split={split} "
        f"shard={len(my_indices)}/{len(ds)} device={device}"
    )

    # Thread pool for disk writes. Bounded queue keeps RAM in check when
    # the GPU runs ahead of disk.
    max_pending = cfg.extract.num_io_workers * cfg.extract.prefetch_factor
    pending: Queue = Queue(maxsize=max_pending)
    pool = ThreadPoolExecutor(max_workers=cfg.extract.num_io_workers)

    def submit(path: Path, payload: dict[str, Any]) -> None:
        future = pool.submit(save_payload, path, payload)
        pending.put(future)
        # Drain completed futures so exceptions surface promptly.
        while not pending.empty() and pending.queue[0].done():
            pending.get().result()

    kept = skipped = 0
    progress = tqdm(
        my_indices, desc=f"r{rank}[{split}]", disable=(rank != 0), mininterval=1.0
    )

    for idx in progress:
        row = ds[idx]
        out_path = out_dir / f"{idx:08d}.pt"
        if out_path.exists():
            # Resume support — skip already-cached samples.
            skipped += 1
            continue

        enc = tok(
            row["text"],
            return_tensors="pt",
            truncation=True,
            max_length=cfg.data.max_len,
            padding=False,
        ).to(device)
        ids = enc["input_ids"]
        if ids.size(1) < cfg.data.min_len:
            continue

        with torch.no_grad():
            out = model(**enc)
        hs = out.hidden_states[-1][0]
        logits = out.logits[0]

        target = ids[0, 1:]
        logits = logits[:-1]
        hs_curr = hs[:-1]
        hs_next = hs[1:]

        metrics = compute_metrics_per_token(logits, target)

        # Move to CPU + fp16 on the GPU thread (fast). The slow part is
        # serialization + fsync, which the writer thread handles.
        payload = {
            "hs_curr": hs_curr.to(torch.float16).cpu().contiguous(),
            "hs_next": hs_next.to(torch.float16).cpu().contiguous(),
            "metrics": metrics.to(torch.float16).cpu().contiguous(),
            "length": int(hs_curr.size(0)),
            "y": label_to_y(row["label"]),
            "src": row.get("src", ""),
        }
        submit(out_path, payload)
        kept += 1

    # Drain remaining writes
    while not pending.empty():
        pending.get().result()
    pool.shutdown(wait=True)

    log.info(
        f"[rank {rank}] split={split} kept={kept} skipped={skipped} "
        f"out={out_dir}"
    )


# ---------- Entrypoint ----------

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    rank, world, local_rank = ddp_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Cap PyTorch CPU thread pool so 2 ranks × N IO workers don't oversubscribe.
    # We split CPU budget evenly: 32 cores / 2 ranks = 16 each, minus IO threads.
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
