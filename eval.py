"""Evaluate a trained PAWN checkpoint on cached feature shards."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pawn.data import CachedFeatureDataset, FilterSpec, PAWNCollator
from pawn.metrics import compute_metrics
from train import build_model

log = logging.getLogger(__name__)


def _to_plain(node) -> Any:
    if node is None:
        return None
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)
    return node


def _to_filter(node) -> FilterSpec:
    value = _to_plain(node)
    if value is None:
        return value
    if callable(value):
        return lambda meta: bool(value(meta))
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    raise TypeError(f"expected filter mapping, got {type(value).__name__}")


def _normalize_testbeds(spec) -> dict[str, dict[str, Any]]:
    value = _to_plain(spec) or ["test"]
    if isinstance(value, list):
        return {str(name): {"dir": str(name)} for name in value}
    if not isinstance(value, dict):
        raise TypeError("eval.testbeds must be a list or mapping")

    out: dict[str, dict] = {}
    for name, entry in value.items():
        key = str(name)
        if isinstance(entry, str):
            out[key] = {"dir": entry}
        elif isinstance(entry, dict):
            out[key] = cast(dict[str, Any], entry)
        else:
            raise TypeError(f"invalid testbed spec for {key!r}: {type(entry).__name__}")
        out[key].setdefault("dir", key)
    return out


def load_state_dict(ckpt: Path) -> dict[str, torch.Tensor]:
    if ckpt.is_dir():
        st = ckpt / "model.safetensors"
        if st.exists():
            from safetensors.torch import load_file

            return load_file(str(st))
        bin_path = ckpt / "pytorch_model.bin"
        if bin_path.exists():
            return torch.load(bin_path, map_location="cpu", weights_only=True)
        raise FileNotFoundError(f"no model.safetensors or pytorch_model.bin under {ckpt}")
    if ckpt.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(ckpt))
    return torch.load(ckpt, map_location="cpu", weights_only=True)


@torch.no_grad()
def predict(
    model,
    dataset: CachedFeatureDataset,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=PAWNCollator(),
        pin_memory=(device.type == "cuda"),
    )
    logits_chunks: list[torch.Tensor] = []
    labels_chunks: list[torch.Tensor] = []
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    for batch in tqdm(loader, desc="predict", mininterval=1.0):
        labels = batch.pop("labels")
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast_ctx:
            out = model(**batch)
        logits_chunks.append(out.logits.detach().float().cpu())
        labels_chunks.append(labels.detach().cpu())

    return torch.cat(logits_chunks).numpy(), torch.cat(labels_chunks).numpy()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    eval_cfg = cfg.get("eval") or OmegaConf.create({})
    checkpoint = eval_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("missing +eval.checkpoint=<path>")
    checkpoint = Path(str(checkpoint))

    testbeds = _normalize_testbeds(eval_cfg.get("testbeds"))
    cache_root = Path(str(eval_cfg.get("cache_root") or cfg.data.cache_dir))
    batch_size = int(eval_cfg.get("batch_size") or cfg.training.per_device_eval_batch_size)
    num_workers = int(
        eval_cfg.get("num_workers")
        if eval_cfg.get("num_workers") is not None
        else cfg.training.dataloader_num_workers
    )
    device = torch.device(str(eval_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")))
    use_amp = bool(eval_cfg.get("bf16", cfg.training.bf16))
    output_dir = Path(str(eval_cfg.get("output_dir") or cfg.training.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    missing, unexpected = model.load_state_dict(load_state_dict(checkpoint), strict=False)
    if missing:
        log.warning(f"{len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        log.warning(f"{len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    model.eval()

    results: dict[str, dict] = {}
    for tb_name, spec in testbeds.items():
        sub = spec["dir"]
        cache_dir = Path(sub) if os.path.isabs(sub) else cache_root / sub
        if not cache_dir.exists():
            log.warning(f"skipping {tb_name!r}: {cache_dir} does not exist")
            continue
        ds = CachedFeatureDataset(
            cache_dir,
            max_len=int(cfg.data.max_len),
            filter=_to_filter(spec.get("filter")),
        )
        logits, labels = predict(
            model,
            ds,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            use_amp=use_amp,
        )
        metrics = compute_metrics(SimpleNamespace(predictions=logits, label_ids=labels))
        results[tb_name] = {
            "n": int(len(labels)),
            "dir": str(cache_dir),
            "filter": spec.get("filter"),
            "metrics": metrics,
        }
        np.savez(output_dir / f"logits_{tb_name}.npz", logits=logits, labels=labels)

    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint),
                "cache_root": str(cache_root),
                "backbone": cfg.data.backbone,
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )


if __name__ == "__main__":
    main()
