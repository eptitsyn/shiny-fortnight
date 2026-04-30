"""
Evaluate a trained PAWN checkpoint on one or more MAGE testbeds.

In MAGE the only metadata column is `src` (322 unique values encoding both
the source domain and the generator model), so a "testbed" is a filter on
the cached `test` shards. extract_features.py saves the full row meta into
each shard, and CachedFeatureDataset filters by it at load time.

Usage
-----
Default testbeds (just the cache subdir name -> in-domain `test` split):
  python eval.py +eval.checkpoint=runs/.../best

Custom testbeds via Hydra dict (recommended for paper-style breakdowns):
  python eval.py +eval.checkpoint=runs/.../best \
      '+eval.testbeds={
          test: {dir: test},
          machine_only: {dir: test, filter: {y: [1]}},
          human_only:   {dir: test, filter: {y: [0]}},
          gpt_models:   {dir: test, filter: {src: "regex:gpt"}}
       }'

Make sure `data=` and `model=` match what was used at training time so the
model topology lines up with the saved state dict.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pawn.data import CachedFeatureDataset, PAWNCollator
from pawn.metrics import compute_metrics
from pawn.model import PAWN

log = logging.getLogger(__name__)

DEFAULT_TESTBEDS = ["test"]


def _to_plain(node):
    if node is None:
        return None
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)
    return node


def _normalize_testbeds(spec, default_dir_for_name=True) -> dict[str, dict]:
    """Accept several input shapes and return {name: {dir, filter}}.

      ["test", "validation"]                                        -> dirs from name
      {"test": "test", "human": {"dir": "test", "filter": {...}}}   -> mixed
    """
    spec = _to_plain(spec) or DEFAULT_TESTBEDS

    if isinstance(spec, list):
        return {name: {"dir": name} for name in spec}

    out: dict[str, dict] = {}
    for name, entry in spec.items():
        if isinstance(entry, str):
            out[name] = {"dir": entry}
        elif isinstance(entry, dict):
            entry = dict(entry)
            if "dir" not in entry and default_dir_for_name:
                entry["dir"] = name
            out[name] = entry
        else:
            raise ValueError(f"unrecognized testbed entry for {name!r}: {entry!r}")
    return out


def build_model(cfg: DictConfig) -> PAWN:
    return PAWN(
        hidden_dim=cfg.data.hidden_dim,
        num_metrics=cfg.model.num_metrics,
        num_hidden_features=cfg.model.num_hidden_features,
        num_hidden_layers=cfg.model.num_hidden_layers,
        gate_nn_num_layers=cfg.model.gate_nn_num_layers,
        num_gates=cfg.model.num_gates,
        activation=cfg.model.activation,
        norm_type=cfg.model.norm_type,
        residual=cfg.model.residual,
        concat_consecutive_hidden_states=cfg.model.concat_consecutive_hidden_states,
        pos_embed_dim=cfg.model.pos_embed_dim,
        max_len=cfg.model.max_len,
        aggregation_method=cfg.model.aggregation_method,
        dropout=cfg.model.dropout,
        dropout_tokens=cfg.model.dropout_tokens,
    )


def load_state_dict(ckpt: Path) -> dict[str, torch.Tensor]:
    """Accept HF-Trainer-style checkpoint dirs (model.safetensors or
    pytorch_model.bin) as well as raw .pt / .safetensors files."""
    if ckpt.is_dir():
        st = ckpt / "model.safetensors"
        if st.exists():
            from safetensors.torch import load_file
            return load_file(str(st))
        bin_path = ckpt / "pytorch_model.bin"
        if bin_path.exists():
            return torch.load(bin_path, map_location="cpu", weights_only=True)
        raise FileNotFoundError(
            f"no model.safetensors or pytorch_model.bin under {ckpt}"
        )
    if ckpt.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(ckpt))
    return torch.load(ckpt, map_location="cpu", weights_only=True)


@torch.no_grad()
def predict(
    model: PAWN,
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

    return (
        torch.cat(logits_chunks).numpy(),
        torch.cat(labels_chunks).numpy(),
    )


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    eval_cfg = cfg.get("eval") or OmegaConf.create({})

    checkpoint = eval_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError(
            "missing +eval.checkpoint=<path>. Point it at the directory "
            "saved by train.py (e.g. runs/<stamp>_<tag>/best)."
        )
    checkpoint = Path(checkpoint)

    testbeds = _normalize_testbeds(eval_cfg.get("testbeds"))
    cache_root = Path(eval_cfg.get("cache_root") or cfg.data.cache_dir)
    batch_size = int(
        eval_cfg.get("batch_size") or cfg.training.per_device_eval_batch_size
    )
    num_workers = int(
        eval_cfg.get("num_workers")
        if eval_cfg.get("num_workers") is not None
        else cfg.training.dataloader_num_workers
    )
    device = torch.device(
        eval_cfg.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = bool(eval_cfg.get("bf16", cfg.training.bf16))
    save_logits = bool(eval_cfg.get("save_logits", True))
    output_dir = Path(eval_cfg.get("output_dir") or cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    state = load_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning(f"{len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        log.warning(f"{len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"loaded {n_params/1e6:.2f}M params from {checkpoint}")
    log.info(f"cache_root={cache_root}  testbeds={testbeds}  device={device}")

    results: dict[str, dict] = {}
    for tb_name, spec in testbeds.items():
        sub = spec["dir"]
        cache_dir = Path(sub) if os.path.isabs(sub) else cache_root / sub
        if not cache_dir.exists():
            log.warning(f"skipping {tb_name!r}: {cache_dir} does not exist")
            continue

        try:
            ds = CachedFeatureDataset(
                cache_dir, max_len=cfg.data.max_len, filter=spec.get("filter"),
            )
        except FileNotFoundError as e:
            log.warning(f"skipping {tb_name!r}: {e}")
            continue

        log.info(
            f"[{tb_name}] n={len(ds)}  dir={cache_dir}  filter={spec.get('filter')}"
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
        log.info({k: round(v, 4) for k, v in metrics.items()})
        results[tb_name] = {
            "n": int(len(labels)),
            "dir": str(cache_dir),
            "filter": spec.get("filter"),
            "metrics": metrics,
        }

        if save_logits:
            np.savez(
                output_dir / f"logits_{tb_name}.npz",
                logits=logits,
                labels=labels,
            )

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
        )
    log.info(f"wrote {output_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
