"""
Single-batch diagnostic for PAWN. Runs as a Hydra app so it shares the same
config tree as train.py (no HydraConfig resolution issues).

Usage:
  # Inspect fresh-init model
  python diagnose.py +diag.init_only=true

  # Inspect a trained checkpoint
  python diagnose.py +diag.checkpoint=runs/2026-04-28_11-30-00_gpt2/best

  # Override model knobs without retraining
  python diagnose.py +diag.init_only=true model.num_gates=64 model.token_dropout=0.0

  # Larger batch, train split
  python diagnose.py +diag.init_only=true +diag.split=train +diag.batch_size=32
"""
from __future__ import annotations

from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from pawn.data import CachedFeatureDataset, PAWNCollator
from pawn.model import PAWN, PAWNOutput


# ---------- formatting ----------

def fmt(x, name: str) -> str:
    if not torch.is_tensor(x):
        return f"{name:24s} <not a tensor> ({type(x).__name__})"

    x_f = x.float().detach()
    finite = torch.isfinite(x_f)
    n_nan = int(torch.isnan(x_f).sum().item())
    n_inf = int(torch.isinf(x_f).sum().item())

    if finite.any():
        xf = x_f[finite]
        stats = (
            f"min={xf.min().item():+.4f} "
            f"mean={xf.mean().item():+.4f} "
            f"max={xf.max().item():+.4f} "
            f"std={xf.std().item():.4f}"
        )
    else:
        stats = "<no finite values>"

    flags = ""
    if n_nan: flags += f" NaN={n_nan}"
    if n_inf: flags += f" Inf={n_inf}"

    return (f"{name:24s} shape={tuple(x.shape)} "
            f"dtype={str(x.dtype).replace('torch.', '')} {stats}{flags}")


# ---------- instrumented model ----------

class InstrumentedPAWN(PAWN):
    """Captures intermediate tensors during forward for inspection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured: dict[str, torch.Tensor] = {}

    def forward(self, hs_curr, hs_next, metrics, attention_mask, labels=None):
        c = self.captured
        c["hs_curr (input)"] = hs_curr
        c["hs_next (input)"] = hs_next
        c["metrics (input)"] = metrics
        c["attention_mask"] = attention_mask

        B, T, _ = hs_curr.shape
        device = hs_curr.device

        processed_metrics = self.metrics_nn(metrics)
        c["processed_metrics"] = processed_metrics

        pos = torch.arange(T, device=device, dtype=hs_curr.dtype) / self.max_len
        pos = pos.view(1, T, 1).expand(B, T, 1)
        gate_x = torch.cat([hs_curr, hs_next, pos], dim=-1)
        c["gate_x"] = gate_x

        gate_logits_raw = self.gate_nn(gate_x)
        c["gate_logits (raw)"] = gate_logits_raw

        invalid_mask = attention_mask == 0
        invalid_mask = self._drop_tokens(invalid_mask)
        c["invalid_token_frac"] = invalid_mask.float().mean()

        gate_logits = gate_logits_raw.masked_fill(
            invalid_mask.unsqueeze(-1), float("-inf")
        )

        G, Fdim = self.num_gates, self.feature_dim
        if G < Fdim:
            gate_logits = gate_logits.repeat(1, 1, Fdim // G)
        c["gate_logits (masked)"] = gate_logits

        gates = gate_logits.softmax(dim=-2)
        c["gates (post-softmax)"] = gates
        c["gates_max_per_seq"] = gates.amax(dim=-2)
        c["gates_entropy"] = -(gates * (gates + 1e-12).log()).sum(dim=-2)

        pooled = (gates * processed_metrics).sum(dim=-2)
        c["pooled"] = pooled

        logits = self.aggregate_nn(pooled).squeeze(-1)
        c["logits"] = logits

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            c["loss"] = loss

        return PAWNOutput(loss=loss, logits=logits)


# ---------- diagnostic ----------

def load_checkpoint(model: torch.nn.Module, ckpt_dir: Path, device: str) -> None:
    sd_path = ckpt_dir / "model.safetensors"
    if sd_path.exists():
        from safetensors.torch import load_file
        sd = load_file(str(sd_path))
    else:
        sd = torch.load(
            ckpt_dir / "pytorch_model.bin", map_location=device, weights_only=True
        )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded checkpoint: {ckpt_dir}")
    if missing:
        print(f"  missing keys: {missing}")
    if unexpected:
        print(f"  unexpected keys: {unexpected}")


@torch.no_grad()
def run_diagnostic(cfg: DictConfig) -> None:
    diag = cfg.get("diag", {})
    split = diag.get("split", "validation")
    batch_size = int(diag.get("batch_size", 8))
    device = diag.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = int(diag.get("seed", 0))
    init_only = bool(diag.get("init_only", False))
    checkpoint = diag.get("checkpoint", None)

    torch.manual_seed(seed)

    print("=" * 80)
    print("CONFIG (relevant slice)")
    print("=" * 80)
    print(OmegaConf.to_yaml({"data": cfg.data, "model": cfg.model}))

    split_dirs = {
        "train": cfg.data.train_dir,
        "validation": cfg.data.eval_dir,
        "test": cfg.data.test_dir,
    }
    ds = CachedFeatureDataset(split_dirs[split], max_len=cfg.data.max_len)
    print(f"dataset: {split}  size={len(ds)}\n")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=PAWNCollator(),
        num_workers=0,
    )
    batch = next(iter(loader))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    print("=" * 80)
    print("BATCH")
    print("=" * 80)
    for k in ("hs_curr", "hs_next", "metrics", "attention_mask", "labels"):
        print(fmt(batch[k], k))
    print(f"  valid_tokens_per_sample: {batch['attention_mask'].sum(-1).tolist()}")
    print(f"  label_distribution:      {batch['labels'].tolist()}\n")

    model = InstrumentedPAWN(
        hidden_dim=cfg.data.hidden_dim,
        num_metrics=cfg.model.num_metrics,
        feature_dim=cfg.model.feature_dim,
        num_gates=cfg.model.num_gates,
        max_len=cfg.model.max_len,
        metrics_hidden=cfg.model.metrics_hidden,
        metrics_depth=cfg.model.metrics_depth,
        gate_hidden=cfg.model.gate_hidden,
        gate_depth=cfg.model.gate_depth,
        agg_hidden=cfg.model.agg_hidden,
        agg_depth=cfg.model.agg_depth,
        dropout=cfg.model.dropout,
        token_dropout=cfg.model.token_dropout,
    ).to(device)

    if checkpoint and not init_only:
        load_checkpoint(model, Path(checkpoint), device)
        mode = "TRAINED (eval mode)"
    else:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"fresh init, trainable params: {n/1e6:.2f}M")
        mode = "FRESH INIT (eval mode)"
    model.eval()

    print("\n" + "=" * 80)
    print(f"FORWARD PASS — {mode}")
    print("=" * 80)
    model(**batch)

    order = [
        "hs_curr (input)", "hs_next (input)", "metrics (input)",
        "processed_metrics",
        "gate_x",
        "gate_logits (raw)",
        "invalid_token_frac",
        "gate_logits (masked)",
        "gates (post-softmax)",
        "gates_max_per_seq",
        "gates_entropy",
        "pooled",
        "logits",
        "loss",
    ]
    for name in order:
        if name in model.captured:
            print(fmt(model.captured[name], name))

    # ---- health checks ----
    print("\n" + "=" * 80)
    print("HEALTH CHECKS")
    print("=" * 80)

    pooled = model.captured["pooled"]
    pooled_std = pooled.float().std().item()
    print(f"pooled.std() = {pooled_std:.6f}")
    if pooled_std < 0.01:
        print("  ⚠  Very low pooled magnitude — classifier likely starved at init.")
        print("      Consider adding LayerNorm(feature_dim) before aggregate_nn.")
    elif pooled_std < 0.05:
        print("  ⚠  Low pooled magnitude — worth watching during training.")
    else:
        print("  ✓ Pooled magnitude looks healthy.")

    valid_T = batch["attention_mask"].sum(-1).float().mean().item()
    uniform_w = 1.0 / valid_T
    gates_max = model.captured["gates_max_per_seq"].float().mean().item()
    print(f"\nuniform-attention baseline = 1/T = {uniform_w:.5f}")
    print(f"observed max gate per seq  = {gates_max:.5f}  "
          f"(ratio {gates_max / uniform_w:.2f}× uniform)")
    if gates_max < 2 * uniform_w:
        print("  ⚠  Gates near uniform — model not yet learning to attend "
              "(expected at init).")
    else:
        print("  ✓ Gates are non-uniform.")

    logits = model.captured["logits"].float()
    print(f"\nlogits range: [{logits.min().item():+.3f}, {logits.max().item():+.3f}]")
    print(f"logits abs mean: {logits.abs().mean().item():.3f}")
    if logits.abs().mean().item() > 10:
        print("  ⚠  Very large logits — sigmoid will saturate.")

    m = model.captured["metrics (input)"].float()
    mask = batch["attention_mask"].bool()
    print(f"\nmetrics per-channel ranges (M={m.size(-1)}, valid tokens only):")
    for i in range(m.size(-1)):
        ch = m[..., i][mask]
        print(f"  m[{i}]: min={ch.min().item():+.3f} "
              f"mean={ch.mean().item():+.3f} "
              f"max={ch.max().item():+.3f} "
              f"std={ch.std().item():.3f}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_diagnostic(cfg)


if __name__ == "__main__":
    main()
