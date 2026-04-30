"""
Evaluate a trained PAWN checkpoint on the RAID benchmark.

Unlike `eval.py` (which reads a precomputed feature cache), RAID is large and
typically one-shot: this script runs the frozen backbone forward and the
trained PAWN classifier in a single online loop. No `.pt` shards are written.

Schema differences from MAGE:
  * text column is `generation` (not `text`)
  * label column is `model`; y = 0 if model == "human" else 1
  * rich metadata (`domain`, `attack`, `decoding`, ...) is reported as
    per-group breakdowns in the output JSON

Usage
-----
  # full split
  python eval_raid.py +eval.checkpoint=runs/<...>/best

  # cap test-set size (deterministic, takes the first N rows after filtering)
  python eval_raid.py +eval.checkpoint=... +eval.sample_size=2000

  # random subsample with a fixed seed (reproducible)
  python eval_raid.py +eval.checkpoint=... +eval.sample_size=2000 +eval.sample_seed=42

  # filter by any field (domain, attack, decoding, model, ...)
  python eval_raid.py +eval.checkpoint=... '+eval.raid_filter={domain: ["news", "books"]}'
  python eval_raid.py +eval.checkpoint=... '+eval.raid_filter={attack: "regex:(?i)homoglyph"}'

`data=` must match what the checkpoint was trained with so backbone hidden_dim
and the PAWN state dict line up.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from scipy.special import expit
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval import build_model, load_state_dict
from extract_features import compute_metrics_per_token
from pawn.metrics import compute_metrics

log = logging.getLogger(__name__)

RAID_TEXT_COLUMN = "generation"
RAID_LABEL_COLUMN = "model"
RAID_HUMAN_VALUE = "human"
DEFAULT_BREAKDOWNS = ["model", "domain", "attack", "decoding"]


def label_from_model(value: Any) -> int | None:
    if value is None:
        return None
    return 0 if value == RAID_HUMAN_VALUE else 1


def _to_plain(node):
    if node is None:
        return None
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)
    return node


# ---------- Filtering --------------------------------------------------------


def _make_value_pred(allowed):
    if isinstance(allowed, re.Pattern):
        return lambda v: v is not None and bool(allowed.search(str(v)))
    if isinstance(allowed, str) and allowed.startswith("regex:"):
        pat = re.compile(allowed[len("regex:") :])
        return lambda v: v is not None and bool(pat.search(str(v)))
    if isinstance(allowed, (list, tuple, set)):
        s = set(allowed)
        return lambda v: v in s
    return lambda v: v == allowed


def _make_row_pred(spec):
    """Compile a filter spec into a row predicate.

    spec forms (mirroring CachedFeatureDataset.apply_filter, plus a top-level
    `__any__` that ORs sub-specs):
      {field: <whitelist|regex:...|scalar>, ...}    -> AND across fields
      {"__any__": [<spec>, <spec>, ...]}            -> OR across sub-specs
    """
    if not spec:
        return lambda row: True
    if "__any__" in spec:
        subs = [_make_row_pred(s) for s in spec["__any__"]]
        return lambda row: any(p(row) for p in subs)
    preds = {k: _make_value_pred(v) for k, v in spec.items()}
    return lambda row: all(p(row.get(k)) for k, p in preds.items())


# ---------- Predict loop -----------------------------------------------------


@torch.no_grad()
def predict_loop(
    ds,
    *,
    tok,
    backbone,
    pawn,
    device: torch.device,
    max_len: int,
    min_len: int,
    batch_size: int,
    breakdown_keys: list[str],
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict], int]:
    """Returns (logits, labels, attrs, n_skipped)."""
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    all_logits: list[np.ndarray] = []
    all_labels: list[int] = []
    all_attrs: list[dict] = []
    n_skipped = 0

    pending_texts: list[str] = []
    pending_labels: list[int] = []
    pending_attrs: list[dict] = []

    def flush():
        nonlocal n_skipped
        if not pending_texts:
            return
        enc = tok(
            pending_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=True,
        ).to(device)
        ids = enc["input_ids"]
        am_full = enc["attention_mask"]

        # Drop samples whose tokenized length is below min_len (PAWN needs
        # >= 2 valid tokens to form a (curr, next) pair).
        lens = am_full.sum(dim=-1)
        keep = lens >= max(min_len, 2)
        n_drop = int((~keep).sum().item())
        if n_drop:
            n_skipped += n_drop
        if not keep.any():
            pending_texts.clear()
            pending_labels.clear()
            pending_attrs.clear()
            return
        if n_drop:
            keep_idx = keep.nonzero(as_tuple=True)[0]
            ids = ids.index_select(0, keep_idx)
            am_full = am_full.index_select(0, keep_idx)
            kept_labels = [pending_labels[i] for i in keep_idx.tolist()]
            kept_attrs = [pending_attrs[i] for i in keep_idx.tolist()]
        else:
            kept_labels = list(pending_labels)
            kept_attrs = list(pending_attrs)

        with autocast_ctx:
            out = backbone(input_ids=ids, attention_mask=am_full)
        hs = out.hidden_states[-1]      # [B, T, H]
        bb_logits = out.logits          # [B, T, V]

        # Shift for next-token prediction. After shift, position t in the
        # PAWN view corresponds to (input pos t, target = input pos t+1),
        # so a position is valid iff the *next* input token is non-pad.
        target = ids[:, 1:]
        bb_logits = bb_logits[:, :-1]
        hs_curr = hs[:, :-1]
        hs_next = hs[:, 1:]
        attn_mask = am_full[:, 1:]

        metrics = compute_metrics_per_token(bb_logits, target)
        pawn_out = pawn(
            hs_curr=hs_curr.float(),
            hs_next=hs_next.float(),
            metrics=metrics.float(),
            attention_mask=attn_mask,
        )
        all_logits.append(pawn_out.logits.detach().float().cpu().numpy())
        all_labels.extend(kept_labels)
        all_attrs.extend(kept_attrs)

        pending_texts.clear()
        pending_labels.clear()
        pending_attrs.clear()

    for row in tqdm(ds, total=len(ds), desc="predict_raid", mininterval=1.0):
        text = row.get(RAID_TEXT_COLUMN)
        if not text:
            n_skipped += 1
            continue
        y = label_from_model(row.get(RAID_LABEL_COLUMN))
        if y is None:
            # Held-out (test split) labels: count as skipped for metrics.
            n_skipped += 1
            continue
        pending_texts.append(text)
        pending_labels.append(y)
        pending_attrs.append({k: row.get(k) for k in breakdown_keys})
        if len(pending_texts) >= batch_size:
            flush()
    flush()

    logits = np.concatenate(all_logits) if all_logits else np.zeros(0, dtype=np.float32)
    return logits, np.array(all_labels, dtype=np.int64), all_attrs, n_skipped


# ---------- Reporting -------------------------------------------------------


def tpr_at_fpr(probs: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, probs)
    keep = fpr <= target_fpr
    return float(tpr[keep].max()) if keep.any() else 0.0


def _group_metrics(p: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    n_machine = int((y == 1).sum())
    n_human = int((y == 0).sum())
    pred = (p >= 0.5).astype(np.int64)
    machine_rec = float((pred[y == 1] == 1).mean()) if n_machine else float("nan")
    human_rec = float((pred[y == 0] == 0).mean()) if n_human else float("nan")
    out: dict[str, Any] = {
        "n": int(len(y)),
        "n_machine": n_machine,
        "n_human": n_human,
        "mean_prob": float(p.mean()),
        "machine_rec_at_0.5": machine_rec,
        "human_rec_at_0.5": human_rec,
    }
    if n_machine and n_human:
        out["auroc"] = float(roc_auc_score(y, p))
        out["tpr_at_fpr_0.05"] = tpr_at_fpr(p, y, 0.05)
    return out


def breakdown(
    probs: np.ndarray, labels: np.ndarray, attrs: list[dict], key: str
) -> dict[str, dict]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(attrs):
        groups[str(a.get(key, "<missing>"))].append(i)
    return {
        k: _group_metrics(probs[np.array(ix)], labels[np.array(ix)])
        for k, ix in sorted(groups.items())
    }


# ---------- Entrypoint ------------------------------------------------------


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

    raid_dataset = str(eval_cfg.get("raid_dataset", "liamdugan/raid"))
    raid_config = eval_cfg.get("raid_config")
    raid_split = str(eval_cfg.get("raid_split", "train"))
    raid_filter = _to_plain(eval_cfg.get("raid_filter"))
    sample_size = eval_cfg.get("sample_size") or eval_cfg.get("limit")
    sample_seed = eval_cfg.get("sample_seed")
    breakdown_keys = list(_to_plain(eval_cfg.get("breakdowns")) or DEFAULT_BREAKDOWNS)

    batch_size = int(
        eval_cfg.get("batch_size") or cfg.training.per_device_eval_batch_size
    )
    device = torch.device(
        eval_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = bool(eval_cfg.get("bf16", cfg.training.bf16))
    save_logits = bool(eval_cfg.get("save_logits", True))
    output_dir = Path(eval_cfg.get("output_dir") or cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"loading PAWN from {checkpoint}")
    pawn = build_model(cfg).to(device)
    state = load_state_dict(checkpoint)
    missing, unexpected = pawn.load_state_dict(state, strict=False)
    if missing:
        log.warning(f"{len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        log.warning(f"{len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    pawn.eval()
    log.info(f"PAWN params: {pawn.num_trainable_params() / 1e6:.2f}M")

    log.info(f"loading frozen backbone {cfg.data.backbone}")
    tok = AutoTokenizer.from_pretrained(cfg.data.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    backbone = (
        AutoModelForCausalLM.from_pretrained(
            cfg.data.backbone,
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
        )
        .to(device)
        .eval()
    )

    log.info(
        f"loading RAID  name={raid_dataset}  config={raid_config}  split={raid_split}"
    )
    ds = (
        load_dataset(raid_dataset, raid_config, split=raid_split)
        if raid_config
        else load_dataset(raid_dataset, split=raid_split)
    )

    if raid_filter:
        n_before = len(ds)
        pred = _make_row_pred(raid_filter)
        ds = ds.filter(pred)
        log.info(f"filter {raid_filter} -> kept {len(ds)}/{n_before}")
        if len(ds) == 0:
            raise RuntimeError(
                f"filter {raid_filter!r} matched 0 rows in {raid_dataset}:{raid_split}. "
                "Inspect the dataset schema (e.g. `language`/`domain` fields) and "
                "override +eval.raid_filter accordingly."
            )

    if sample_size:
        n = min(int(sample_size), len(ds))
        if sample_seed is not None:
            ds = ds.shuffle(seed=int(sample_seed)).select(range(n))
            log.info(f"random subsample: n={n}  seed={int(sample_seed)}")
        else:
            ds = ds.select(range(n))
            log.info(f"head subsample: n={n} (set +eval.sample_seed=... for random)")
    log.info(
        f"running on {len(ds)} rows  batch_size={batch_size}  "
        f"max_len={cfg.data.max_len}  min_len={cfg.data.min_len}  device={device}"
    )

    logits, labels, attrs, n_skipped = predict_loop(
        ds,
        tok=tok,
        backbone=backbone,
        pawn=pawn,
        device=device,
        max_len=cfg.data.max_len,
        min_len=cfg.data.min_len,
        batch_size=batch_size,
        breakdown_keys=breakdown_keys,
        use_amp=use_amp,
    )
    log.info(f"scored {len(labels)} samples ({n_skipped} skipped)")

    if len(labels) == 0:
        raise RuntimeError("no scorable rows after filtering and length filter")

    probs = expit(logits)
    overall = compute_metrics(SimpleNamespace(predictions=logits, label_ids=labels))
    if len(np.unique(labels)) > 1:
        overall["tpr_at_fpr_0.05"] = tpr_at_fpr(probs, labels, 0.05)
    log.info({k: round(float(v), 4) for k, v in overall.items()})

    breakdowns = {k: breakdown(probs, labels, attrs, k) for k in breakdown_keys}

    results = {
        "checkpoint": str(checkpoint),
        "backbone": cfg.data.backbone,
        "raid_dataset": raid_dataset,
        "raid_config": raid_config,
        "raid_split": raid_split,
        "raid_filter": raid_filter,
        "n_scored": int(len(labels)),
        "n_skipped": n_skipped,
        "overall": overall,
        "breakdowns": breakdowns,
    }

    out_json = output_dir / "eval_raid_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"wrote {out_json}")

    if save_logits:
        np.savez(
            output_dir / "eval_raid_logits.npz",
            logits=logits,
            labels=labels,
        )


if __name__ == "__main__":
    main()
