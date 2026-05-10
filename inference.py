"""Run PAWN inference on a raw text file.

Examples:
  python inference.py --checkpoint runs/.../best sample.txt
  python inference.py --checkpoint runs/.../best --one-per-line samples.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from eval import load_state_dict
from extract_features import compute_metrics_per_token, dtype_from_name
from train import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("txt_file", type=Path, help="Text file to classify")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="PAWN checkpoint directory or file, e.g. runs/.../best",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional Hydra config.yaml. Defaults to the config saved next to the run.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for inference. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Probability threshold for the machine/generated label. Defaults to "
            "model.decision_threshold from config.yaml when present, otherwise 0.5."
        ),
    )
    parser.add_argument(
        "--one-per-line",
        action="store_true",
        help="Classify each non-empty line as a separate sample instead of the whole file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text table.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides used only when no saved config is found, e.g. data=mage_llama.",
    )
    return parser.parse_args()


def default_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_config(checkpoint: Path) -> Path | None:
    candidates: list[Path] = []
    if checkpoint.is_dir():
        candidates.extend(
            [checkpoint / "config.yaml", checkpoint.parent / "config.yaml"]
        )
    else:
        candidates.extend(
            [
                checkpoint.parent / "config.yaml",
                checkpoint.parent.parent / "config.yaml",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    return None


def load_cfg(config_path: Path | None, checkpoint: Path, overrides: list[str]) -> DictConfig:
    path = config_path or find_config(checkpoint)
    if path is not None:
        return cast(DictConfig, OmegaConf.load(path))

    conf_dir = Path(__file__).resolve().parent / "conf"
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(conf_dir)):
        return hydra.compose(config_name="config", overrides=overrides)


def decision_threshold(cfg: DictConfig, override: float | None) -> float:
    if override is not None:
        return override
    threshold = cfg.model.get("decision_threshold")
    if threshold is not None:
        return float(threshold)

    thresholds = cfg.model.get("decision_thresholds")
    if thresholds and len(thresholds) == 1:
        return float(next(iter(thresholds.values())))
    return 0.5



def read_texts(path: Path, one_per_line: bool) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if one_per_line:
        return [line.strip() for line in text.splitlines() if line.strip()]
    return [text.strip()]


def load_backbone_for_inference(
    backbone: str,
    device: torch.device,
    model_dtype: torch.dtype,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tok = AutoTokenizer.from_pretrained(backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            backbone,
            torch_dtype=model_dtype,
            output_hidden_states=True,
        ),
    )
    model.to(device)  # pyright: ignore[reportArgumentType]
    model.eval()
    return tok, model


@torch.no_grad()
def text_to_features(
    text: str,
    *,
    tok: PreTrainedTokenizerBase,
    backbone: PreTrainedModel,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=int(cfg.data.max_len),
        padding=False,
    ).to(device)
    ids = enc["input_ids"]
    if ids.size(1) < int(cfg.data.min_len):
        raise ValueError(
            f"text is too short after tokenization: {ids.size(1)} tokens "
            f"(minimum {int(cfg.data.min_len)})"
        )

    out = backbone(**enc)
    hs = out.hidden_states[-1][0]
    metrics_dtype = dtype_from_name(str(cfg.extract.get("metrics_dtype", "float32")))
    metrics = compute_metrics_per_token(
        out.logits[0, :-1],
        ids[0, 1:],
        metrics_dtype=metrics_dtype,
    )
    length = min(int(hs.size(0) - 1), int(cfg.data.max_len))

    return {
        "hs_curr": hs[:-1][:length].float().unsqueeze(0),
        "hs_next": hs[1:][:length].float().unsqueeze(0),
        "metrics": metrics[:length].float().unsqueeze(0),
        "attention_mask": torch.ones(1, length, dtype=torch.long, device=device),
    }


@torch.no_grad()
def predict_texts(
    texts: list[str],
    *,
    cfg: DictConfig,
    checkpoint: Path,
    device: torch.device,
    threshold: float,
) -> list[dict[str, Any]]:
    pawn = build_model(cfg).to(device)
    missing, unexpected = pawn.load_state_dict(load_state_dict(checkpoint), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    pawn.eval()

    model_dtype = dtype_from_name(str(cfg.extract.get("model_dtype", "float32")))
    tok, backbone = load_backbone_for_inference(str(cfg.data.backbone), device, model_dtype)

    results: list[dict[str, Any]] = []
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    for idx, text in enumerate(texts):
        batch = text_to_features(
            text,
            tok=tok,
            backbone=backbone,
            cfg=cfg,
            device=device,
        )
        with autocast_ctx:
            out = pawn(**batch)
        logit = float(out.logits.detach().float().cpu().item())
        machine_prob = float(torch.sigmoid(torch.tensor(logit)).item())
        results.append(
            {
                "index": idx,
                "tokens": int(batch["attention_mask"].sum().item()),
                "logit": logit,
                "machine_probability": machine_prob,
                "human_probability": 1.0 - machine_prob,
                "prediction": "machine" if machine_prob >= threshold else "human",
                "threshold": threshold,
            }
        )
    return results


def print_table(results: list[dict[str, Any]]) -> None:
    print("idx\tprediction\tmachine_prob\thuman_prob\tlogit\ttokens")
    for row in results:
        print(
            f"{row['index']}\t{row['prediction']}\t"
            f"{row['machine_probability']:.4f}\t"
            f"{row['human_probability']:.4f}\t"
            f"{row['logit']:.4f}\t{row['tokens']}"
        )


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config, args.checkpoint, args.overrides)
    device = default_device(args.device)
    texts = read_texts(args.txt_file, args.one_per_line)
    if not texts:
        raise ValueError(f"{args.txt_file} contains no text to classify")

    results = predict_texts(
        texts,
        cfg=cfg,
        checkpoint=args.checkpoint,
        device=device,
        threshold=decision_threshold(cfg, args.threshold),
    )
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_table(results)


if __name__ == "__main__":
    main()
