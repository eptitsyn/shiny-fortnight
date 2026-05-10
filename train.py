"""
Hydra entrypoint for training PAWN with HuggingFace Trainer.

Examples:
  python train.py
  torchrun --standalone --nproc_per_node=2 train.py data=mage_gpt2
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Any, cast

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Sampler
from torch.utils.data import WeightedRandomSampler
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from pawn.data import CachedFeatureDataset, FilterSpec, PAWNCollator
from pawn.metrics import compute_metrics
from pawn.model import PAWN

log = logging.getLogger(__name__)


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


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


def _as_testbeds(node) -> dict[str, dict[str, Any]] | None:
    value = _to_plain(node)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("data.testbeds must be a mapping from name to spec")

    out: dict[str, dict[str, Any]] = {}
    for name, spec in value.items():
        if isinstance(spec, str):
            out[str(name)] = {"dir": spec}
        elif isinstance(spec, dict):
            out[str(name)] = cast(dict[str, Any], spec)
        else:
            raise TypeError(f"invalid testbed spec for {name!r}: {type(spec).__name__}")
    return out


def _build_eval_datasets(cfg: DictConfig) -> CachedFeatureDataset | dict[str, CachedFeatureDataset]:
    testbeds = _as_testbeds(cfg.data.get("testbeds"))
    if not testbeds:
        return CachedFeatureDataset(
            str(cfg.data.eval_dir),
            max_len=int(cfg.data.max_len),
            filter=_to_filter(cfg.data.get("eval_filter")),
        )

    out: dict[str, CachedFeatureDataset] = {}
    for name, spec in testbeds.items():
        d = str(spec.get("dir", cfg.data.eval_dir))
        if not os.path.isabs(d) and not os.path.exists(d):
            cand = os.path.join(str(cfg.data.cache_dir), d)
            if os.path.exists(cand):
                d = cand
        out[name] = CachedFeatureDataset(
            d,
            max_len=int(cfg.data.max_len),
            filter=_to_filter(spec.get("filter")),
        )
    return out


def _training_args_kwargs(cfg: DictConfig) -> dict[str, Any]:
    t = cfg.training
    return dict(
        output_dir=str(t.output_dir),
        num_train_epochs=float(t.num_train_epochs),
        per_device_train_batch_size=int(t.per_device_train_batch_size),
        per_device_eval_batch_size=int(t.per_device_eval_batch_size),
        gradient_accumulation_steps=int(t.gradient_accumulation_steps),
        learning_rate=float(t.learning_rate),
        weight_decay=float(t.weight_decay),
        warmup_steps=int(t.warmup_steps),
        lr_scheduler_type=str(t.lr_scheduler_type),
        max_grad_norm=float(t.max_grad_norm),
        eval_strategy=str(t.eval_strategy),
        save_strategy=str(t.save_strategy),
        save_total_limit=int(t.save_total_limit),
        load_best_model_at_end=True,
        metric_for_best_model=str(t.metric_for_best_model),
        greater_is_better=bool(t.greater_is_better),
        logging_strategy="steps",
        logging_steps=int(t.logging_steps),
        logging_first_step=True,
        report_to=_to_plain(t.report_to),
        bf16=t.bf16 and torch.cuda.is_available(),
        dataloader_num_workers=int(t.dataloader_num_workers),
        dataloader_pin_memory=t.dataloader_pin_memory and torch.cuda.is_available(),
        remove_unused_columns=False,
        label_names=["labels"],
        seed=int(cfg.seed),
        disable_tqdm=False,
    )


def build_training_args(cfg: DictConfig) -> TrainingArguments:
    kwargs = _training_args_kwargs(cfg)
    report_to = _to_plain(cfg.training.report_to)
    if report_to == "tensorboard" or (
        isinstance(report_to, list) and "tensorboard" in report_to
    ):
        os.environ.setdefault(
            "TENSORBOARD_LOGGING_DIR",
            os.path.join(str(cfg.training.output_dir), "tb"),
        )
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        kwargs["ddp_find_unused_parameters"] = bool(cfg.training.ddp_find_unused_parameters)
        kwargs["ddp_backend"] = str(cfg.training.ddp_backend)
    return TrainingArguments(**kwargs)


def make_balanced_sampler(train_ds: CachedFeatureDataset) -> WeightedRandomSampler:
    ys = train_ds.labels
    counts = Counter(ys)
    class_w = {c: 1.0 / n for c, n in counts.items()}
    weights = [class_w[y] for y in ys]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class PAWNTrainer(Trainer):
    def __init__(
        self,
        *args,
        label_smoothing: float = 0.0,
        pos_weight: float = 1.0,
        train_sampler: Sampler | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.label_smoothing = float(label_smoothing)
        self._pos_weight = float(pos_weight)
        self._train_sampler = train_sampler

    def _get_train_sampler(self, train_dataset=None):
        if self._train_sampler is not None:
            return self._train_sampler
        return super()._get_train_sampler(train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        _ = num_items_in_batch
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        smoothed = (
            labels.float() * (1.0 - self.label_smoothing)
            + 0.5 * self.label_smoothing
        )
        pos_weight = torch.tensor(
            self._pos_weight,
            device=logits.device,
            dtype=logits.dtype,
        )
        loss = F.binary_cross_entropy_with_logits(
            logits,
            smoothed,
            pos_weight=pos_weight,
        )

        inputs["labels"] = labels
        outputs.loss = loss
        return (loss, outputs) if return_outputs else loss


def build_model(cfg: DictConfig) -> PAWN:
    return PAWN(
        hidden_dim=int(cfg.data.hidden_dim),
        num_metrics=int(cfg.model.num_metrics),
        num_hidden_features=int(cfg.model.num_hidden_features),
        num_hidden_layers=int(cfg.model.num_hidden_layers),
        gate_nn_num_layers=(
            None
            if cfg.model.gate_nn_num_layers is None
            else int(cfg.model.gate_nn_num_layers)
        ),
        num_gates=None if cfg.model.num_gates is None else int(cfg.model.num_gates),
        activation=str(cfg.model.activation),
        norm_type=str(cfg.model.norm_type),
        residual=bool(cfg.model.residual),
        concat_consecutive_hidden_states=bool(cfg.model.concat_consecutive_hidden_states),
        pos_embed_dim=int(cfg.model.pos_embed_dim),
        max_len=int(cfg.model.max_len),
        aggregation_method=cfg.model.aggregation_method,
        dropout=float(cfg.model.dropout),
        dropout_tokens=float(cfg.model.dropout_tokens),
    )


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    if is_main_process():
        log.info("\n" + OmegaConf.to_yaml(cfg))
        os.makedirs(str(cfg.training.output_dir), exist_ok=True)
        OmegaConf.save(cfg, os.path.join(str(cfg.training.output_dir), "config.yaml"))

    train_ds = CachedFeatureDataset(
        str(cfg.data.train_dir),
        max_len=int(cfg.data.max_len),
        filter=_to_filter(cfg.data.get("train_filter")),
    )
    eval_ds = _build_eval_datasets(cfg)

    if is_main_process():
        if isinstance(eval_ds, dict):
            eval_sizes = {k: len(v) for k, v in eval_ds.items()}
            log.info(f"train={len(train_ds)} eval={eval_sizes}")
        else:
            log.info(f"train={len(train_ds)} eval={len(eval_ds)}")

    model = build_model(cfg)
    if is_main_process():
        log.info(f"trainable params: {model.num_trainable_params() / 1e6:.2f}M")

    sampler = make_balanced_sampler(train_ds) if cfg.training.balanced_sampler else None
    if is_main_process() and sampler is not None:
        log.info(f"label distribution: {Counter(train_ds.labels)}")

    trainer = PAWNTrainer(
        model=model,
        args=build_training_args(cfg),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PAWNCollator(),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=int(cfg.training.early_stopping_patience),
            )
        ],
        label_smoothing=float(cfg.training.label_smoothing),
        pos_weight=float(cfg.training.pos_weight),
        train_sampler=sampler,
    )

    trainer.train()

    if is_main_process():
        trainer.save_model(os.path.join(str(cfg.training.output_dir), "best"))

    eval_metrics = trainer.evaluate()
    if is_main_process():
        log.info({k: round(v, 4) for k, v in eval_metrics.items() if isinstance(v, (int, float))})
        with open(os.path.join(str(cfg.training.output_dir), "eval_metrics.json"), "w") as f:
            json.dump(
                {k: v for k, v in eval_metrics.items() if isinstance(v, (int, float, str))},
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
