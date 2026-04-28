"""
Hydra entrypoint for training PAWN with HuggingFace Trainer + DDP.

Single GPU:
  python train.py

Two GPUs on one node (recommended for your setup):
  torchrun --standalone --nproc_per_node=2 train.py

Override anything from the CLI:
  torchrun --standalone --nproc_per_node=2 train.py \
      training.learning_rate=5e-4 training.num_train_epochs=10

Switch backbone:
  torchrun --standalone --nproc_per_node=2 train.py \
      data=mage_llama
"""
from __future__ import annotations

import logging
import os
from collections import Counter

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import WeightedRandomSampler
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from pawn.data import CachedFeatureDataset, PAWNCollator
from pawn.metrics import compute_metrics
from pawn.model import PAWN

log = logging.getLogger(__name__)


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def build_training_args(cfg: DictConfig) -> TrainingArguments:
    t = cfg.training
    return TrainingArguments(
        output_dir=t.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,

        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        max_grad_norm=t.max_grad_norm,

        eval_strategy=t.eval_strategy,
        save_strategy=t.save_strategy,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better,

        logging_dir=os.path.join(t.output_dir, "tb"),
        logging_strategy="steps",
        logging_steps=t.logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"],

        bf16=t.bf16 and torch.cuda.is_available(),
        dataloader_num_workers=t.dataloader_num_workers,
        dataloader_pin_memory=t.dataloader_pin_memory,

        ddp_find_unused_parameters=t.ddp_find_unused_parameters,
        ddp_backend=t.ddp_backend,

        remove_unused_columns=False,
        label_names=["labels"],
        seed=cfg.seed,
        disable_tqdm=False,
    )


def make_balanced_sampler(train_ds: CachedFeatureDataset) -> WeightedRandomSampler:
    ys = train_ds.labels
    counts = Counter(ys)
    class_w = {c: 1.0 / n for c, n in counts.items()}
    weights = torch.tensor([class_w[y] for y in ys], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class PAWNTrainer(Trainer):
    """Trainer that

    - reproduces the paper's loss: BCE-with-logits with `pos_weight` and the
      label smoothing scheme `y' = y * (1 - eps) + 0.5 * eps`
      (ai-gen-detection/train.py:103-105);
    - optionally swaps the random sampler for a balanced one.
    """

    def __init__(
        self,
        *args,
        label_smoothing: float = 0.0,
        pos_weight: float = 1.0,
        train_sampler=None,
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

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        smoothed = labels.float() * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        pos_weight = torch.tensor(self._pos_weight, device=logits.device, dtype=logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, smoothed, pos_weight=pos_weight)

        # Put labels back so compute_metrics can read them downstream.
        inputs["labels"] = labels
        outputs.loss = loss
        return (loss, outputs) if return_outputs else loss


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    if is_main_process():
        log.info("\n" + OmegaConf.to_yaml(cfg))
        os.makedirs(cfg.training.output_dir, exist_ok=True)
        with open(os.path.join(cfg.training.output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

    train_ds = CachedFeatureDataset(cfg.data.train_dir, max_len=cfg.data.max_len)
    eval_ds = CachedFeatureDataset(cfg.data.eval_dir, max_len=cfg.data.max_len)
    if is_main_process():
        log.info(f"train={len(train_ds)}  eval={len(eval_ds)}")

    model = PAWN(
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
    if is_main_process():
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"trainable params: {n/1e6:.2f}M")

    training_args = build_training_args(cfg)

    sampler = make_balanced_sampler(train_ds) if cfg.training.balanced_sampler else None
    if is_main_process() and sampler is not None:
        log.info(f"label distribution: {Counter(train_ds.labels)}")

    trainer = PAWNTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PAWNCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg.training.early_stopping_patience,
        )],
        label_smoothing=cfg.training.label_smoothing,
        pos_weight=cfg.training.pos_weight,
        train_sampler=sampler,
    )

    trainer.train()

    if is_main_process():
        trainer.save_model(os.path.join(cfg.training.output_dir, "best"))

    eval_metrics = trainer.evaluate()
    if is_main_process():
        log.info({k: round(v, 4) for k, v in eval_metrics.items()
                  if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()
