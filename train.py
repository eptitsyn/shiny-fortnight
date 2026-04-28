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
      data=mage_llama  # if you create conf/data/mage_llama.yaml
"""
from __future__ import annotations

import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from pawn.data import CachedFeatureDataset, PAWNCollator
from pawn.metrics import compute_metrics
from pawn.model import PAWN
from tqdm import tqdm

log = logging.getLogger(__name__)


def is_main_process() -> bool:
    """True on rank 0 (or in single-process runs)."""
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

        # DDP knobs
        ddp_find_unused_parameters=t.ddp_find_unused_parameters,
        ddp_backend=t.ddp_backend,

        remove_unused_columns=False,
        label_names=["labels"],
        seed=cfg.seed,
        disable_tqdm=False,
    )


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
        use_metrics_nn=cfg.model.use_metrics_nn,
    )
    if is_main_process():
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"trainable params: {n/1e6:.2f}M")

    training_args = build_training_args(cfg)

    from collections import Counter
    from torch.utils.data import WeightedRandomSampler


    def make_balanced_sampler(train_ds: CachedFeatureDataset) -> WeightedRandomSampler:
        ys = train_ds.labels  # O(N) attribute access, no torch.load
        counts = Counter(ys)
        class_w = {c: 1.0 / n for c, n in counts.items()}
        weights = torch.tensor([class_w[y] for y in ys], dtype=torch.double)
        return WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )


    class BalancedTrainer(Trainer):
        def __init__(self, *args, train_sampler=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._train_sampler = train_sampler

        def _get_train_sampler(self, train_dataset=None):
            return self._train_sampler


    # in main():
    sampler = make_balanced_sampler(train_ds) if cfg.training.balanced_sampler else None
    if is_main_process() and sampler is not None:
        log.info(f"label distribution: {Counter(train_ds.labels)}")

    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PAWNCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg.training.early_stopping_patience,
        )],
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
