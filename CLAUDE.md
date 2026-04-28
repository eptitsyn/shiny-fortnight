# CLAUDE.md

Reimplementation of PAWN — *Perplexity Attention Weighted Network* (Miralles-González et al., 2025, [arXiv:2501.03940](https://arxiv.org/pdf/2501.03940)) — for AI-generated text detection on MAGE.

The reference implementation lives in [ai-gen-detection/](ai-gen-detection/) (read-only, do not modify). This repo is the from-scratch port.

## Repo layout

- [pawn/model.py](pawn/model.py) — `PAWN` and the supporting `MLP` / `SeqBatchNorm` (mirrors `ai-gen-detection/src/models/llm_metrics_mlp.py` + `src/models/utils/{mlp,seq_batch_norm,activations}.py`).
- [pawn/data.py](pawn/data.py) — `CachedFeatureDataset` + `PAWNCollator` for the precomputed `.pt` shards.
- [pawn/index.py](pawn/index.py) — one-time directory index of shards (`_index.pt`).
- [pawn/metrics.py](pawn/metrics.py) — `compute_metrics` for HF `Trainer` (accuracy/F1/AUROC + best-threshold variants).
- [extract_features.py](extract_features.py) — multi-GPU feature extraction (frozen LLM → `hs_curr`, `hs_next`, `metrics`, `attention_mask` → `.pt` shards). Replaces the original's online `FrozenPretrainedModel`.
- [train.py](train.py) — Hydra entrypoint, `PAWNTrainer` subclass with paper loss (BCE + `pos_weight` + label smoothing).
- [conf/](conf/) — Hydra config tree (`data/`, `model/`, `training/`, `extract/`).
- [scripts/](scripts/) — `extract.sh`, `train.sh` (torchrun launchers).

## Key design choice vs the original

The reference repo computes hidden states + metrics **online** every batch via a frozen LLM. We **precompute once** to `.pt` shards and train on cached features. This means:

- `PAWN.forward` takes pre-extracted tensors, not raw text.
- The five metrics are baked into shards in **alphabetical** order — `[entropy, max_log_probs, next_token_log_probs, quantile, top_p]` — matching `frozen_pretrained_model.py:__get_metrics`'s `sorted(metrics)` behavior. **Re-extract if you change metric order.**
- `quantile` and `top_p` use `>=` (target token included in its own quantile/mass), matching the original.

## Workflow

```bash
# 1. Extract features (writes to $PAWN_CACHE/<backbone_tag>/<split>/*.pt)
./scripts/extract.sh data=mage_gpt2

# 2. Train PAWN on the cache
./scripts/train.sh data=mage_gpt2 model=pawn_small

# Common Hydra overrides
./scripts/train.sh model=pawn_simple                   # base config (3 metrics, learned pos-embed)
./scripts/train.sh data=mage_llama                     # LLaMA-3.2-1B backbone
./scripts/train.sh training.balanced_sampler=true      # sampler instead of pos_weight
./scripts/train.sh model.aggregation_method=sigmoid    # ablation
```

## Paper-faithful defaults

Defaults in `conf/model/pawn_small.yaml` + `conf/training/default.yaml` match `ai-gen-detection/configs/experiment/mage/llm_metrics_mlp_gpt2.yaml`:

- 256 hidden features × 3 layers, GELU, LayerNorm(bias=False), residual=true
- `num_gates = num_hidden_features` (one gate per feature; no repeat)
- `concat_consecutive_hidden_states = true`, `pos_embed_dim = 0` (normalized scalar position)
- `dropout = 0.0`, `dropout_tokens = 0.15`
- `lr = 1e-3`, `weight_decay = 1e-2`, batch 128, 5 epochs, cosine, no warmup, `max_grad_norm = 1.0`
- BCE-with-logits + `pos_weight = 0.413` + label smoothing `y' = y(1-ε) + 0.5ε` with `ε = 0.2`

`pawn_simple` mirrors the base `configs/model/llm_metrics_mlp.yaml` (128/1, learned pos-embed dim 16, 3 metrics, dropout 0.1).

## Things not (yet) ported

- `LLMMetricsConv`, `HiddenStatesFF`, `LM`, `Ensemble`, `Binoculars` — only `LLMMetricsMLP` is in scope.
- Online `FrozenPretrainedModel` cache logic (we don't need it; features live on disk).
- W&B logger (we use TensorBoard via HF Trainer).
- `WeightDecayParamFilter` — HF Trainer's built-in `weight_decay` already excludes biases/LayerNorm.

## Editing rules for me (Claude)

- **Don't modify `ai-gen-detection/`** — it's the reference, kept verbatim. If a divergence is found, fix the port, not the reference.
- When changing the model or extraction logic, check `ai-gen-detection/src/models/llm_metrics_mlp.py` and `src/models/utils/frozen_pretrained_model.py` first; mirror naming and behavior.
- Anything that changes the contents of a feature shard (metric order, dtype, included metrics, `>=` vs `>`) requires a **re-extract** — call this out explicitly when proposing such a change.
- Hydra configs are the source of truth for hyperparameters; don't hardcode values in `train.py` / `extract_features.py`.
