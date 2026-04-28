from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput


@dataclass
class PAWNOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None  # [B] raw logit


class MLP(nn.Module):
    """MLP with GELU + dropout between hidden layers, no activation on output.
    dims = [in, h1, h2, ..., out]. If len(dims) == 2, this is a single Linear."""

    def __init__(self, dims: list[int], dropout: float = 0.1):
        super().__init__()
        if len(dims) < 2:
            raise ValueError(f"MLP needs at least [in, out]; got {dims}")
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers += [nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PAWN(nn.Module):
    """Perplexity Attention Weighted Network (Miralles-González et al., 2025).

    Inputs (pre-extracted from a frozen LLM):
      hs_curr:        [B, T, H]  hidden state at position t
      hs_next:        [B, T, H]  hidden state at position t+1
      metrics:        [B, T, M]  M raw next-token-distribution metrics
                                 (log_prob, entropy, max_log_prob, rank, top_p)
      attention_mask: [B, T]     1 = valid token, 0 = padding
      labels:         [B]        in {0, 1}, 1 = machine-generated  (optional)

    Pipeline:
      1. metrics_nn:    [B,T,M] -> [B,T,F]
                        Per-token MLP that learns nonlinear metric combinations.
                        If use_metrics_nn=False, F is set to M and metrics pass
                        through identity (the simplified ablation).
      2. gate inputs:   [h_t || h_{t+1} || pos/max_len] -> [B, T, 2H+1]
      3. gate_nn:       [B,T,2H+1] -> [B,T,G] with G | F
                        Optionally dropout valid tokens during training.
                        Pad/dropped positions get -inf so softmax ignores them.
      4. gate repeat:   [B,T,G] -> [B,T,F] by tiling along feature axis
                        (each gate channel is shared by F/G consecutive features)
      5. softmax(dim=T): per-feature attention over time
      6. weighted sum:  sum_t gate[b,t,f] * processed_metrics[b,t,f] -> [B, F]
      7. aggregate_nn:  [B,F] -> [B] logit
    """

    def __init__(
        self,
        hidden_dim: int,
        num_metrics: int = 5,
        feature_dim: int = 64,
        num_gates: int = 1,
        max_len: int = 512,
        # metrics_nn
        use_metrics_nn: bool = True,
        metrics_hidden: int = 128,
        metrics_depth: int = 2,
        # gate_nn
        gate_hidden: int = 512,
        gate_depth: int = 2,
        # aggregate_nn
        agg_hidden: int = 256,
        agg_depth: int = 2,
        # regularization
        dropout: float = 0.1,
        token_dropout: float = 0.0,
    ):
        super().__init__()

        # If metrics_nn is disabled, the per-token feature dim equals the
        # number of raw metrics (the "simplified PAWN" ablation).
        if not use_metrics_nn:
            feature_dim = num_metrics

        if feature_dim % num_gates != 0:
            raise ValueError(
                f"num_gates ({num_gates}) must divide feature_dim ({feature_dim})"
            )

        self.hidden_dim = hidden_dim
        self.num_metrics = num_metrics
        self.feature_dim = feature_dim
        self.num_gates = num_gates
        self.max_len = max_len
        self.use_metrics_nn = use_metrics_nn
        self.token_dropout = token_dropout

        # 1. metrics_nn: M -> F (or identity if disabled)
        if use_metrics_nn:
            metrics_dims = (
                [num_metrics] + [metrics_hidden] * metrics_depth + [feature_dim]
            )
            self.metrics_nn: nn.Module = MLP(metrics_dims, dropout)
        else:
            self.metrics_nn = nn.Identity()

        # 3. gate_nn: 2H + 1 -> G
        gate_in = 2 * hidden_dim + 1
        gate_dims = [gate_in] + [gate_hidden] * gate_depth + [num_gates]
        self.gate_nn = MLP(gate_dims, dropout)

        # 7. aggregate_nn: F -> 1
        agg_dims = [feature_dim] + [agg_hidden] * agg_depth + [1]
        self.aggregate_nn = MLP(agg_dims, dropout)

    # ------------------------------------------------------------------ utils

    def _drop_tokens(self, invalid_mask: torch.Tensor) -> torch.Tensor:
        """Optionally mark additional valid tokens as invalid during training.

        invalid_mask: [B, T] bool, True where token is pad-or-already-dropped.
        Returns a mask of the same shape with extra True entries.
        """
        if not self.training or self.token_dropout <= 0.0:
            return invalid_mask
        drop = torch.rand_like(invalid_mask, dtype=torch.float) < self.token_dropout
        return invalid_mask | drop

    @torch.no_grad()
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        hs_curr: torch.Tensor,           # [B, T, H]
        hs_next: torch.Tensor,           # [B, T, H]
        metrics: torch.Tensor,           # [B, T, M]
        attention_mask: torch.Tensor,    # [B, T]
        labels: Optional[torch.Tensor] = None,  # [B]
    ) -> PAWNOutput:
        B, T, _ = hs_curr.shape
        device = hs_curr.device
        dtype = hs_curr.dtype

        # 1. project metrics
        processed_metrics = self.metrics_nn(metrics)  # [B, T, F]

        # 2. build gate inputs: hidden states + normalized position
        # position normalized by max_len (fixed), not T — keeps positions
        # comparable across sequences of different lengths.
        pos = torch.arange(T, device=device, dtype=dtype) / self.max_len
        pos = pos.view(1, T, 1).expand(B, T, 1)
        gate_x = torch.cat([hs_curr, hs_next, pos], dim=-1)  # [B, T, 2H+1]

        # 3. gate logits, with masking
        gate_logits = self.gate_nn(gate_x)  # [B, T, G]

        invalid = (attention_mask == 0)
        invalid = self._drop_tokens(invalid)
        gate_logits = gate_logits.masked_fill(
            invalid.unsqueeze(-1), float("-inf")
        )

        # 4. expand gates from G channels to F by tiling
        G, Fdim = self.num_gates, self.feature_dim
        if G < Fdim:
            gate_logits = gate_logits.repeat(1, 1, Fdim // G)  # [B, T, F]
        # If G == Fdim, gate_logits is already [B, T, F].
        # G > Fdim is impossible by construction (validated in __init__).

        # 5. softmax over the time dimension (per feature channel)
        gates = gate_logits.softmax(dim=-2)  # [B, T, F]

        # 6. weighted sum across time
        pooled = (gates * processed_metrics).sum(dim=-2)  # [B, F]

        # 7. classifier
        logits = self.aggregate_nn(pooled).squeeze(-1)  # [B]

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        return PAWNOutput(loss=loss, logits=logits)
