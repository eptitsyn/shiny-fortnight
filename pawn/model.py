from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils.generic import ModelOutput


@dataclass
class PAWNOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "elu": F.elu,
}


def _get_activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation function: {name}")
    return _ACTIVATIONS[name]


class SeqBatchNorm(nn.Module):
    """BatchNorm over the channel axis of a [B, T, C] tensor."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() >= 3:
            x = x.transpose(1, 2).contiguous()
        x = self.bn(x)
        if x.dim() >= 3:
            x = x.transpose(1, 2).contiguous()
        return x


def _get_norm_layer(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "batch":
        return SeqBatchNorm(dim, affine=False)
    if norm_type == "layer":
        return nn.LayerNorm(dim, bias=False)
    if norm_type == "rms":
        return nn.RMSNorm(dim)
    raise ValueError(f"Invalid norm type: {norm_type}")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        num_hidden_layers: int = 1,
        activation: str = "gelu",
        dropout: float = 0.1,
        norm_type: str = "layer",
        residual: bool = True,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim

        self.num_hidden_layers = num_hidden_layers
        self.activation_fn = _get_activation_fn(activation)
        self.residual = residual

        if num_hidden_layers == 0:
            self.linear = nn.Linear(input_dim, output_dim)
            return

        self.linear_layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers - 1)]
            + [nn.Linear(hidden_dim, output_dim)]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_layers = nn.ModuleList(
            [
                _get_norm_layer(norm_type, hidden_dim)
                for _ in range(num_hidden_layers - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_hidden_layers == 0:
            return self.linear(x)

        x = self.linear_layers[0](x)
        for linear, norm in zip(self.linear_layers[1:-1], self.norm_layers):
            h = linear(self.dropout(self.activation_fn(x)))
            x = norm(h + x) if self.residual else norm(h)
        return self.linear_layers[-1](self.dropout(self.activation_fn(x)))


class PAWN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_metrics: int = 5,
        num_hidden_features: int = 256,
        num_hidden_layers: int = 3,
        gate_nn_num_layers: int | None = None,
        num_gates: int | None = None,
        activation: str = "gelu",
        norm_type: str = "layer",
        residual: bool = True,
        concat_consecutive_hidden_states: bool = True,
        pos_embed_dim: int = 0,
        max_len: int = 512,
        aggregation_method: Literal["attention", "sigmoid", "mean"] = "attention",
        dropout: float = 0.0,
        dropout_tokens: float = 0.15,
    ):
        super().__init__()

        self.aggregation_method = aggregation_method
        self.dropout_tokens = dropout_tokens
        self.concat_consecutive_hidden_states = concat_consecutive_hidden_states
        self.hidden_dim = hidden_dim
        self.num_metrics = num_metrics
        self.pos_embed_dim = pos_embed_dim
        self.max_len = max_len

        if pos_embed_dim < 0 or pos_embed_dim % 2 != 0:
            raise ValueError("pos_embed_dim must be a non-negative even integer")
        if pos_embed_dim > 0:
            freqs = torch.arange(1, pos_embed_dim // 2 + 1, dtype=torch.float32) * (
                2.0 * torch.pi
            )
            self.register_buffer("pos_freqs", freqs, persistent=False)

        gate_nn_input_dim = (
            (2 if concat_consecutive_hidden_states else 1) * hidden_dim
            + 1
            + pos_embed_dim
        )
        num_gates = num_gates or num_hidden_features
        gate_nn_num_layers = (
            gate_nn_num_layers if gate_nn_num_layers is not None else num_hidden_layers
        )

        if num_gates > num_hidden_features or num_hidden_features % num_gates != 0:
            raise ValueError(
                f"num_gates ({num_gates}) must divide num_hidden_features "
                f"({num_hidden_features})"
            )

        self.gate_nn = MLP(
            input_dim=gate_nn_input_dim,
            hidden_dim=num_hidden_features,
            output_dim=num_gates,
            num_hidden_layers=gate_nn_num_layers,
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            residual=residual,
        )
        self.metrics_nn = MLP(
            input_dim=num_metrics,
            output_dim=num_hidden_features,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            residual=residual,
        )
        self.aggregate_nn = MLP(
            input_dim=num_hidden_features,
            hidden_dim=num_hidden_features,
            output_dim=1,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            residual=residual,
        )

    def _dropout_tokens(self, mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout_tokens == 0:
            return mask

        B, T = mask.size()
        device = mask.device
        dropout_mask = torch.rand(B, T, device=device) < self.dropout_tokens
        final_mask = dropout_mask | mask
        while final_mask.all(dim=-1).any().item():
            dropout_mask = torch.rand(B, T, device=device) < self.dropout_tokens
            final_mask = dropout_mask | mask
        return final_mask

    @torch.no_grad()
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _position_features(
        self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device, dtype=dtype) / float(self.max_len)
        pos = pos.view(1, seq_len, 1).expand(batch_size, seq_len, 1)
        if self.pos_embed_dim == 0:
            return pos

        pos_freqs = cast(torch.Tensor, self.pos_freqs)
        angles = pos * pos_freqs.to(device=device, dtype=dtype)
        return torch.cat([pos, angles.sin(), angles.cos()], dim=-1)

    def forward(
        self,
        hs_curr: torch.Tensor,
        hs_next: torch.Tensor,
        metrics: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> PAWNOutput:
        B, T, _ = hs_curr.shape
        dtype = next(self.metrics_nn.parameters()).dtype
        hs_curr = hs_curr.to(dtype)
        hs_next = hs_next.to(dtype)
        metrics = metrics.to(dtype)
        if metrics.size(-1) < self.num_metrics:
            raise ValueError(
                f"metrics has {metrics.size(-1)} channels, expected at least {self.num_metrics}"
            )
        metrics = metrics[..., : self.num_metrics]

        processed_metrics = self.metrics_nn(metrics)

        if self.aggregation_method == "mean":
            am = attention_mask.float()
            coeffs = am / am.sum(dim=-1, keepdim=True).clamp_min(1.0)
            pooled = torch.einsum("btf,bt->bf", processed_metrics, coeffs)
            return PAWNOutput(logits=self.aggregate_nn(pooled).squeeze(-1))

        gate_x_list = [hs_curr]
        if self.concat_consecutive_hidden_states:
            gate_x_list.append(hs_next)
        gate_x_list.append(self._position_features(B, T, hs_curr.device, hs_curr.dtype))
        gate_x = torch.cat(gate_x_list, dim=-1)

        gate_mask = self._dropout_tokens(attention_mask == 0)
        gate_logits = self.gate_nn(gate_x)
        gate_logits = gate_logits.masked_fill(gate_mask.unsqueeze(-1), float("-inf"))

        G, Fdim = gate_logits.size(-1), processed_metrics.size(-1)
        if 1 < G < Fdim:
            gate_logits = gate_logits.repeat(1, 1, Fdim // G)

        if self.aggregation_method == "attention":
            pooled = (gate_logits.softmax(dim=-2) * processed_metrics).sum(dim=-2)
        elif self.aggregation_method == "sigmoid":
            valid_count = attention_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            pooled = (gate_logits.sigmoid() * processed_metrics).sum(
                dim=-2
            ) / valid_count
        else:
            raise ValueError(f"Unknown aggregation_method: {self.aggregation_method!r}")

        return PAWNOutput(logits=self.aggregate_nn(pooled).squeeze(-1))
