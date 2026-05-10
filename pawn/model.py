from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput


@dataclass
class PAWNOutput(ModelOutput):
    logits: Optional[torch.Tensor] = None  # [B] raw logit


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
            [_get_norm_layer(norm_type, hidden_dim) for _ in range(num_hidden_layers - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_hidden_layers == 0:
            return self.linear(x)

        x = self.linear_layers[0](x)
        for linear, norm in zip(self.linear_layers[1:-1], self.norm_layers):
            h = linear(self.dropout(self.activation_fn(x)))
            x = norm(h + x) if self.residual else norm(h)
        return self.linear_layers[-1](self.dropout(self.activation_fn(x)))


class GateConvContext(nn.Module):
    """Depthwise+pointwise conv stack over time. Mixes neighbor info into gate input.

    Pre-zeros padding so neighbors of valid tokens don't pull from garbage values.
    Residual update is masked so padded rows pass through unchanged.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 9,
        num_layers: int = 1,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size ({kernel_size}) must be odd")
        self.activation_fn = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        pad = kernel_size // 2
        self.norms = nn.ModuleList(
            [nn.LayerNorm(dim, bias=False) for _ in range(num_layers)]
        )
        self.dw_convs = nn.ModuleList(
            [nn.Conv1d(dim, dim, kernel_size, padding=pad, groups=dim) for _ in range(num_layers)]
        )
        self.pw_convs = nn.ModuleList(
            [nn.Conv1d(dim, dim, 1) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        m = attention_mask.unsqueeze(-1).to(x.dtype)  # [B, T, 1]
        for norm, dw, pw in zip(self.norms, self.dw_convs, self.pw_convs):
            h = norm(x) * m                       # zero pad rows
            h = h.transpose(1, 2)                 # [B, D, T]
            h = dw(h)
            h = self.activation_fn(h)
            h = pw(h)
            h = h.transpose(1, 2)                 # [B, T, D]
            x = x + self.dropout(h) * m           # mask update so pad rows untouched
        return x


class PAWN(nn.Module):


    def __init__(
        self,
        hidden_dim: int,
        num_metrics: int = 5,
        # MLP shape
        num_hidden_features: int = 256,
        num_hidden_layers: int = 3,
        gate_nn_num_layers: int | None = None,
        num_gates: int | None = None,
        activation: str = "gelu",
        norm_type: str = "layer",
        residual: bool = True,
        # Gate input composition
        concat_consecutive_hidden_states: bool = True,
        pos_encoding: Literal["normalized", "fourier", "learned", "hybrid"] = "hybrid",
        pos_embed_dim: int = 0,
        max_len: int = 512,
        # Gate context (neighbor mixing before gate MLP)
        gate_context: Literal["none", "conv"] = "none",
        gate_context_kernel: int = 9,
        gate_context_layers: int = 1,
        # Aggregation
        aggregation_method: Literal["attention", "sigmoid", "mean"] = "attention",
        # Metric-sequence DFT features
        dft_features: bool = False,
        dft_num_bins: int = 8,
        dft_metric_indices: Sequence[int] | None = None,
        dft_log_scale: bool = True,
        dft_eps: float = 1e-8,
        metrics_clip_value: float | None = None,
        # Regularization
        dropout: float = 0.0,
        dropout_tokens: float = 0.15,
    ):
        super().__init__()

        self.aggregation_method = aggregation_method
        self.dropout_tokens = dropout_tokens
        self.concat_consecutive_hidden_states = concat_consecutive_hidden_states
        self.hidden_dim = hidden_dim
        self.num_metrics = num_metrics
        self.pos_encoding = pos_encoding
        self.pos_embed_dim = pos_embed_dim
        self.max_len = max_len
        self.dft_features = dft_features
        self.dft_num_bins = int(dft_num_bins)
        self.dft_log_scale = dft_log_scale
        self.dft_eps = float(dft_eps)
        self.metrics_clip_value = metrics_clip_value

        if dft_metric_indices is None:
            dft_metric_indices = tuple(range(num_metrics))
        self.dft_metric_indices = tuple(int(i) for i in dft_metric_indices)
        if self.dft_features:
            if self.dft_num_bins <= 0:
                raise ValueError(
                    f"dft_num_bins ({self.dft_num_bins}) must be > 0 when "
                    "dft_features=true"
                )
            if self.dft_eps <= 0:
                raise ValueError(
                    f"dft_eps ({self.dft_eps}) must be > 0 when dft_features=true"
                )
            invalid = [i for i in self.dft_metric_indices if i < 0 or i >= num_metrics]
            if invalid:
                raise ValueError(
                    f"dft_metric_indices contains out-of-range indices {invalid}; "
                    f"num_metrics={num_metrics}"
                )

        if pos_embed_dim < 0:
            raise ValueError(
                f"pos_embed_dim ({pos_embed_dim}) must be a non-negative integer"
            )
        if pos_encoding not in {"normalized", "fourier", "learned", "hybrid"}:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding!r}")
        if pos_encoding in {"learned", "hybrid"} and pos_embed_dim <= 0:
            raise ValueError(
                f"pos_embed_dim ({pos_embed_dim}) must be > 0 when "
                f"pos_encoding={pos_encoding!r}"
            )
        if pos_encoding == "fourier" and pos_embed_dim % 2 != 0:
            raise ValueError(
                f"pos_embed_dim ({pos_embed_dim}) must be even when "
                f"pos_encoding='fourier' (sin/cos pairs)"
            )
        if pos_encoding == "fourier" and pos_embed_dim > 0:
            freqs = torch.arange(1, pos_embed_dim // 2 + 1, dtype=torch.float32) * (
                2.0 * torch.pi
            )
            self.register_buffer("pos_freqs", freqs, persistent=False)
        if pos_encoding in {"learned", "hybrid"}:
            self.pos_embedding = nn.Embedding(max_len, pos_embed_dim)

        pos_input_dim = 2
        if pos_encoding in {"fourier", "learned"}:
            pos_input_dim += pos_embed_dim
        elif pos_encoding == "hybrid":
            pos_input_dim += pos_embed_dim + 1

        gate_nn_input_dim = (
            (2 if concat_consecutive_hidden_states else 1) * hidden_dim
            + pos_input_dim
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

        if gate_context == "conv":
            self.gate_context_nn = GateConvContext(
                dim=gate_nn_input_dim,
                kernel_size=gate_context_kernel,
                num_layers=gate_context_layers,
                activation=activation,
                dropout=dropout,
            )
        elif gate_context == "none":
            self.gate_context_nn = None
        else:
            raise ValueError(f"Unknown gate_context: {gate_context!r}")

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
        metrics_input_dim = num_metrics
        if self.dft_features:
            metrics_input_dim += self.dft_num_bins * len(self.dft_metric_indices)

        self.metrics_nn = MLP(
            input_dim=metrics_input_dim,
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

    # ------------------------------------------------------------------ utils

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

    def _append_dft_features(
        self, metrics: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        metrics = torch.nan_to_num(metrics, nan=0.0, posinf=0.0, neginf=0.0)
        if self.metrics_clip_value is not None:
            metrics = metrics.clamp(
                min=-float(self.metrics_clip_value),
                max=float(self.metrics_clip_value),
            )
        if not self.dft_features:
            return metrics

        x = metrics[..., self.dft_metric_indices].float()  # [B, T, C]
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * mask).sum(dim=1, keepdim=True) / lengths
        centered = (x - mean) * mask

        coeffs = torch.fft.rfft(centered, dim=1, norm="ortho")
        spectrum = torch.sqrt(coeffs.real.square() + coeffs.imag.square() + self.dft_eps)
        spectrum = spectrum[:, 1 : self.dft_num_bins + 1]  # skip near-zero DC bin
        if spectrum.size(1) < self.dft_num_bins:
            pad = self.dft_num_bins - spectrum.size(1)
            spectrum = F.pad(spectrum, (0, 0, 0, pad))
        if self.dft_log_scale:
            spectrum = torch.log1p(spectrum)

        dft = spectrum.flatten(1).to(metrics.dtype)
        dft = dft.unsqueeze(1).expand(metrics.size(0), metrics.size(1), -1)
        return torch.cat([metrics, dft], dim=-1)

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
    ) -> PAWNOutput:
        B, T, _ = hs_curr.shape
        device = hs_curr.device

        metrics = self._append_dft_features(metrics, attention_mask)
        processed_metrics = self.metrics_nn(metrics)  # [B, T, F]

        if self.aggregation_method == "mean":
            am = attention_mask.float()
            coeffs = am / am.sum(dim=-1, keepdim=True).clamp_min(1.0)
            pooled = torch.einsum("blf,bl->bf", processed_metrics, coeffs)
            logits = self.aggregate_nn(pooled).squeeze(-1)
            return PAWNOutput(logits=logits)

        # ---- gate input ----
        gate_x_list = [hs_curr]
        if self.concat_consecutive_hidden_states:
            gate_x_list.append(hs_next)

        T_actual = (
            attention_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(hs_curr.dtype)
        )  # [B, 1]
        pos_idx = torch.arange(T, device=device, dtype=hs_curr.dtype)  # [T]
        p = pos_idx.unsqueeze(0) / T_actual  # [B, T] — padding positions get p > 1, masked out downstream
        pos_feats = [p.unsqueeze(-1), (1.0 - p).unsqueeze(-1)]
        if self.pos_encoding == "fourier" and self.pos_embed_dim > 0:
            angles = p.unsqueeze(-1) * self.pos_freqs  # [B, T, K]
            pos_feats.extend([angles.sin(), angles.cos()])
        elif self.pos_encoding in {"learned", "hybrid"}:
            pos_ids = torch.arange(T, device=device).clamp_max(self.max_len - 1)
            pos_abs = self.pos_embedding(pos_ids).unsqueeze(0).expand(B, T, -1)
            pos_feats.append(pos_abs.to(hs_curr.dtype))
            if self.pos_encoding == "hybrid":
                length = T_actual / float(self.max_len)
                length = length.clamp_max(1.0).unsqueeze(-1).expand(B, T, 1)
                pos_feats.append(length)
        pos_embed = torch.cat(pos_feats, dim=-1)
        gate_x_list.append(pos_embed)
        gate_x = torch.cat(gate_x_list, dim=-1)

        if self.gate_context_nn is not None:
            gate_x = self.gate_context_nn(gate_x, attention_mask)

        # ---- gate logits with masking ----
        gate_mask = self._dropout_tokens(attention_mask == 0)  # [B, T]
        gate_logits = self.gate_nn(gate_x)                      # [B, T, G]
        gate_logits = gate_logits.masked_fill(
            gate_mask.unsqueeze(-1), float("-inf")
        )

        G, Fdim = gate_logits.size(-1), processed_metrics.size(-1)
        if 1 < G < Fdim:
            gate_logits = gate_logits.repeat(1, 1, Fdim // G)  # [B, T, F]

        if self.aggregation_method == "attention":
            pooled = (gate_logits.softmax(dim=-2) * processed_metrics).sum(dim=-2)
        elif self.aggregation_method == "sigmoid":
            valid_count = attention_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            pooled = (gate_logits.sigmoid() * processed_metrics).sum(dim=-2) / valid_count
        else:
            raise ValueError(f"Unknown aggregation_method: {self.aggregation_method!r}")

        logits = self.aggregate_nn(pooled).squeeze(-1)
        return PAWNOutput(logits=logits)
