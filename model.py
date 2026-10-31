"""Nevrale nettverk / Neural network architectures.

Two models, both fully configurable:
  * TextBrain  - token ids -> Embedding -> mean pooling -> MLP   (text path)
  * DenseBrain - ready-made feature/embedding vectors -> MLP     (plug in your own embeddings)

Both support task="classification" (logits, one per class) and
task="regression" (one continuous value per output).
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


def build_mlp(in_size: int, hidden_sizes: Sequence[int], out_size: int,
              dropout: float = 0.3, use_batchnorm: bool = True) -> nn.Sequential:
    """Linear -> (BatchNorm) -> ReLU -> Dropout, repeated, then a final Linear.
    Skjulte lag med valgfri BatchNorm og Dropout for bedre generalisering."""
    layers: List[nn.Module] = []
    prev = in_size
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_size))
    return nn.Sequential(*layers)


class DenseBrain(nn.Module):
    """MLP on fixed-size float vectors (e.g. pre-computed sentence embeddings)."""

    def __init__(self, input_size: int, num_outputs: int,
                 hidden_sizes: Sequence[int] = (128, 64), dropout: float = 0.3,
                 use_batchnorm: bool = True, task: str = "classification"):
        super().__init__()
        assert task in ("classification", "regression")
        self.task = task
        self.net = build_mlp(input_size, hidden_sizes, num_outputs, dropout, use_batchnorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextBrain(nn.Module):
    """Token ids -> Embedding -> masked mean pooling -> MLP.

    pretrained: optional (vocab_size, embed_dim) tensor, e.g. word vectors you
    computed yourself - lets you plug in pre-computed embeddings.
    """

    def __init__(self, vocab_size: int, num_outputs: int, embed_dim: int = 64,
                 hidden_sizes: Sequence[int] = (128, 64), dropout: float = 0.3,
                 use_batchnorm: bool = True, task: str = "classification",
                 pad_idx: int = 0, pretrained: Optional[torch.Tensor] = None,
                 freeze_embeddings: bool = False):
        super().__init__()
        assert task in ("classification", "regression")
        self.task = task
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained is not None:
            if tuple(pretrained.shape) != (vocab_size, embed_dim):
                raise ValueError(f"pretrained must be {(vocab_size, embed_dim)}, got {tuple(pretrained.shape)}")
            self.embedding.weight.data.copy_(pretrained)
        self.embedding.weight.requires_grad = not freeze_embeddings
        self.embed_dropout = nn.Dropout(dropout)
        self.mlp = build_mlp(embed_dim, hidden_sizes, num_outputs, dropout, use_batchnorm)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (batch, seq_len) int64
        mask = (ids != self.pad_idx).unsqueeze(-1).float()          # (B, L, 1)
        emb = self.embedding(ids) * mask                             # (B, L, D)
        pooled = emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)     # mean over real tokens
        return self.mlp(self.embed_dropout(pooled))


def build_model(config: dict, pretrained: Optional[torch.Tensor] = None) -> nn.Module:
    """Bygg modell fra en config-dict (lagres i checkpoint). / Build model from a saved config."""
    cfg = dict(config)
    kind = cfg.pop("kind", "text")
    if kind == "text":
        return TextBrain(pretrained=pretrained, **cfg)
    if kind == "dense":
        return DenseBrain(**cfg)
    raise ValueError(f"Unknown model kind: {kind}")
