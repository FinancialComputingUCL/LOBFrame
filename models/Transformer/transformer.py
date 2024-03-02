from typing import Optional

import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(
        self, input_ids_shape: torch.Size, past_key_values_length: int = 0
    ) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        _, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class Transformer(pl.LightningModule):
    def __init__(
        self,
        lighten,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = False,
    ):
        super().__init__()
        self.name = "transformer"
        if lighten:
            self.name += "-lighten"

        d_model = 64 if not lighten else 32
        dim_feedforward = 256 if not lighten else 128
        nhead = 8 if not lighten else 4
        num_layers = 2 if not lighten else 1

        self.embed = nn.Linear(40, d_model, bias=False)

        self.embed_positions = SinusoidalPositionalEmbedding(100, d_model)

        layer_norm_eps: float = 1e-5
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            batch_first=True,
        )
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=encoder_norm
        )
        self.cat_head = nn.Linear(d_model, 3)

    def forward(self, x):
        x = self.embed(x.squeeze(1))

        embed_pos = self.embed_positions(x.shape)

        # transformer encoder
        x = self.transformer_encoder(x + embed_pos)

        # mean pool for classification
        x = torch.mean(x, dim=1)

        logits = self.cat_head(x)
        return logits
