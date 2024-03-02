import pytorch_lightning as pl
import torch
import torch.nn as nn


class ITransformer(pl.LightningModule):
    def __init__(
        self,
        lighten,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = False,
    ):
        super().__init__()
        self.name = "itransformer"
        if lighten:
            self.name += "-lighten"

        d_model = 64 if not lighten else 32
        dim_feedforward = 256 if not lighten else 128
        nhead = 8 if not lighten else 4
        num_layers = 2 if not lighten else 1

        self.embed = nn.Linear(100, d_model, bias=False)
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
        x = x.squeeze(1)
        # transpose
        x = x.permute(0, 2, 1)
        x = self.embed(x)

        # transformer encoder
        x = self.transformer_encoder(x)

        # mean pool for classification
        x = torch.mean(x, dim=1)

        logits = self.cat_head(x)
        return logits
