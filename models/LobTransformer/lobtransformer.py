import pytorch_lightning as pl
import torch
import torch.nn as nn


class LobTransformer(pl.LightningModule):
    def __init__(self, lighten):
        super().__init__()
        self.name = "lobtransformer"
        if lighten:
            self.name += "-lighten"

        hidden = 32 if not lighten else 16
        d_model = hidden * 2 * 3
        nhead = 8 if not lighten else 4
        num_layers = 2 if not lighten else 1

        # Convolution blocks.
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=hidden, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
        )

        if lighten:
            conv3_kernel_size = 5
        else:
            conv3_kernel_size = 10

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden, kernel_size=(1, conv3_kernel_size)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden),
        )

        # Inception modules.
        self.inp1 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden*2, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden*2),
            nn.Conv2d(
                in_channels=hidden*2, out_channels=hidden*2, kernel_size=(3, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden*2),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden*2, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden*2),
            nn.Conv2d(
                in_channels=hidden*2, out_channels=hidden*2, kernel_size=(5, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden*2),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden*2, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(hidden*2),
        )

        # transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cat_head = nn.Linear(d_model, 3)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
    
        x = self.transformer_encoder(x)
        # mean pool
        x = torch.mean(x, dim=1)

        logits = self.cat_head(x)
        return logits
    