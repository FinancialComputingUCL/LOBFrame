import pytorch_lightning as pl
import torch
import torch.nn as nn


class Complete_HCNN(pl.LightningModule):
    def __init__(self, lighten, homological_structures):
        super().__init__()
        self.name = "hcnn"
        if lighten:
            self.name += "-lighten"

        self.homological_structures = homological_structures
        self.tetrahedra = self.homological_structures['tetrahedra']
        self.triangles = self.homological_structures['triangles']
        self.edges = self.homological_structures['edges']

        # ------------ #

        self.conv1_tetrahedra = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.ReLU(),
        )

        self.conv1_triangles = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.ReLU(),
        )

        self.conv1_edges = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.ReLU(),
        )

        # ------------ #

        self.conv2_tetrahedra = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 4), stride=(1, 4)
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.ReLU(),
        )

        self.conv2_triangles = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 3)
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.ReLU(),
        )

        self.conv2_edges = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.ReLU(),
        )

        # ------------ #

        self.conv3_tetrahedra = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, int(len(self.tetrahedra) / 8))
            ),
            nn.Dropout(0.35),
            nn.ReLU(),
        )

        self.conv3_triangles = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, int(len(self.triangles) / 6))
            ),
            nn.Dropout(0.35),
            nn.ReLU(),
        )

        self.conv3_edges = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, int(len(self.edges) / 4))
            ),
            nn.Dropout(0.35),
            nn.ReLU(),
        )

        # ------------ #

        self.lstm = nn.LSTM(
            input_size=96, hidden_size=32, num_layers=1, batch_first=True
        )
        self.fc1 = nn.Linear(32, 3)

    def forward(self, x):
        x_tetrahedra = x[:, :, :, self.tetrahedra]
        x_triangles = x[:, :, :, self.triangles]
        x_edges = x[:, :, :, self.edges]

        x_tetrahedra = self.conv1_tetrahedra(x_tetrahedra)
        x_triangles = self.conv1_triangles(x_triangles)
        x_edges = self.conv1_edges(x_edges)

        x_tetrahedra = self.conv2_tetrahedra(x_tetrahedra)
        x_triangles = self.conv2_triangles(x_triangles)
        x_edges = self.conv2_edges(x_edges)

        x_tetrahedra = self.conv3_tetrahedra(x_tetrahedra)
        x_triangles = self.conv3_triangles(x_triangles)
        x_edges = self.conv3_edges(x_edges)

        x = torch.cat((x_tetrahedra, x_triangles, x_edges), dim=1)

        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        logits = self.fc1(x)

        return logits
