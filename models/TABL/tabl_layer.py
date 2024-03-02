import pytorch_lightning as pl
import torch
from torch import nn


class TABL_layer(pl.LightningModule):
    def __init__(self, d2, d1, t1, t2):
        super().__init__()
        self.t1 = t1

        weight = torch.Tensor(d2, d1)
        self.W1 = nn.Parameter(weight)
        nn.init.kaiming_uniform_(self.W1, nonlinearity='relu')

        weight2 = torch.Tensor(t1, t1)
        self.W = nn.Parameter(weight2)
        nn.init.constant_(self.W, 1 / t1)

        weight3 = torch.Tensor(t1, t2)
        self.W2 = nn.Parameter(weight3)
        nn.init.kaiming_uniform_(self.W2, nonlinearity='relu')

        bias1 = torch.Tensor(d2, t2)
        self.B = nn.Parameter(bias1)
        nn.init.constant_(self.B, 0)

        l = torch.Tensor(1, )
        self.l = nn.Parameter(l)
        nn.init.constant_(self.l, 0.5)

        self.activation = nn.ReLU()

    def forward(self, X):

        # maintaining the weight parameter between 0 and 1.
        if (self.l[0] < 0):
            l = torch.Tensor(1, )
            self.l = nn.Parameter(l)
            nn.init.constant_(self.l, 0.0)

        if (self.l[0] > 1):
            l = torch.Tensor(1, )
            self.l = nn.Parameter(l)
            nn.init.constant_(self.l, 1.0)

        # modelling the dependence along the first mode of X while keeping the temporal order intact (7)
        X = self.W1 @ X

        # enforcing constant (1) on the diagonal
        W = self.W - self.W * torch.eye(self.t1, dtype=torch.float32, device="cuda") + torch.eye(self.t1, dtype=torch.float32, device="cuda") / self.t1

        # attention, the aim of the second step is to learn how important the temporal instances are to each other (8)
        E = X @ W

        # computing the attention mask  (9)
        A = torch.softmax(E, dim=-1)

        # applying a soft attention mechanism  (10)
        # he attention mask A obtained from the third step is used to zero out the effect of unimportant elements
        X = self.l[0] * (X) + (1.0 - self.l[0]) * X * A

        # the final step of the proposed layer estimates the temporal mapping W2, after the bias shift (11)
        y = X @ self.W2 + self.B
        return y