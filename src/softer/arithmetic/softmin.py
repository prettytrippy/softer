import torch
import torch.nn as nn

class SoftMin(nn.Module):
    """
    """

    def __init__(self, k=2) -> None:
        super().__init__()
        self.k = k

    def forward(self, x):
        return torch.logsumexp(-self.k*x, dim=-1) / -self.k