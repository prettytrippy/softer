import torch
import torch.nn as nn

class SoftLt(nn.Module):
    """
    """

    def __init__(self, k=2) -> None:
        super().__init__()
        self.k = k

    def forward(self, x, y):
        return torch.sigmoid(self.k * (y - x))