import torch
import torch.nn as nn

class SoftEq(nn.Module):
    """
    """

    def __init__(self, k=2) -> None:
        super().__init__()
        self.k = k

    def forward(self, x, y):
        return torch.exp(-self.k * (x - y)**2)