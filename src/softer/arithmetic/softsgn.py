import torch
import torch.nn as nn

class SoftSgn(nn.Module):
    """
    """

    def __init__(self, k=2) -> None:
        super().__init__()
        self.k = k

    def forward(self, x):
        numerator = 1 - torch.exp(-self.k * x)
        denominator = 1 + torch.exp(-self.k * x)
        return numerator / denominator
