import torch
import torch.nn as nn

class SoftAbs(nn.Module):
    """
    """

    def __init__(self, k=2) -> None:
        super().__init__()
        self.k = k

    def forward(self, x):
        return torch.log(torch.exp(self.k * x) + torch.exp(-self.k * x)) / self.k
