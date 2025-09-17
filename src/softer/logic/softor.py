import torch
import torch.nn as nn

class SoftOr(nn.Module):
    """
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return x + y - x * y