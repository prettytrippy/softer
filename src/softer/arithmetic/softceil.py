from softer.arithmetic.softfloor import SoftFloor
import torch
import torch.nn as nn

class SoftRound(nn.Module):
    """
    """

    def __init__(self, k=2) -> None:
        super().__init__()
        self.softfloor = SoftFloor(k)

    def forward(self, x):
        return self.softfloor(x + 1.0)