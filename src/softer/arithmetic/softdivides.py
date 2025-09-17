import torch
import torch.nn as nn
from softer.arithmetic.softfloor import SoftFloor
from softer.arithmetic.softmod import SoftMod
from softer.comparison.softeq import SoftEq

class SoftDivides(nn.Module):
    """
    """

    def __init__(self, k=2) -> None:
        super().__init__()
        self.mod = SoftMod(k)
        self.eq = SoftEq(k)
        self.floor = SoftFloor(k)

    def forward(self, x, y):
        return self.eq(0.0, self.floor(self.mod(x + 0.5, y)))