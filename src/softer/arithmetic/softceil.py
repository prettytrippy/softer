from softer.arithmetic.softmod import SoftMod
import torch
import torch.nn as nn

class SoftCeil(nn.Module):
    """
    """

    def __init__(self, k=2) -> None:
        super().__init__()
        self.softmod = SoftMod(k)

    def forward(self, x):
        return x - self.softmod(x + 1.0, 1.0) 