import torch
import torch.nn as nn
import numpy as np
from softer.softstep import SoftStep

class SoftBoxcar(nn.Module):
    """
    """

    def __init__(self, k=2) -> None:
        super().__init__()
        self.step = SoftStep(k)

    def forward(self, x, c):
        return self.step(x + c) - self.step(x - c)
        

