import torch
import torch.nn as nn
import numpy as np

class SoftStep(nn.Module):
    """
    """

    def __init__(self, k=2) -> None:
        super().__init__()
        self.k = k

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.k * x))
        

