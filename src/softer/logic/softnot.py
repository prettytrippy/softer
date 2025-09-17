import torch
import torch.nn as nn

class SoftNot(nn.Module):
    """
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return 1 - x