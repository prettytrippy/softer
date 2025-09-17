import torch
import torch.nn as nn

class SoftXor(nn.Module):
    """
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return (x + y - x * y)(1 - x * y)