import torch
import torch.nn as nn
import numpy as np

class SoftMod(nn.Module):
    """
    """

    def __init__(self, k=2) -> None:
        super().__init__()
        self.k = k

    def forward(self, x, y):
        x_prime = np.pi * x / y
        sine_x = torch.sin(x_prime)

        numerator = sine_x * torch.cos(x_prime)
        denominator = sine_x * sine_x + np.exp(-self.k)

        atan_term = torch.atan(numerator / denominator)
        return y * (0.5 - atan_term / np.pi)

