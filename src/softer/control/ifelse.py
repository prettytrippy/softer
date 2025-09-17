import torch
import torch.nn as nn

class IfElse(nn.Module):
    """
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, condition, consequent, alternative):
        return (condition * consequent) + ((1 - condition) * alternative)
