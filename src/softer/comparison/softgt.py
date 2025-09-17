import torch
import torch.nn as nn

class SoftGt(nn.Module):
    """
    """

    def __init__(self, k=2) -> None:
        super().__init__()
        self.k = k

    