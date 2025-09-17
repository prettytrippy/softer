import torch
import torch.nn as nn

class SoftArray(nn.Module):
    """
    SoftArray is an array-like module with fully differentiable operations.
    All standard functionality (iteration, length) is implemented.
    Getting and setting values, usually a non-differentiable operation,
    is made possible by approximating one-hot vectors with Gaussians.
    
    TODO:
    - Aggregate functions (max, mean,...)
    - Arithmetic with other tensors
    - Error handling and logging
    """

    def __init__(self, n, d=1, data=None, k=8) -> None:
        super().__init__()

        if data is None:
            data = torch.zeros((n, d), dtype=torch.float64)

        self.data = nn.parameter.Buffer(data)
        self.n = n
        self.k = k

    def onehot(self, m):
        """
        Generates a pseudo-Gaussian, centered at m.
        """
        x = torch.arange(self.n, dtype=self.data.dtype)
        return 1.0 / torch.cosh(self.k * (x - m))

    def forward(self, x):
        # It's not clear what to do here.
        # Should softarray(x) be softarray[x]?
        return self.data

    def __repr__(self):
        return str(self.data)

    def __getitem__(self, index):
        indexer = self.onehot(index)
        return indexer @ self.data

    def __setitem__(self, index, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        vals = value.repeat(self.n, 1)
        onehot = self.onehot(index).unsqueeze(1)

        self.data += onehot * (vals - self.data)

        # self.data -= onehot * self.data
        # self.data += onehot * vals

    def __len__(self):
        return self.n

    def __iter__(self):
        for idx in range(self.n):
            yield self[idx]

    def __reversed__(self):
        for idx in range(self.n-1, -1, -1):
            yield self[idx]
