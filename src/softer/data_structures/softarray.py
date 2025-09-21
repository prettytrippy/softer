import torch
import torch.nn as nn
from softer.softboxcar import SoftBoxcar

class SoftArray(nn.Module):
    """
    Soft, differentiable array of length n where each element has payload shape `shape`.
    Axis 0 is the index axis; `shape` describes the per-item payload (may be empty for scalars).
    """

    def __init__(
        self,
        n: int,
        *,
        shape: tuple = (),
        data=None,
        k: float = 2.0,
        dtype: torch.dtype = torch.float64,
        device=None,
        learnable: bool = False,
    ):
        super().__init__()
        if not isinstance(shape, tuple):
            raise TypeError("`shape` must be a tuple")
        self.n = int(n)
        self.shape = shape
        self.k = float(k)

        self.boxcar = SoftBoxcar(k=k)

        if data is None:
            tensor = torch.zeros((self.n, *self.shape), dtype=dtype, device=device)
        else:
            tensor = torch.as_tensor(data, dtype=dtype, device=device)
            if tensor.shape != (self.n, *self.shape):
                raise ValueError(
                    f"`data` must have shape {(self.n, *self.shape)}, got {tuple(tensor.shape)}"
                )

        if learnable:
            self.data = nn.Parameter(tensor)
        else:
            self.register_buffer("data", tensor)

    def _onehot(self, m):
        """
        Pseudo-Gaussian weights over index axis (n,). Supports fractional m.
        """
        x = torch.arange(self.n, dtype=self.data.dtype, device=self.data.device)
        w = self.boxcar(x - m, 0.5)
        w = w / (w.sum() + torch.finfo(w.dtype).eps)
        return w  # (n,)

    def _wview(self, w):
        return w.view(self.n, *([1] * len(self.shape)))

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"SoftArray with {self.n} elements of shape {self.shape} and smoothing value {self.k}\n{self.data})"

    def forward(self, x):
        # What to do?
        return self.data

    def __getitem__(self, index):
        """
        Soft read at `index` (can be float). Returns a tensor of shape `shape`.
        """
        w = self._onehot(index)                        # (n,)
        wv = self._wview(w)                            # (n, *1s)
        return (wv * self.data).sum(dim=0)             # (*shape,)

    def __setitem__(self, index, value):
        """
        Soft write: blends current data toward `value` at `index`.
        Accepts any `value` broadcastable to `shape` (including scalar).
        """
        v = torch.as_tensor(value, dtype=self.data.dtype, device=self.data.device)

        try:
            v_b = torch.broadcast_to(v, self.shape) if v.shape != self.shape else v
        except RuntimeError as e:
            raise ValueError(
                f"value with shape {tuple(v.shape)} is not broadcastable to {self.shape}"
            ) from e

        vals = v_b.unsqueeze(0).expand(self.n, *self.shape)  # (n, *shape)
        w = self._onehot(index)
        wv = self._wview(w)                                   # (n, *1s)

        new_data = self.data + wv * (vals - self.data)        # (n, *shape)

        if isinstance(self.data, nn.Parameter):
            self.data.data = new_data
        else:
            self.data = new_data

    def __iter__(self):
        for i in range(self.n):
            yield self[i]

    def __reversed__(self):
        for i in range(self.n - 1, -1, -1):
            yield self[i]
