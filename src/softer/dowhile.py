import torch
import torch.nn as nn
import numpy as np

class DoWhile(nn.Module):
    """
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, condition_fn, fn, x=0, i=0):
        """
        This is a tricky one, possibly the trickiest in the library.
        Fully differentiable while loop, assuming stop converges to 1.0 (see https://arxiv.org/abs/2110.05651)

        x is a buffer storing intermediate results, 
        i is a counter for number of iterations,
        condition_fn is a function of x and i, returning a stopping probability,
        and fn is a function of i and x, to update x every iteration

        See `num_divisors` in examples as a good reference for how to use this.
        """

        x = torch.tensor(x, dtype=torch.float64)
        i = torch.tensor(i, dtype=torch.float64)
        stop = condition_fn(i, x)
        ret = stop * x
        not_stop = 1.0

        one = torch.tensor(1.0, dtype=torch.float64)

        # Although technically not differentiable, once stop is 1, all derivatives after are 0
        while not torch.isclose(stop, one):
            # TODO: Allow for more advanced incrementing
            i += 1
            # Set to 0 when we stop, 1 until then
            not_stop *= (1.0 - stop)
            # Update x
            x = fn(i, x)
            # Stopping probability
            stop = condition_fn(i, x)
            # Update our return value with 0, unless we are exactly at a stop
            ret += not_stop * x * stop
        
        return ret
