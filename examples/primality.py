import torch
import numpy as np

from softer.arithmetic.softdivides import SoftDivides
from softer.arithmetic.softround import SoftRound
from softer.softstep import SoftStep
from softer.ifelse import IfElse
from softer.dowhile import DoWhile
from softer.comparison.softeq import SoftEq
from softer.comparison.softgt import SoftGt

k = 3

def num_divisors(n):
    """
    Return the number of unique divisors of n, between 1 and n (exclusive).
    """

    eq = SoftEq(k=k)
    gt = SoftGt(k=k)
    ifelse = IfElse()
    dowhile = DoWhile()
    divides = SoftDivides(k=k)

    def condition(i, x):
        # Stop when we hit half of n
        return gt(i, n / 2)

    def func(i, x):
        # If i is a divisor of n, add 1 to x
        return x + divides(n, i)

    # Pretty clunky, but 2 and 3 misbehave, given the stopping criterion.
    return ifelse(eq(n,torch.tensor(2.0)), 0, 
           ifelse(eq(n,torch.tensor(3.0)), 0, 
           dowhile(condition, func, x=torch.tensor(0), i=1)))

def is_prime(n):
    step = SoftStep(k=k)
    divisors = num_divisors(n)
    return 1 - step(divisors - 0.5)

if __name__ == "__main__":
    for i in range(2, 20):
        print(i, is_prime(i))