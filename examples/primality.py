import torch

from softer.arithmetic.softdivides import SoftDivides
from softer.softstep import SoftStep
from softer.ifelse import IfElse
from softer.comparison.softeq import SoftEq
from softer.arithmetic.softround import SoftRound

MAX_NUM = int(1e2)
k = 1

def num_divisors(n):
    divides = SoftDivides(k=k)
    eq = SoftEq(k=k)
    ifelse = IfElse()

    x = torch.arange(2, MAX_NUM, dtype=torch.float64)
    divisors = ifelse(eq(n, x), 0, divides(n, x))
    
    return torch.sum(divisors)

def is_prime(n):
    step = SoftStep(k=k)
    divisors = num_divisors(n)
    return 1 - step(divisors - 0.5)

def find_closest_prime(x):
    round = SoftRound()
    x = torch.nn.Parameter(torch.tensor(x, dtype=torch.float64))

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD([x], lr=0.001, momentum=0.9)

    for i in range(1000):
        optimizer.zero_grad()

        y = is_prime(x)

        loss = loss_fn(y, torch.tensor(1.0, dtype=torch.float64))
        loss.backward()
        optimizer.step()

    return round(x).item()

for i in range(2, 100):
    print(find_closest_prime(i))