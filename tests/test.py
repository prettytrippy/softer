import torch
import matplotlib.pyplot as plt 
from tqdm import tqdm
# from softmatrix import SoftMatrix
from softer.comparison.softeq import SoftEq

f = SoftEq(k=32)

t = torch.linspace(-2, 2, 100)

plt.plot(t, f(t, 1))
plt.show()
