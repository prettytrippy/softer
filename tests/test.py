import torch
import matplotlib.pyplot as plt 
from tqdm import tqdm
# from softmatrix import SoftMatrix
from softer.arithmetic.softround import SoftRound

f = SoftRound(k=32)

t = torch.linspace(-2, 2, 100)

plt.plot(t, f(t))
plt.show()
