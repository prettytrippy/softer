import torch
import matplotlib.pyplot as plt 
from tqdm import tqdm
# from softmatrix import SoftMatrix
from softer.softstep import SoftStep
from softer.arithmetic.softmin import SoftMin

f = SoftMin(k=32)

t = torch.linspace(-5, 5, 100)

print(f(t))
# plt.plot(f(t))
# plt.show()
