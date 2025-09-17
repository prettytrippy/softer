import torch
import matplotlib.pyplot as plt 
from tqdm import tqdm
# from softmatrix import SoftMatrix
from softer.softstep import SoftStep
from softer.arithmetic.softabs import SoftAbs

f = SoftAbs(k=2)

t = torch.linspace(-5, 5, 100)

plt.plot(f(t))
plt.show()
