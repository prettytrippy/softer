import torch
import matplotlib.pyplot as plt 
from tqdm import tqdm
# from softmatrix import SoftMatrix
from softer.arithmetic.softdivides import SoftDivides
from softer.softboxcar import SoftBoxcar

f = SoftBoxcar(k=32)

t = torch.linspace(-5, 5, 100)

plt.plot(t, f(t, 1))
plt.show()
