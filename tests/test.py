import torch
import matplotlib.pyplot as plt 
from tqdm import tqdm
# from softmatrix import SoftMatrix
from softer.arithmetic.softdivides import SoftDivides

f = SoftDivides(k=32)

t = torch.linspace(-5, 5, 100)

plt.plot(t, f(t, 3))
plt.show()
