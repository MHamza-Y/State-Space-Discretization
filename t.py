import matplotlib.pyplot as plt
import numpy as np

epsilon = np.linspace(500, 1, 500)
epsilon = np.logspace(np.log(0.9), np.log(0.01), 1000, base=np.exp(1))
plt.plot(epsilon)
plt.show()
