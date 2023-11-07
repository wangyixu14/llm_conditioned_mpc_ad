import numpy as np
from matplotlib import pyplot as plt

v = np.load("velocity_trace.npy")
plt.plot(v[:, 0], label='vx')
plt.plot(v[:, 1], label='vy')
plt.savefig('velocity.png')