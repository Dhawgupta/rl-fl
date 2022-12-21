import numpy as np
rl = np.load('rl_array.npy')
ra = np.load('random_array.npy')

import matplotlib.pyplot as plt
plt.plot(rl, label = 'rl')
plt.plot(ra, label = 'random')
plt.legend()
plt.savefig('save.png')
