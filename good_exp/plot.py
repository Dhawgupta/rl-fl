import numpy as np
rl = np.load('rl_array.npy')
ra = np.load('random_array.npy')

import matplotlib.pyplot as plt
plt.plot(rl[:100], label = 'rl', linewidth = 1, color= 'red')
plt.plot(ra[:100], label = 'random', linewidth=1, color = 'blue')
plt.xlabel('Round Number')
plt.ylabel('Validation Accuracy')
plt.title('Performance RL vs Random in FL Training')

plt.legend()
plt.savefig('save.png', dpi=600)
