import numpy as np
rl = np.load('rl_array.npy')
ra = np.load('random_array.npy')

import matplotlib.pyplot as plt
def smoothen(arr, alpha =0.5):
    smooth = np.zeros_like(arr)
    smooth[0] = arr[0]
    
    for i in range(1, len(arr)):
        smooth[i] = alpha * smooth[i-1] + (1-alpha) * arr[i]
    return smooth
plt.plot(smoothen(rl[:15]), label = 'rl', linewidth = 2, color= 'green')
plt.plot(smoothen(ra[:15]), label = 'random', linewidth=2, color = 'blue')
plt.xlabel('Round ssNumber')
plt.ylabel('Validation Accuracy')
plt.title('Performance RL vs Random in FL Training')

plt.legend()
plt.savefig('save3.png', dpi=600)
plt.savefig('save3.pdf', dpi=600)
