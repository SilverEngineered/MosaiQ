from matplotlib import pyplot as plt
import numpy as np
z = np.load('upper_bounds_9.npy')
x = np.load('gen_loss.npy')
y = np.load('disc_loss.npy')


plt.plot(z)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Normalized Generator and Discriminator Loss')
plt.legend()
plt.show()