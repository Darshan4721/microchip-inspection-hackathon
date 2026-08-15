import numpy as np

img = np.load("results/000000_restored.npy")
print(img.shape, img.min(), img.max())