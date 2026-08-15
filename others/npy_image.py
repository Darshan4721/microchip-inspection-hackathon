import numpy as np 
from PIL import Image
arr = np.load("./Train/train/NoisyLR/000000.npy")
arr = (arr * 255).astype(np.uint8)
img = Image.fromarray(arr)


img.save("./npy_image_outputs/000000_lr.png")
