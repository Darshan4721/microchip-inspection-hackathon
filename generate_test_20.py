import cv2
import numpy as np
from PIL import Image
from pathlib import Path

batch_dir = Path("test_batch_20")
batch_dir.mkdir(exist_ok=True)

# Copy base images from input_custom
src_files = [
    Path("input_custom/000005.npy"),
    Path("input_custom/000150.npy"),
    Path("input_custom/000300.npy"),
    Path("input_custom/real_semicon_die.npy"),
    Path("input_custom/real_semicon_die.png"),
    Path("input_custom/degraded_semicon_gaussian.png"),
    Path("input_custom/degraded_semicon_lowres.png"),
    Path("input_custom/degraded_semicon_speckle.png"),
    Path("input_custom/semicon_test_pattern.png"),
]

count = 0
for f in src_files:
    if f.exists():
        dest = batch_dir / f"sample_{count:02d}{f.suffix}"
        dest.write_bytes(f.read_bytes())
        count += 1

# Generate additional diverse degraded images from the base GT patterns
gt_img = Image.open("input_custom/new_semicon_clean_gt.png").convert("L")
gt_arr = np.array(gt_img).astype(np.float32) / 255.0

np.random.seed(42)
while count < 20:
    idx = count - 9
    # Downsample by 2x to 128x128
    h, w = gt_arr.shape
    lr = cv2.resize(gt_arr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    
    # Apply diverse degradation profiles
    if idx % 4 == 0:
        # Heavy speckle noise (multiplicative)
        noise = np.random.gamma(shape=10.0, scale=0.1, size=lr.shape)
        degraded = lr * noise
    elif idx % 4 == 1:
        # Gaussian blur + Gaussian noise
        blurred = cv2.GaussianBlur(lr, (5, 5), sigmaX=1.5)
        noise = np.random.normal(0, 0.05, size=lr.shape)
        degraded = blurred + noise
    elif idx % 4 == 2:
        # Severe mixed degradation: blur + speckle + extreme values
        blurred = cv2.GaussianBlur(lr, (3, 3), sigmaX=1.0)
        noise = np.random.gamma(shape=8.0, scale=0.125, size=lr.shape)
        degraded = blurred * noise + np.random.normal(0, 0.03, size=lr.shape)
    else:
        # Salt-and-pepper / impulse noise
        degraded = lr.copy()
        mask = np.random.rand(*lr.shape)
        degraded[mask < 0.03] = 0.0
        degraded[mask > 0.97] = 1.4  # out-of-range speckle
        
    out_file = batch_dir / f"sample_{count:02d}.png"
    clipped = np.clip(degraded, 0.0, 1.0)
    uint8_img = (clipped * 255.0).astype(np.uint8)
    Image.fromarray(uint8_img, mode="L").save(out_file)
    count += 1

print(f"Prepared exactly {count} distinct test images in {batch_dir}")
