from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import create_model

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = ROOT / "checkpoints" / "best.pth"
INPUT_IMAGE = ROOT / "input.png"          # change this to your image
OUTPUT_IMAGE = ROOT / "output_restored.png"

# -----------------------------
# Device
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load model
# -----------------------------
model = create_model().to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# If your checkpoint was saved as {"model": state_dict, ...}
if isinstance(checkpoint, dict) and "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()

# -----------------------------
# Load image or .npy
# -----------------------------
if INPUT_IMAGE.suffix.lower() == ".npy":
    img_np = np.load(INPUT_IMAGE).astype(np.float32)
else:
    img = Image.open(INPUT_IMAGE).convert("L")   # grayscale
    img_np = np.array(img).astype(np.float32) / 255.0

# H,W -> 1,1,H,W
tensor = (
    torch.from_numpy(img_np)
    .unsqueeze(0)      # channel
    .unsqueeze(0)      # batch
    .to(DEVICE)
)

# -----------------------------
# Inference (2x Super-Resolution & Denoising)
# -----------------------------
with torch.inference_mode():
    output = model(tensor)

# -----------------------------
# Save restored output image (256x256)
# -----------------------------
output = output.squeeze().cpu().numpy()
output = np.clip(output, 0.0, 1.0)
output = (output * 255).astype(np.uint8)

Image.fromarray(output).save(OUTPUT_IMAGE)

print(f"Input : {INPUT_IMAGE}")
print(f"Output: {OUTPUT_IMAGE}")