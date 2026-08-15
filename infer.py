from pathlib import Path
import argparse
import numpy as np
import torch
from PIL import Image
import cv2

from model import create_model


def load_input(input_path: Path):
    """
    Load either a .npy file or a regular image file.
    Returns a normalized float32 numpy array in [0,1].
    """
    if input_path.suffix.lower() == ".npy":
        image = np.load(input_path).astype(np.float32)
    else:
        img = Image.open(input_path).convert("L")  # grayscale
        # Resize PNG input to 128x128 matching model's trained degraded input resolution
        img = img.resize((128, 128), Image.BICUBIC)
        image = np.array(img).astype(np.float32) / 255.0

    return image


def save_png(array, path, enhance=False):
    """
    Save a numpy array as an 8-bit grayscale PNG with optional contrast & sharpening enhancement.
    """
    array = np.squeeze(array)
    array = np.clip(array, 0.0, 1.0)
    
    if enhance:
        p2, p98 = np.percentile(array, 1), np.percentile(array, 99)
        array = np.clip((array - p2) / (p98 - p2 + 1e-5), 0.0, 1.0)
        uint8_arr = (array * 255).astype(np.uint8)
        blur = cv2.GaussianBlur(uint8_arr, (0, 0), 2.0)
        uint8_arr = cv2.addWeighted(uint8_arr, 1.5, blur, -0.5, 0)
    else:
        uint8_arr = (array * 255).astype(np.uint8)

    Image.fromarray(uint8_arr, mode="L").save(path)


def main():
    parser = argparse.ArgumentParser(
        description="Run SwinIR restoration on a single image or .npy file"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image (.png/.jpg/.tif/.bmp) or .npy file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where output images will be saved"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pth",
        help="Path to trained model checkpoint"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = create_model().to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # --------------------------------------------------
    # Load input
    # --------------------------------------------------
    input_path = Path(args.input)

    image = load_input(input_path)

    tensor = torch.from_numpy(image).float()

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)      # C,H,W

    tensor = tensor.unsqueeze(0).to(device)   # B,C,H,W

    # --------------------------------------------------
    # Inference with window-size padding
    # --------------------------------------------------
    h, w = image.shape
    window_size = 8
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        tensor_padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        tensor_padded = tensor

    with torch.inference_mode():
        restored_padded = model(tensor_padded)

    # Crop back to 2x original dimensions
    restored = restored_padded[:, :, :h * 2, :w * 2]
    restored_np = restored.squeeze().cpu().numpy()

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_png = output_dir / f"{input_path.stem}_input.png"
    output_png = output_dir / f"{input_path.stem}_restored.png"

    is_image_file = input_path.suffix.lower() != ".npy"

    save_png(image, input_png, enhance=False)
    save_png(restored_np, output_png, enhance=is_image_file)

    print(f"Input image ({w}x{h}) saved : {input_png}")
    print(f"Restored 2x output ({w*2}x{h*2}) saved: {output_png}")


if __name__ == "__main__":
    main()