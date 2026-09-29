#!/usr/bin/env python3
"""
demo_single.py - End-to-end single image restoration & evaluation pipeline for Semiconductor Inspection.
Handles joint denoising and 2x super-resolution with automatic ground-truth metric calculation.
"""

from pathlib import Path
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from model import create_model
from utils import calculate_psnr, calculate_ssim


def load_input_data(input_path: Path):
    """
    Loads input from either .npy or standard image file.
    Preserves raw speckle noise values for .npy (allowing intensities > 1.0).
    """
    if input_path.suffix.lower() == ".npy":
        array = np.load(input_path).astype(np.float32)
        # Ensure 2D
        if array.ndim == 3 and array.shape[0] == 1:
            array = array.squeeze(0)
        return array, "npy"
    else:
        img = Image.open(input_path).convert("L")
        # Ensure 128x128 degraded resolution if not already
        if img.size != (128, 128):
            img = img.resize((128, 128), Image.BICUBIC)
        array = np.array(img).astype(np.float32) / 255.0
        return array, "image"


def load_gt_data(gt_path: Path, target_shape=(256, 256)):
    """Loads ground truth array for metric evaluation."""
    if gt_path.suffix.lower() == ".npy":
        array = np.load(gt_path).astype(np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array.squeeze(0)
    else:
        img = Image.open(gt_path).convert("L")
        if img.size != (target_shape[1], target_shape[0]):
            img = img.resize((target_shape[1], target_shape[0]), Image.BICUBIC)
        array = np.array(img).astype(np.float32) / 255.0
    return array


def find_matching_gt(input_path: Path, explicit_gt: str = None):
    """Detects matching ground truth file only if a genuine match exists."""
    if explicit_gt:
        p = Path(explicit_gt)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified ground truth file does not exist: {p}")

    stem = input_path.stem
    parent = input_path.parent

    # Direct filename matching
    direct_candidates = [
        parent / f"{stem}_gt.png",
        parent / f"{stem}_gt.npy",
        parent / f"{stem}_clean_gt.png",
        ROOT / "Train" / "train" / "train" / "GT" / f"{stem}.npy",
    ]
    for cand in direct_candidates:
        if cand.exists():
            return cand

    # Specific authentic pairs in input_custom
    if "degraded_semicon" in stem:
        paired = parent / "new_semicon_clean_gt.png"
        if paired.exists():
            return paired
    elif "real_semicon_die" in stem:
        paired = parent / "real_semicon_die_gt.png"
        if paired.exists():
            return paired

    return None


def save_image_output(array: np.ndarray, path: Path):
    """Saves output array as normalized 8-bit PNG."""
    clipped = np.clip(array, 0.0, 1.0)
    uint8_img = (clipped * 255.0).round().astype(np.uint8)
    Image.fromarray(uint8_img, mode="L").save(path)


def run_restoration(
    input_path: str,
    output_path: str = None,
    gt_path: str = None,
    checkpoint: str = "model_files/best.pth",
    device: str = None
):
    inp_p = Path(input_path)
    if not inp_p.exists():
        raise FileNotFoundError(f"Input file not found: {inp_p}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_p = Path(checkpoint)
    if not ckpt_p.exists() and Path("checkpoints/best.pth").exists():
        ckpt_p = Path("checkpoints/best.pth")
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

    # Load model
    model = create_model().to(device)
    checkpoint_data = torch.load(ckpt_p, map_location=device)
    if isinstance(checkpoint_data, dict) and "model" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model"])
    else:
        model.load_state_dict(checkpoint_data)
    model.eval()

    # Load input
    image_arr, file_type = load_input_data(inp_p)
    h, w = image_arr.shape

    tensor = torch.from_numpy(image_arr).float()
    while tensor.ndim < 4:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    # Pad to window size multiple (window_size=8)
    window_size = 8
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        tensor_padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        tensor_padded = tensor

    # Run inference and time it accurately
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        restored_padded = model(tensor_padded)

    if device == "cuda":
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - start_time

    # Unpad back to exact 2x resolution
    restored = restored_padded[:, :, :h * 2, :w * 2]
    restored_np = restored.squeeze().cpu().numpy()

    # Output directory and naming
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = ROOT / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_p = out_dir / f"{inp_p.stem}_restored.png"

    # Save output
    if file_type == "npy":
        npy_out = out_p.with_suffix(".npy")
        np.save(npy_out, restored_np)
        png_out = out_p.with_suffix(".png")
        save_image_output(restored_np, png_out)
    else:
        save_image_output(restored_np, out_p)

    # Evaluate against GT if found
    matched_gt = find_matching_gt(inp_p, gt_path)
    psnr_val = None
    ssim_val = None
    bicubic_psnr = None
    bicubic_ssim = None

    if matched_gt:
        gt_arr = load_gt_data(matched_gt, target_shape=(h * 2, w * 2))
        gt_tensor = torch.from_numpy(gt_arr).float().to(device)

        # SwinIR metrics
        psnr_val = calculate_psnr(restored.squeeze(), gt_tensor)
        ssim_val = calculate_ssim(restored.squeeze(), gt_tensor)

        # Baseline Bicubic metrics for honest comparison
        if file_type == "npy":
            # Normalize npy temporarily to [0,1] for bicubic PIL comparison
            norm_in = np.clip(image_arr, 0.0, 1.0)
            pil_in = Image.fromarray((norm_in * 255.0).astype(np.uint8))
        else:
            pil_in = Image.fromarray((np.clip(image_arr, 0.0, 1.0) * 255.0).astype(np.uint8))

        bicubic_pil = pil_in.resize((w * 2, h * 2), Image.BICUBIC)
        bicubic_arr = np.array(bicubic_pil).astype(np.float32) / 255.0
        bicubic_tensor = torch.from_numpy(bicubic_arr).float().to(device)

        bicubic_psnr = calculate_psnr(bicubic_tensor, gt_tensor)
        bicubic_ssim = calculate_ssim(bicubic_tensor, gt_tensor)

    # Print Clean Console Report
    print("=" * 65)
    print(f"  SEMICONDUCTOR IMAGE RESTORATION (SwinIR 2x)")
    print("=" * 65)
    print(f" Input File       : {inp_p.name} ({w}x{h}) [{file_type}]")
    print(f" Output File      : {out_p.name} ({w*2}x{h*2})")
    print(f" Device Used      : {device.upper()}")
    print(f" Inference Time   : {inference_time:.4f} seconds ({inference_time*1000:.1f} ms)")
    print(f" Input Val Range  : [{image_arr.min():.4f}, {image_arr.max():.4f}]")
    print(f" Output Val Range : [{restored_np.min():.4f}, {restored_np.max():.4f}]")
    
    if matched_gt:
        print("-" * 65)
        print(f" Ground Truth Match : {matched_gt.name}")
        print(f" Bicubic Baseline   : PSNR = {bicubic_psnr:.2f} dB  |  SSIM = {bicubic_ssim:.4f}")
        print(f" SwinIR Model       : PSNR = {psnr_val:.2f} dB  |  SSIM = {ssim_val:.4f}")
        print(f" Model vs Baseline  : Delta PSNR = {psnr_val - bicubic_psnr:+.2f} dB | Delta SSIM = {ssim_val - bicubic_ssim:+.4f}")
    else:
        print("-" * 65)
        print(" Ground Truth Match : None found (saved restored image without metrics)")
    print("=" * 65 + "\n")

    return {
        "input": str(inp_p),
        "output": str(out_p),
        "inference_time_s": inference_time,
        "input_shape": (w, h),
        "output_shape": (w * 2, h * 2),
        "psnr": psnr_val,
        "ssim": ssim_val,
        "bicubic_psnr": bicubic_psnr,
        "bicubic_ssim": bicubic_ssim,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Restore a single degraded semiconductor inspection image using SwinIR"
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image (.npy or .png/.jpg)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save restored image (optional)")
    parser.add_argument("--gt", "-g", type=str, default=None, help="Path to ground truth image (optional)")
    parser.add_argument("--checkpoint", "-c", type=str, default="model_files/best.pth", help="Model weights (.pth)")
    parser.add_argument("--device", "-d", type=str, default=None, choices=["cuda", "cpu"], help="Compute device")

    args = parser.parse_args()
    run_restoration(
        input_path=args.input,
        output_path=args.output,
        gt_path=args.gt,
        checkpoint=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
