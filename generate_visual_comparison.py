import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def create_comparison_grid(input_path, restored_path, gt_path, output_path, title, metrics_text=""):
    # Load images
    inp = Image.open(input_path).convert("L").resize((256, 256), Image.NEAREST)
    bicubic = Image.open(input_path).convert("L").resize((256, 256), Image.BICUBIC)
    restored = Image.open(restored_path).convert("L")
    
    has_gt = gt_path and Path(gt_path).exists()
    gt = Image.open(gt_path).convert("L") if has_gt else None
    
    n_cols = 4 if has_gt else 3
    panel_w, panel_h = 256, 256
    header_h = 40
    footer_h = 35
    total_w = n_cols * panel_w + (n_cols + 1) * 10
    total_h = panel_h + header_h + footer_h + 30
    
    grid = Image.new("L", (total_w, total_h), color=255)
    draw = ImageDraw.Draw(grid)
    
    images = [
        ("Degraded Input (128x128)", inp),
        ("Bicubic 2x Baseline", bicubic),
        ("SwinIR 2x Restored", restored)
    ]
    if has_gt:
        images.append(("Clean Ground Truth", gt))
        
    for i, (label, img) in enumerate(images):
        x = 10 + i * (panel_w + 10)
        y = header_h + 10
        grid.paste(img, (x, y))
        draw.text((x + 10, y + 10), label, fill=255)
        draw.text((x + 10, y + panel_h - 20), f"{img.size[0]}x{img.size[1]}", fill=255)

    # Title & metrics footer
    draw.text((15, 10), title, fill=0)
    if metrics_text:
        draw.text((15, total_h - 25), metrics_text, fill=0)
        
    grid.save(output_path)
    print(f"Saved visual comparison to: {output_path}")

# Generate comparisons
Path("results/comparisons").mkdir(parents=True, exist_ok=True)

create_comparison_grid(
    "input_custom/real_semicon_die.png",
    "results/real_semicon_die_restored.png",
    "input_custom/real_semicon_die_gt.png",
    "results/comparisons/comparison_real_die.png",
    "Real Semiconductor Die Inspection: Degraded vs Bicubic vs SwinIR vs Ground Truth",
    "Bicubic: 15.45 dB / 0.5781 SSIM  |  SwinIR: 15.37 dB / 0.6335 SSIM (Delta SSIM: +0.0554)  |  Time: 47ms"
)

create_comparison_grid(
    "input_custom/degraded_semicon_speckle.png",
    "results/degraded_semicon_speckle_restored.png",
    "input_custom/new_semicon_clean_gt.png",
    "results/comparisons/comparison_speckle_wafer.png",
    "Speckle Noise Degradation: Degraded vs Bicubic vs SwinIR vs Ground Truth",
    "Bicubic: 15.93 dB / 0.6048 SSIM  |  SwinIR: 16.18 dB / 0.6901 SSIM (Delta: +0.26 dB, +0.0853 SSIM)  |  Time: 46ms"
)

create_comparison_grid(
    "input_custom/semicon_test_pattern.png",
    "results/semicon_test_pattern_restored.png",
    None,
    "results/comparisons/comparison_test_pattern.png",
    "Microchip Resolution Target: Degraded vs Bicubic vs SwinIR",
    "Output resolution: 256x256 (2x Super-Resolution)  |  Inference latency: 43ms (GPU)"
)
