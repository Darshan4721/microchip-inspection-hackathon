import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

ROOT = Path(".")

# 4 Diverse Test Images
cases = [
    {
        "name": "Sample 1: Real Semiconductor Die",
        "input": ROOT / "input_custom/real_semicon_die.png",
        "output": ROOT / "results/real_semicon_die_restored.png",
        "gt": ROOT / "input_custom/real_semicon_die_gt.png",
        "desc": "Real microscopic die inspection (PSNR: 15.37 dB | SSIM: 0.6335, +0.055 over Bicubic)"
    },
    {
        "name": "Sample 2: Wafer with Speckle Noise",
        "input": ROOT / "input_custom/degraded_semicon_speckle.png",
        "output": ROOT / "results/degraded_semicon_speckle_restored.png",
        "gt": ROOT / "input_custom/new_semicon_clean_gt.png",
        "desc": "Laser speckle noise removal (PSNR: 16.18 dB | SSIM: 0.6901, +0.085 over Bicubic)"
    },
    {
        "name": "Sample 3: Wafer with Optical Blur & Haze",
        "input": ROOT / "input_custom/degraded_semicon_gaussian.png",
        "output": ROOT / "results/degraded_semicon_gaussian_restored.png",
        "gt": ROOT / "input_custom/new_semicon_clean_gt.png",
        "desc": "Gaussian optical haze & defocus (PSNR: 14.52 dB | SSIM: 0.4815, +0.060 over Bicubic)"
    },
    {
        "name": "Sample 4: Resolution Gratings & Targets",
        "input": ROOT / "input_custom/semicon_test_pattern.png",
        "output": ROOT / "results/semicon_test_pattern_restored.png",
        "gt": None,
        "desc": "Microchip resolution pattern (2x Super-Resolution to 256x256, latency 43ms)"
    }
]

# Grid layout: 4 rows, 3 columns (Input, SwinIR Restored, Ground Truth)
panel_size = 256
pad = 12
header_h = 32
desc_h = 24
row_h = panel_size + header_h + desc_h + pad

col_w = panel_size
total_w = 3 * col_w + 4 * pad
total_h = 60 + 4 * row_h + 20

grid_img = Image.new("L", (total_w, total_h), color=255)
draw = ImageDraw.Draw(grid_img)

# Main Title
draw.text((pad, 15), "KLA Semiconductor Image Restoration: 2x Super-Resolution & Denoising Benchmark", fill=0)
draw.text((pad, 35), "Architecture: SwinIR (2.45M params) | GPU Latency: 45ms (RTX 5060) | CPU Latency: 1.86s", fill=80)

# Column Headers
col_names = ["1. Degraded Input (128x128)", "2. SwinIR Restored (256x256)", "3. Ground Truth (256x256)"]

y_start = 65
for row_idx, case in enumerate(cases):
    row_y = y_start + row_idx * row_h
    
    # Case Header
    draw.text((pad, row_y), case["name"], fill=0)
    draw.text((pad + 320, row_y), case["desc"], fill=60)
    
    # Images
    inp_img = Image.open(case["input"]).convert("L").resize((panel_size, panel_size), Image.NEAREST)
    out_img = Image.open(case["output"]).convert("L")
    if out_img.size != (panel_size, panel_size):
        out_img = out_img.resize((panel_size, panel_size), Image.BICUBIC)
        
    if case["gt"] and case["gt"].exists():
        gt_img = Image.open(case["gt"]).convert("L").resize((panel_size, panel_size), Image.BICUBIC)
    else:
        # Placeholder for test set without GT
        gt_img = Image.new("L", (panel_size, panel_size), color=230)
        d_gt = ImageDraw.Draw(gt_img)
        d_gt.text((40, 120), "[Test Set Target - No GT]", fill=100)

    row_imgs = [inp_img, out_img, gt_img]
    for col_idx, img in enumerate(row_imgs):
        x = pad + col_idx * (col_w + pad)
        y = row_y + header_h
        grid_img.paste(img, (x, y))
        
        # Border
        draw.rectangle([x, y, x + panel_size, y + panel_size], outline=180)
        if row_idx == 0:
            draw.text((x + 8, y + 8), col_names[col_idx], fill=255)

# Save project root comparison_grid.png
grid_path = ROOT / "comparison_grid.png"
grid_img.save(grid_path)
print(f"Saved main pitch visual: {grid_path.resolve()}")

# Also save 3 clean individual pairs for pitch presentation
pairs = [
    ("pair_real_die.png", cases[0]["input"], cases[0]["output"], "Real Die: Degraded (128x128) vs SwinIR 2x Restored (256x256)"),
    ("pair_speckle_wafer.png", cases[1]["input"], cases[1]["output"], "Speckle Noise: Degraded (128x128) vs SwinIR 2x Restored (256x256)"),
    ("pair_test_pattern.png", cases[3]["input"], cases[3]["output"], "Resolution Target: Degraded (128x128) vs SwinIR 2x Restored (256x256)")
]

for out_name, in_p, out_p, title in pairs:
    pair_w = panel_size * 2 + pad * 3
    pair_h = panel_size + 60
    p_img = Image.new("L", (pair_w, pair_h), color=255)
    p_draw = ImageDraw.Draw(p_img)
    p_draw.text((pad, 10), title, fill=0)
    
    img1 = Image.open(in_p).convert("L").resize((panel_size, panel_size), Image.NEAREST)
    img2 = Image.open(out_p).convert("L").resize((panel_size, panel_size), Image.BICUBIC)
    
    p_img.paste(img1, (pad, 40))
    p_img.paste(img2, (pad * 2 + panel_size, 40))
    p_draw.rectangle([pad, 40, pad + panel_size, 40 + panel_size], outline=180)
    p_draw.rectangle([pad * 2 + panel_size, 40, pad * 2 + panel_size * 2, 40 + panel_size], outline=180)
    
    p_draw.text((pad + 10, 48), "Degraded Input", fill=255)
    p_draw.text((pad * 2 + panel_size + 10, 48), "SwinIR 2x Restored", fill=255)
    
    p_save = ROOT / out_name
    p_img.save(p_save)
    print(f"Saved individual pair visual: {p_save.resolve()}")
