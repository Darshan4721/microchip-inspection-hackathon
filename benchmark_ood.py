import cv2
import numpy as np
from PIL import Image
from demo_single import run_restoration

ood_samples = [
    ("input_custom/real_semicon_die.png", "input_custom/real_semicon_die_gt.png", "Real Die Microscopy"),
    ("input_custom/degraded_semicon_speckle.png", "input_custom/new_semicon_clean_gt.png", "Synthetic Wafer (Speckle Noise)"),
    ("input_custom/degraded_semicon_gaussian.png", "input_custom/new_semicon_clean_gt.png", "Synthetic Wafer (Gaussian Haze)"),
    ("input_custom/degraded_semicon_lowres.png", "input_custom/new_semicon_clean_gt.png", "Synthetic Wafer (Pure Downsampling)"),
    ("input_custom/semicon_test_pattern.png", None, "Microchip Test Pattern (Gratings & Targets)")
]

results = []
for inp, gt, label in ood_samples:
    print(f"\n>>> RUNNING OOD SAMPLE: {label} ({inp})")
    res = run_restoration(input_path=inp, gt_path=gt)
    
    # Visual quality analysis
    out_img = cv2.imread(res["output"], cv2.IMREAD_GRAYSCALE)
    inp_img = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)
    
    lap_inp = cv2.Laplacian(inp_img, cv2.CV_64F).var() if inp_img is not None else 0.0
    lap_out = cv2.Laplacian(out_img, cv2.CV_64F).var() if out_img is not None else 0.0
    
    res["label"] = label
    res["lap_in"] = lap_inp
    res["lap_out"] = lap_out
    results.append(res)

print("\n" + "=" * 95)
print("SUMMARY TABLE: OUT-OF-DISTRIBUTION BENCHMARK RESULTS")
print("=" * 95)
header = f"| {'Sample Description':<35} | {'Bicubic PSNR':<12} | {'SwinIR PSNR':<12} | {'Bicubic SSIM':<12} | {'SwinIR SSIM':<12} | {'Time (ms)':<9} |"
print(header)
print("|" + "-" * 37 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 11 + "|")

for r in results:
    bpsnr = f"{r['bicubic_psnr']:.2f} dB" if r["bicubic_psnr"] is not None else "N/A"
    mpsnr = f"{r['psnr']:.2f} dB" if r["psnr"] is not None else "N/A"
    bssim = f"{r['bicubic_ssim']:.4f}" if r["bicubic_ssim"] is not None else "N/A"
    mssim = f"{r['ssim']:.4f}" if r["ssim"] is not None else "N/A"
    itime = f"{r['inference_time_s']*1000:.1f}"
    print(f"| {r['label']:<35} | {bpsnr:<12} | {mpsnr:<12} | {bssim:<12} | {mssim:<12} | {itime:<9} |")

print("=" * 95)
