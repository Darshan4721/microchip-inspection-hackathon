# Phase 8: Fine-Tuning Verification & Side-by-Side Model Comparison

> **Goal:** Run a comprehensive, head-to-head verification comparison between the original model (`model_files/best.pth`) and the fine-tuned model (`model_files/finetuned.pth`) across out-of-distribution generalization, in-distribution fidelity, and inference latency.

---

## 1. Saved Checkpoint Audit (`model_files/finetuned.pth`)

- **Active File:** `model_files/finetuned.pth` (Size: 80.9 MB)
- **Originating Epoch:** **Epoch 6** (the final epoch of fine-tuning, selected by peak validation score)
- **Training Loss (L1):** `0.029745`
- **Held-Out Validation PSNR (400 Pairs):** `28.5870 dB`
- **Held-Out Validation SSIM (400 Pairs):** `0.7679`

---

## 2, 3 & 4. Out-of-Distribution (OOD) Head-to-Head Comparison

Tested on the exact 5 external semiconductor microscopy and synthetic wafer samples:

| Image / Description | Metric | Bicubic Baseline | Original (`best.pth`) | Fine-Tuned (`finetuned.pth`) | Delta (New vs Old) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Real Die Microscopy**<br>`input_custom/real_semicon_die.png` | PSNR<br>SSIM | 15.45 dB<br>0.5781 | 15.37 dB<br>0.6335 | **15.37 dB**<br>**0.6350** | +0.008 dB<br>+0.0015 |
| **Synthetic Wafer (Speckle Noise)**<br>`input_custom/degraded_semicon_speckle.png` | PSNR<br>SSIM | 15.93 dB<br>0.6048 | 16.18 dB<br>0.6901 | **16.20 dB**<br>**0.6919** | +0.019 dB<br>+0.0018 |
| **Synthetic Wafer (Gaussian Haze)**<br>`input_custom/degraded_semicon_gaussian.png` | PSNR<br>SSIM | 14.45 dB<br>0.4215 | 14.52 dB<br>0.4815 | **14.53 dB**<br>**0.4833** | +0.012 dB<br>+0.0018 |
| **Synthetic Wafer (Pure Downsampling)**<br>`input_custom/degraded_semicon_lowres.png` | PSNR<br>SSIM | 14.12 dB<br>0.4791 | 13.92 dB<br>0.4588 | **13.93 dB**<br>**0.4613** | +0.018 dB<br>+0.0025 |
| **Microchip Test Pattern (Gratings)**<br>`input_custom/semicon_test_pattern.png` | Sharpness<br>(Laplacian Var) | 120.4<br>(Soft blur) | 156.6<br>(Sharp gratings) | **152.5**<br>(Sharp gratings) | Preserved<br>(Identical visual clarity) |
| **Average Across OOD Samples (with GT)** | **PSNR**<br>**SSIM** | **14.99 dB**<br>**0.5209** | **15.00 dB**<br>**0.5660** | **15.01 dB**<br>**0.5679** | **+0.014 dB**<br>**+0.0019** |

---

## 5. Original In-Distribution Test Comparison

Tested on 3 of the original paired test images to verify preservation of baseline fidelity:

| Test Sample | Metric | Bicubic Baseline | Original (`best.pth`) | Fine-Tuned (`finetuned.pth`) | Delta | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`000005.npy`**<br>(Standard circuit lines) | PSNR<br>SSIM | 26.62 dB<br>0.6083 | 28.66 dB<br>0.6868 | **28.64 dB**<br>**0.6828** | -0.02 dB<br>-0.0040 | Maintained (+2.02 dB over bicubic) |
| **`000150.npy`**<br>(Severe speckle $> 1.0$) | PSNR<br>SSIM | 20.93 dB<br>0.2983 | 28.67 dB<br>0.5683 | **28.70 dB**<br>**0.5677** | +0.03 dB<br>-0.0006 | Maintained (+7.77 dB over bicubic) |
| **`000300.npy`**<br>(Dense circuit grid) | PSNR<br>SSIM | 25.92 dB<br>0.4323 | 27.15 dB<br>0.5058 | **27.21 dB**<br>**0.5030** | +0.06 dB<br>-0.0028 | Maintained (+1.29 dB over bicubic) |
| **Average Across In-Dist Samples** | **PSNR**<br>**SSIM** | **24.49 dB**<br>**0.4463** | **28.16 dB**<br>**0.5870** | **28.18 dB**<br>**0.5845** | **+0.02 dB**<br>**-0.0024** | **Completely Stable** |

---

## 6. Plain, Honest Verdict on Generalization

**Did fine-tuning genuinely improve out-of-distribution performance?**

**No.** Real numbers show that the difference is negligible and within normal statistical/floating-point noise:
- Out-of-distribution average PSNR shifted by only **+0.014 dB** ($15.00\text{ dB} \rightarrow 15.01\text{ dB}$).
- Out-of-distribution average SSIM shifted by only **+0.0019** ($0.5660 \rightarrow 0.5679$), which is visually imperceptible.
- In-distribution performance remained essentially identical (**+0.02 dB** PSNR, **-0.0024** SSIM).

**Why didn't fine-tuning produce a breakthrough?**
The new dataset (`train.zip`) came from the exact same microscope and sensor distribution as the original training split. Because `model_files/best.pth` had already completed 20 epochs of convergence on this distribution, fine-tuning for 6 additional epochs could not teach the network how to handle different physical sensors or foreign silicon substrate reflectivity. 

---

## 7. Inference Latency Benchmark

Re-run on 20 test images to verify computational runtime parity:

| Device | Model Architecture | Average Latency | Min Latency | Max Latency | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **GPU (NVIDIA RTX 5060)** | SwinIR (`finetuned.pth`) | **49.4 ms** (0.0494 s) | 45.1 ms | 74.0 ms | **20.2 img/s** |
| **GPU (NVIDIA RTX 5060)** | SwinIR (`best.pth` baseline) | **45.2 ms** (0.0452 s) | 42.3 ms | 69.6 ms | **22.1 img/s** |
| **CPU (Intel Host)** | SwinIR (`finetuned.pth`) | **1,749.3 ms** (1.75 s) | 1,602.7 ms | 1,852.8 ms | **0.57 img/s** |
| **CPU (Intel Host)** | SwinIR (`best.pth` baseline) | **1,868.2 ms** (1.87 s) | 1,779.5 ms | 1,937.0 ms | **0.54 img/s** |

*Conclusion: Inference speed is identical within normal system scheduling variance.*

---

## Summary for Darshan

1. **Before/After OOD Numbers:** OOD average moved from **15.00 dB PSNR / 0.5660 SSIM** (`best.pth`) to **15.01 dB PSNR / 0.5679 SSIM** (`finetuned.pth`) — a statistically negligible change of **+0.01 dB / +0.0019 SSIM**.
2. **In-Distribution Stability:** In-distribution performance held up solidly across all test samples (**28.16 dB $\rightarrow$ 28.18 dB**), confirming zero catastrophic forgetting.
3. **Demo Recommendation:** **Keep using the original checkpoint (`model_files/best.pth`) for the demo** — it is already thoroughly proven, matches the claimed 28.68 dB benchmark in the original documentation, and the fine-tuned model offers no measurable real-world advantage on unseen images.
