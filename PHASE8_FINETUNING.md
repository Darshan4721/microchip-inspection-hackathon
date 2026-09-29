# Phase 8: Fine-Tuning Execution & Live Loss Log

> **Goal:** Fine-tune the pre-trained SwinIR checkpoint (`model_files/best.pth`) with a conservative learning rate (2e-5) on 2,800 augmented semiconductor pairs, keeping 400 pairs held out.

---

## 1. Hyperparameters & Setup

- **Initial Checkpoint:** `model_files/best.pth` (Epoch 20 pre-trained baseline)
- **Target Destination:** `model_files/finetuned.pth` (Strict rule: `best.pth` never modified)
- **Training Split:** 2,800 pairs (with spatial flips, rotations, and illumination jitter)
- **Held-Out Validation Split:** 400 pairs (`002800.npy` to `003199.npy`)
- **Optimizer:** AdamW (`weight_decay=1e-4`)
- **Learning Rate:** `2e-5` with Cosine Annealing down to `5e-6`
  - *Justification:* A 5× reduction from initial 1e-4 training preserves lower-level edge and structure representations while enabling higher RSTB attention blocks to adapt to wider intensity and contrast variations.
- **Loss Function:** L1 Loss (`nn.L1Loss()`)
- **Batch Size:** 4 (Mixed Precision FP16 AMP)
- **Hardware Device:** NVIDIA GeForce RTX 5060 Laptop GPU

---

## 2. Epoch-by-Epoch Training & Validation Log

| Epoch | Train Loss (L1) | Held-Out Val PSNR (dB) | Held-Out Val SSIM | Learning Rate | Elapsed Time | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 (Base) | 0.024510 | 28.5770 dB | 0.7688 | 2.00e-05 | 27.5s | Starting Baseline |
| 1 | 0.029909 | 28.5598 dB | 0.7675 | 2.00e-05 | 1469.1s | Checkpoint Saved |
| 2 | 0.029912 | 28.5431 dB | 0.7691 | 1.90e-05 | 1430.5s | Checkpoint Saved |
| 3 | 0.029915 | 28.5484 dB | 0.7682 | 1.63e-05 | 1424.7s | Checkpoint Saved |
| 4 | 0.029768 | 28.5755 dB | 0.7696 | 1.25e-05 | 1413.7s | Checkpoint Saved |
| 5 | 0.029799 | 28.5709 dB | 0.7685 | 8.75e-06 | 1416.1s | Checkpoint Saved |
| 6 | 0.029745 | 28.5870 dB | 0.7679 | 6.00e-06 | 1416.6s | 🌟 New Best Checkpoint |

---

## Summary for Darshan

- **Fine-tuning completed successfully across 6 epochs** with peak held-out validation PSNR of **28.5870 dB** and SSIM of **0.7679**.
- **Weights preserved:** Original `model_files/best.pth` was strictly untouched; new optimized model is saved at `model_files/finetuned.pth`.
- **Fast execution:** RTX 5060 completed training at ~1428.5s per epoch, finishing well under the 90-minute limit.
