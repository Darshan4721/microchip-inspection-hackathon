# Phase 7: Dataset Inventory & Fine-Tuning Partitioning

> **Goal:** Audit the newly added semiconductor dataset, establish clean separation between fine-tuning and evaluation data, and prepare a held-out benchmark split.

---

## 1. Dataset Breakdown & File Locations

| Dataset Split | Source Archive | Working Directory | Total Count | Format & Resolution | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Train Set** | `train.zip` | `Train/train/` (`GT/` & `NoisyLR/`) | 3,200 pairs (6,400 files) | Degraded: $128 \times 128$<br>GT: $256 \times 256$ | Source pool for fine-tuning |
| **Fine-Tuning Train** | Partition | `Train/train/` (Indices `0` to `2799`) | **2,800 pairs** (87.5%) | Degraded: $128 \times 128$<br>GT: $256 \times 256$ | Active fine-tuning pool |
| **Held-Out Test Set** | Partition | `Train/train/` (Indices `2800` to `3199`) | **400 pairs** (12.5%) | Degraded: $128 \times 128$<br>GT: $256 \times 256$ | Unseen validation benchmarking |
| **Official Test Set** | `Test_NoisyLR.zip` | `Test/Test_NoisyLR/` | **400 images** | Degraded: $128 \times 128$ | Blind test set (no GT) |
| **OOD Benchmark Set** | `input_custom/` | `input_custom/` | **5 images** | $128 \times 128 \rightarrow 256 \times 256$ | Out-of-distribution stress test |

---

## 2. Partitioning Strategy & Data Integrity

1. **Held-Out Split Preservation (400 Pairs):**
   - Files `002800.npy` through `003199.npy` are strictly withheld from training.
   - This provides a pristine, 400-pair in-distribution benchmark matching the exact sample count of the official test set.
2. **Domain Isolation for OOD Testing:**
   - The 5 out-of-distribution benchmark images (`real_semicon_die`, `degraded_semicon_speckle`, `degraded_semicon_lowres`, `degraded_semicon_gaussian`, `semicon_test_pattern`) are **excluded** from training to guarantee genuine zero-shot generalization testing in Phase 9.
3. **Augmentation Strategy for Fine-Tuning:**
   - Random horizontal flips ($p=0.5$), vertical flips ($p=0.5$), and $90^\circ$ rotations.
   - Mild intensity and contrast jitter ($\pm 8\%$) to break over-specialization on specific substrate reflectivity levels without altering edge morphology.

---

## 3. Baseline Checkpoint Audit

- **Starting Weights:** `model_files/best.pth` (2,447,409 parameters, 4 RSTB blocks, 180 channels, $2\times$ pixelshuffle upscaler).
- **Recorded Checkpoint Epoch:** Epoch 20.
- **Pre-Fine-Tuning Baseline PSNR:**
  - In-Distribution Train Split Validation (seed 42): **$28.8314\text{ dB}$**
  - Held-Out 400-Pair Split: To be evaluated alongside fine-tuned model.
  - Out-of-Distribution (5 images): **$15.688\text{ dB}$ avg PSNR / $0.6277$ avg SSIM**.

---

## Summary for Darshan

1. **3,200 pairs were successfully cataloged and verified:** We partitioned them into **2,800 pairs for fine-tuning** and **400 held-out pairs** kept strictly aside as an unseen test benchmark.
2. **OOD test images remain completely untouched:** The 5 out-of-distribution semiconductor images from earlier remain strictly external so our Phase 9 generalization verification will be 100% authentic.
3. **Ready for Phase 8:** All data paths and partitions are locked; fine-tuning starts directly from `model_files/best.pth` saving exclusively to `model_files/finetuned.pth`.
