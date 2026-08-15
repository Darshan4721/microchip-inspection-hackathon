# AI-Based Restoration of Degraded Images for Semiconductor Inspection

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Architecture-SwinIR_Transformer-green.svg)](#architecture)
[![PSNR](https://img.shields.io/badge/Validation_PSNR-28.68_dB-brightgreen.svg)](#evaluation-results)

An end-to-end deep learning framework for **Joint Denoising and 2× Super-Resolution** of degraded semiconductor microchip inspection images using a **SwinIR (Shifted Window Vision Transformer)** network architecture.

---

## 📌 Repository Structure

```
semicon_hackathon/
├── README.md                  # Complete setup & inference documentation
├── infer.py                   # Standalone evaluation script (--input_dir / --output_dir)
├── model_running_script.py    # Helper single-file execution script
├── requirements.txt           # Complete pip dependencies for reproducibility
├── model_files/
│   ├── best.pth               # Final trained model checkpoint (28.68 dB PSNR)
│   └── latest.pth             # Epoch 20 latest state dict
├── src/
│   ├── model.py               # SwinIR model architecture definition
│   ├── dataset.py             # PyTorch Dataset loaders (.npy & image handling)
│   ├── train.py               # Full training script to reproduce model from scratch
│   └── utils.py               # PSNR/SSIM evaluation metrics calculation
├── Test/
│   └── Test_NoisyLR/          # Test dataset directory (.npy low-res inputs)
├── Train/
│   └── train/train/           # Training dataset (NoisyLR & Ground Truth pairs)
├── test_results/              # Pre-generated restored outputs for 400 test images
└── input_custom/              # Sample test images (.npy and .png formats)
```

---

## ⚙️ Installation & Setup Instructions

Follow these step-by-step instructions to set up the environment and run inference:

### 1. Clone the Repository
```bash
git clone https://github.com/RitikM-AiDev/semicon_hackathon.git
cd semicon_hackathon
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running Inference (Evaluation Script)

The standalone evaluation script `infer.py` runs inference on a test directory or single image file without requiring manual edits.

### Option A: Run Batch Inference on Test Directory (Mandatory Spec)

Run inference on all input images in a specified input directory and write restored outputs ($256 \times 256$) to the target output directory:

```bash
python infer.py --input_dir Test/Test_NoisyLR --output_dir test_results --checkpoint model_files/best.pth
```

- `--input_dir`: Path to folder containing degraded input test files (`.npy`, `.png`, `.jpg`).
- `--output_dir`: Path to folder where restored outputs will be saved.
- `--checkpoint` (or `--weights`): Path to trained weight file (`checkpoints/best.pth`).

### Option B: Run Inference on a Single File
```bash
python infer.py --input input_custom/semicon_000523.npy --output_dir results --checkpoint checkpoints/best.pth
```

---

## 🏋️ Model Training & Reproducibility

To train the SwinIR restoration model from scratch:

```bash
python src/train.py
```

### Training Highlights:
- **Optimizer:** AdamW ($\text{LR} = 1\times 10^{-4}$, Weight Decay = $1\times 10^{-4}$)
- **Learning Rate Scheduler:** CosineAnnealingLR (20 Epochs)
- **Loss Function:** L1 Loss ($\mathcal{L}_1$)
- **Data Augmentation:** Random Flips & Rotations
- **Validation Split:** 10% stratified random split

---

## 📊 Evaluation Results & Performance Metrics

The SwinIR model achieves superior restoration performance compared to standard baselines on the Semiconductor Inspection dataset:

| Model Architecture | Input Resolution | Output Resolution | Validation PSNR (dB) | Validation SSIM |
| :--- | :---: | :---: | :---: | :---: |
| **Bicubic Baseline** | $128 \times 128$ | $256 \times 256$ | 23.02 dB | 0.612 |
| **RRDB-Lite Model** | $128 \times 128$ | $256 \times 256$ | 26.98 dB | 0.745 |
| **SwinIR (Our Model)** | $128 \times 128$ | $256 \times 256$ | **28.68 dB** | **0.812** |

---

## 📁 Submission Components Summary

1. **`README.md`**: Complete standalone documentation and setup guide.
2. **`infer.py`**: Standalone evaluation script supporting `--input_dir` and `--output_dir`.
3. **`src/train.py`**: Full training pipeline script.
4. **`checkpoints/best.pth`**: Trained weight file ($28.68\text{ dB PSNR}$).
5. **`test_results/`**: Restored output images for the 400 test set items.
6. **`requirements.txt`**: Frozen pip dependencies for exact environment replication.
