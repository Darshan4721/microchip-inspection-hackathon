# VELTRAXX: Reinstall, Setup & Future AI Agent Onboarding Guide

> **Project:** VELTRAXX — Semiconductor Metrology Image Restoration Platform  
> **Repository:** `https://github.com/Darshan4721/microchip-inspection-hackathon.git`  
> **Archival Event:** Botathon Hackathon Finalist Submission (October 2026)  
> **Audience:** Future Developer, Collaborator, or AI Agent (Claude Code / Antigravity / Aider)

---

## 1. Executive Context (For Future AI Agents & Developers)

If you are an **AI coding assistant** or a **human engineer** reopening this repository months or years in the future, read this section first:

1. **What is this repository?**  
   A high-speed deep learning and metrology platform designed to restore noisy, blurred, low-dose Scanning Electron Microscope (SEM) images of semiconductor microchip wafers.
2. **What model is used?**  
   A customized **Swin Transformer for Image Restoration (SwinIR-2x)** with shifted-window self-attention ($8\times 8$ local tiles, 4 RSTBs, PixelShuffle $2\times$ upsampling).
3. **Where are the trained model weights?**  
   In `model_files/`:
   - `model_files/finetuned_v2.pth` (**Flagship Production Model**, 77 MB, trained with synthetic reflectance shifts to overcome wafer charging domain gap).
   - `model_files/best.pth` (Pre-trained baseline model, 77 MB).
4. **Where is the primary application?**  
   `veltraxx_ui.py` — A native dark-mode Tkinter metrology suite featuring 4 synchronized panels, real-time 22pt Consolas telemetry cards (PSNR, SSIM, Latency), interactive defect repair, and keyboard shortcuts.
5. **Where are the test benchmark images?**  
   `final_test_50/clean/` — 50 pristine $256 \times 256$ semiconductor layout samples.

---

## 2. One-Command Setup & Reinstallation

When cloning this repository onto a fresh computer or after wiping your local `.venv`, follow these simple steps:

### Prerequisites:
- Python 3.10 or 3.11 installed.
- (Recommended) NVIDIA GPU with CUDA drivers installed for ~40ms inference. (CPU fallback is supported automatically, running in ~1.2s).
- Package Manager: `uv` (recommended for 10x faster setup) or standard `pip`.

### Quickstart with `uv` (Recommended — Takes ~60 Seconds):
```powershell
# 1. Clone the repository
git clone https://github.com/Darshan4721/microchip-inspection-hackathon.git
cd microchip-inspection-hackathon

# 2. Create a clean virtual environment
uv venv .venv --python 3.11

# 3. Install all dependencies directly
uv pip install -r requirements.txt
```

### Quickstart with Standard `pip`:
```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 3. How to Run the Product

### Method 1: The One-Click Launcher (Windows)
Double-click `run_veltraxx.bat` in File Explorer, or run in terminal:
```powershell
.\run_veltraxx.bat
```

### Method 2: Direct Python Execution
```powershell
.\.venv\Scripts\python.exe veltraxx_ui.py
```

### Keyboard Shortcuts Reference:
- **`[SPACEBAR]`**: Executes SwinIR restoration (~40ms on RTX GPU).
- **`[S-06]` / `[S-35]` / `[S-23]`**: Quick-picks the 3 highest-contrast proof samples (+4.4 dB, +4.0 dB, +5.0 dB leaps).
- **`[ ← ]` / `[ → ]`**: Cycle previous/next semiconductor test samples.
- **`[ R ]`**: Resets degradation sliders back to default ($\sigma=1.0$, noise $6\%$).
- **Mouse Drag (Panel 2)**: Hand-draw a defect to test reference-free inpainting.

---

## 4. How to Run Headless Benchmarks & Tests

If an AI agent needs to verify the code without launching a graphical window:

```powershell
# 1. Run the headless UI verification test suite
.\.venv\Scripts\python.exe test_ui_verification.py

# 2. Run single image inference CLI
.\.venv\Scripts\python.exe demo_single.py

# 3. Run out-of-distribution benchmark
.\.venv\Scripts\python.exe benchmark_ood.py
```

---

## 5. Architectural Directory Layout (Mental Map)

```
microchip-inspection-hackathon/
├── src/                               # Core Neural Network Package
│   ├── model.py                       # create_model() factory function
│   ├── network_swinir.py              # Pure PyTorch SwinIR implementation
│   ├── utils.py                       # calculate_psnr() & calculate_ssim()
│   ├── dataset.py                     # NPY / image dataset loader
│   ├── finetune_v2.py                 # Reflectance-invariance training script
│   └── train.py                       # Base training loop
├── model_files/                       # Release Model Checkpoints (Tracked in Git)
│   ├── finetuned_v2.pth               # 🌟 Flagship Model (77 MB, used by UI)
│   └── best.pth                       # Baseline pre-trained model (77 MB)
├── final_test_50/                     # 50-Image Unseen Benchmark Suite
│   ├── clean/                         # 50 clean Ground Truth PNGs
│   ├── degraded/                      # 50 degraded sensor inputs
│   └── viewer.html                    # Zero-dependency browser comparison tool
├── veltraxx_ui.py                     # Flagship Desktop Metrology Suite (Tkinter)
├── run_veltraxx.bat                   # 1-click Windows launcher
├── requirements.txt                   # Frozen production dependencies
├── pyproject.toml                     # Python project packaging metadata
│
└── Documentation / Knowledge Base:
    ├── REINSTALL_AND_AGENT_SETUP_GUIDE.md       # (This file) Reinstall & Agent Onboarding
    ├── PROJECT_HISTORY_AND_RETROSPECTIVE.md     # Chronological development & decisions
    ├── VELTRAXX_HACKATHON_WINNING_PITCH.md      # Pitch script, cues & judge Q&A
    ├── VELTRAXX_DEEP_PROJECT_BRAIN_AND_THINKING.md # Non-IT project manual & semiconductor physics
    ├── VELTRAXX_UI_DESIGN_AND_ARCHITECTURE.md  # UI design philosophy, 7 decisions & threading
    └── CLEANUP_AND_GITHUB_EXPORT_REPORT.md      # Storage & cleanup master audit
```

---

## 6. Guidance for Future AI Agents

If a user prompts you: *"Continue working on this semiconductor restoration project"*:
1. **DO NOT retrain from scratch** unless explicitly asked. The flagship weights in `model_files/finetuned_v2.pth` have already achieved a 100% win rate across 50 unseen layout images.
2. **DO NOT re-introduce generative models (GANs, Stable Diffusion)**. In semiconductor inspection, hallucinating fake 5nm wires causes wafer scrapping. Keep the model strictly deterministic.
3. **If adding features:**
   - Keep Tkinter thread safety intact by dispatching UI updates through `self.dispatch_to_ui()` via `queue.Queue`.
   - Prevent canvas memory leaks by recycling canvas items with `delete("all")` before drawing.
   - For rapid inference boost without retraining, refer to `MASTER_AUDIT_REPORT.md` (8-fold geometric TTA or Model Soup weight averaging).

---
*Authored by Antigravity AI for Team VELTRAXX — Preserved for long-term open-source reproducibility.*
