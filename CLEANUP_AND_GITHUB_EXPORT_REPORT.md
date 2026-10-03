# VELTRAXX: Storage Cleanup & GitHub Export Master Report

> **Prepared For:** Project Owner / Hackathon Finalist  
> **Repository:** `D:\tmp\eorde_hackathon_2`  
> **GitHub Remote:** `https://github.com/Darshan4721/microchip-inspection-hackathon.git`  
> **Audit Date:** October 3, 2026

---

## 1. Congratulations on Reaching the Finals!

Reaching the **final round** of an advanced hardware/AI hackathon (AI + VLSI / Semiconductor Metrology) is a massive achievement. Competing against specialized engineering teams and advancing all the way to the final stage proves that the problem statement, working demonstration, and presentation defense were competitive at the highest level.

This document will help you cleanly preserve this project on GitHub for your portfolio, explain why your Windows C: drive was filling up, and give you the exact steps to reclaim **~11 GB on your C: drive** and **up to ~8.3 GB on your D: drive**.

---

## 2. The Mystery Solved: Why Did the Previous Report Say 14 GB When the Folder Was 11 GB?

You noticed that `D:\tmp\eorde_hackathon_2` was only ~8.6 to 11 GB on disk, but the audit reported over 14 GB of recoverable files.

### The Answer: The Missing 11 GB Was Hidden on Your C: Drive!
When Python packages for heavy deep learning (PyTorch with CUDA 12.8, triton, cuDNN, torchvision) were installed, the package managers (`uv` and `pip`) **downloaded and cached the installer wheels onto your Windows C: drive** inside your user `AppData` directory, even though the project folder was on `D:\`!

```
+───────────────────────────────────────────────────────────────────────────────────────────+
|                                THE FULL DISK USAGE BREAKDOWN                              |
+───────────────────────────────────────────────────────────────────────────────────────────+
|  1. C: DRIVE HIDDEN CACHES (Installed during project setup):                              |
|     - uv Package Cache: C:\Users\radar\AppData\Local\uv\cache         -> 10.04 GB (10,280 MB)|
|     - pip Wheel Cache:   C:\Users\radar\AppData\Local\pip\cache        ->  0.95 GB    (973 MB)|
|     - uv Python Runtimes: C:\Users\radar\AppData\Roaming\uv\python    ->  0.38 GB    (389 MB)|
|     SUBTOTAL ON C: DRIVE                                              = 11.37 GB          |
|                                                                                           |
|  2. D: DRIVE PROJECT FOLDER (D:\tmp\eorde_hackathon_2):                                   |
|     - .venv (PyTorch + CUDA 12.8 libraries in virtual environment)    ->  4.48 GB (4,480 MB) |
|     - Leftover Root .zip Archives (train.zip, stitch_*.zip, etc.)    ->  1.53 GB (1,565 MB) |
|     - Train/ Dataset (3,200 unzipped training patches + junction)     ->  1.00 GB (1,000 MB) |
|     - checkpoints/ (9 obsolete intermediate training saves)           ->  0.68 GB   (695 MB) |
|     - model_files/ (4 .pth models: best, v2, finetuned, latest)       ->  0.30 GB   (309 MB) |
|     - .git/ (Repository history)                                      ->  0.25 GB   (251 MB) |
|     - test_results/ (old batch test dumps)                            ->  0.21 GB   (212 MB) |
|     - SwinIR/ + Test/ + other test directories                        ->  0.12 GB   (120 MB) |
|     - UI scripts, presentation docs, final_test_50/                   ->  0.05 GB    (50 MB) |
|     SUBTOTAL ON D: DRIVE                                              =  8.62 GB          |
+───────────────────────────────────────────────────────────────────────────────────────────+
|  TOTAL COMBINED DISK USAGE ACROSS BOTH DRIVES                         = 19.99 GB          |
+───────────────────────────────────────────────────────────────────────────────────────────+
```

When the earlier report said **"~14 GB recoverable"**, it was combining:
- **11.0 GB** in C: drive package caches
- **1.53 GB** in leftover root zip files on D:
- **0.68 GB** in obsolete checkpoints on D:
- **0.21 GB** in obsolete test results on D:
- **Total = 13.42 GB of 100% safe throwaway bloat!**

---

## 3. C: DRIVE CLEANUP GUIDE (Strict Read-Only Compliance)

> [!IMPORTANT]
> **Strict Rule Honored:** We did **NOT** delete, modify, or touch any files on your C: drive.  
> As a web and desktop developer, you need your C: drive clean and fast. Below are the exact commands you can run yourself in PowerShell to reclaim **11 GB** of free space on C: immediately.

### What is Taking Space on C: Drive?
1. **`C:\Users\radar\AppData\Local\uv\cache` (10.04 GB):**  
   Inside `archive-v0`, the folder `eJwiNLQXIVUA6-1g` is **4.17 GB** (the raw PyTorch CUDA 12.8 wheel), `Ne5H1QjQ5A9t4Dtk` is **828 MB** (cuBLAS/cuDNN), and several other CUDA packages totaling 10.04 GB.  
   *Is it safe to delete?* **YES, 100% safe.** A package cache is just an installation temp file. Deleting it does NOT break Python or your projects.
2. **`C:\Users\radar\AppData\Local\pip\cache` (0.95 GB):**  
   Cached `.whl` files from pip installations.  
   *Is it safe to delete?* **YES, 100% safe.**

### Commands to Run in PowerShell to Free ~11 GB on C: Drive:

#### Command 1: Clean the `uv` cache (Frees 10.04 GB on C:)
```powershell
& "C:\Users\radar\.local\bin\uv.exe" cache clean
```
*(Or simply `uv cache clean` if `uv` is in your system PATH).*

#### Command 2: Clean the `pip` cache (Frees 0.95 GB on C:)
```powershell
pip cache purge
```
*(Or if running inside the project folder: `.\.venv\Scripts\pip.exe cache purge`).*

#### Result on C: Drive:
Your C: drive free space will immediately jump from **17.66 GB $\rightarrow$ ~28.7 GB**!

---

## 4. D: DRIVE CLEANUP PLAN (`D:\tmp\eorde_hackathon_2`)

Here is the exact item-by-item breakdown of what is in your project folder, what is safe to delete, and what should be kept for GitHub:

| Item / Directory | Current Size | Status | Action / Recommendation |
| :--- | :---: | :---: | :--- |
| **5 Root `.zip` Files** | **1.53 GB** | Obsolete | **Delete immediately.** (`train.zip`, `stitch_*.zip`, `test_clean_images.zip`, `Test_NoisyLR.zip`). These were extracted during Phase 1-7 and are useless duplicates. |
| **`checkpoints/`** | **694.58 MB** | Obsolete | **Delete immediately.** Contains 9 intermediate epoch checkpoints (`finetune_epoch_*.pth`, `v2_epoch_*.pth`). The final trained model was already saved to `model_files/`. |
| **`test_results/`** | **211.93 MB** | Obsolete | **Delete immediately.** 400 old `.npy` and `.png` test dumps from an intermediate validation script. |
| **`Train/` directory** | **1,000.78 MB** | Raw Training Data | **Safe to delete or archive.** Contains 3,200 raw training patches and an NTFS junction link `Train\train\train`. Since the hackathon is over and you are not training another model, this is not needed to run the UI or view the project. |
| **`Test/` + `SwinIR/` + `results/`** | **116.13 MB** | Historical Scratch | **Safe to delete.** Unused benchmark scratch folders. |
| **`.venv/`** | **4.48 GB** | Virtual Environment | **Delete if you want maximum disk space.** If you mainly do web development, keeping a 4.5 GB PyTorch CUDA venv sitting on disk is unnecessary. You can recreate it anytime in 2 minutes using `uv sync` or `pip install -r requirements.txt`. If you want to keep the UI runnable with 1-click right now, keep `.venv/`. |
| **`model_files/`** | **308.61 MB** | **KEEP** | Contains `finetuned_v2.pth` (our flagship 77MB model) and `best.pth`. *(Already tracked in Git).* |
| **`final_test_50/`** | **7.59 MB** | **KEEP** | Contains the 50 clean benchmark test images used by the UI and viewer. |
| **`src/`** | **0.19 MB** | **KEEP** | Core Python source code (`model.py`, `network_swinir.py`, `utils.py`). |
| **UI Scripts & Docs** | **< 1.0 MB** | **KEEP** | `veltraxx_ui.py`, `run_veltraxx.bat`, and the 4 master `.md` presentation and architecture guides. |

### Summary of Recoverable Space on D: Drive:
- Deleting Category 1 (Zips + Checkpoints + Test Results): **Reclaims 2.44 GB**.
- Deleting Category 2 (Raw Train dataset): **Reclaims 1.00 GB**.
- Deleting Category 3 (`.venv`): **Reclaims 4.48 GB**.
- **Total Potential Space Freed on D: = ~7.92 GB!**  
- The final clean project size on disk will be **under 350 MB** (or ~4.8 GB if keeping `.venv`).

---

## 5. GitHub Export & Upload Checklist

To upload this repository cleanly to GitHub so you can showcase it on your portfolio or use it in a future AI + VLSI hackathon:

### Step 1: Ensure `.gitignore` Excludes All Large Files
Your `.gitignore` file should ensure no large datasets, zips, or venvs are pushed:
```gitignore
# Virtual environment
.venv/
env/
ENV/

# Python cache
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/

# Datasets and heavy temporary artifacts
Test/
Train/
SwinIR/
*.zip
checkpoints/
test_results/
results/
test_batch_20/
varied_imaging_test/
input_custom/
*.npy

# Scratch backups
veltraxx_ui_backup.py
test_ui_verification.py
```

### Step 2: What Will Land on GitHub?
When pushed, your GitHub repository will contain:
1. `src/` — Clean, complete SwinIR model implementation.
2. `model_files/` — The trained `finetuned_v2.pth` and `best.pth` checkpoints *(already on GitHub!)*.
3. `final_test_50/` — The benchmark test suite with sample images.
4. `veltraxx_ui.py` & `run_veltraxx.bat` — The full presentation desktop application with 22pt Consolas cards and memory leak fixes.
5. All 4 Documentation Manuals:
   - `VELTRAXX_HACKATHON_WINNING_PITCH.md`
   - `VELTRAXX_DEEP_PROJECT_BRAIN_AND_THINKING.md`
   - `VELTRAXX_UI_DESIGN_AND_ARCHITECTURE.md`
   - `README.md`
6. `requirements.txt` & `pyproject.toml` — Exact dependency list for reproducibility.

### Step 3: Git Commands to Commit and Push
Run these commands in PowerShell inside `D:\tmp\eorde_hackathon_2`:
```powershell
# 1. Stage the updated UI, batch files, and documentation
git add veltraxx_ui.py whale_tracks_ui.py run_veltraxx.bat run_whale_tracks.bat .gitignore
git add VELTRAXX_HACKATHON_WINNING_PITCH.md VELTRAXX_DEEP_PROJECT_BRAIN_AND_THINKING.md VELTRAXX_UI_DESIGN_AND_ARCHITECTURE.md WHALE_TRACKS_PITCH_AND_TEAM_DEFENSE.md

# 2. Commit the changes
git commit -m "Final Hackathon Submission: Veltraxx UI polish, pitch guides, and architecture documentation"

# 3. Push to GitHub
git push origin main
```

---

## 6. Commands to Delete Unwanted D: Drive Files (When You Choose)

Whenever you are ready to delete the obsolete zip archives, old checkpoints, and test result dumps on your D: drive, you can run this clean command in PowerShell inside `D:\tmp\eorde_hackathon_2`:

```powershell
# Delete the 5 leftover .zip archives (frees 1.53 GB)
Remove-Item *.zip -Force

# Delete obsolete intermediate checkpoints (frees 695 MB)
Remove-Item -Recurse -Force checkpoints

# Delete old test dumps (frees 212 MB)
Remove-Item -Recurse -Force test_results

# Delete temporary backup/test scripts
Remove-Item -Force veltraxx_ui_backup.py, test_ui_verification.py -ErrorAction SilentlyContinue

# (Optional) Delete the 1 GB raw training patches if not re-training
# Remove-Item -Recurse -Force Train
```

---
*Report compiled for Team VELTRAXX — Ready for clean GitHub archival and C: drive recovery.*
