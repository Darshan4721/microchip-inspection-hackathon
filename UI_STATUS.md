# Phase 1 Report

## Step 1: Clone and Branch
- **Repo status**: Cloned successfully. Clean working tree initially.
- **Branch**: Created and switched to `ui-demo` successfully.
- **Push access**: Verified `git push -u origin ui-demo`. Failed with authentication/permission error (no credentials configured for push access).

## Step 2: Read & Directory Structure
Read `README.md`, `STATUS.md`, and `demo_single.py`.
**Contents of `input_custom/`:**
- `000005.npy` (64KB)
- `000150.npy` (64KB)
- `000300.npy` (64KB)
- `degraded_semicon_gaussian.png` (14KB)
- `degraded_semicon_lowres.png` (1KB)
- `degraded_semicon_speckle.png` (14KB)
- `new_semicon_clean_gt.png` (1KB)
- `real_semicon_die.npy` (128KB)
- `real_semicon_die.png` (13KB)
- `real_semicon_die_gt.png` (1KB)
- `semicon_test_pattern.png` (629B)

## Step 3: Understanding `demo_single.py`
- **Function/Signature**: `run_restoration(input_path: str, output_path: str = None, gt_path: str = None, checkpoint: str = "model_files/best.pth", device: str = None)`
  - Returns a dictionary with keys: `input`, `output`, `inference_time_s`, `input_shape`, `output_shape`, `psnr`, `ssim`, `bicubic_psnr`, `bicubic_ssim`.
- **Module Side-Effects**: Importing `demo_single.py` has no side effects. The execution logic is guarded by `if __name__ == "__main__":`. The model loading logic is encapsulated inside `run_restoration()`.
- **Metrics & Ground Truth**: 
  - PSNR and SSIM are computed using `calculate_psnr` and `calculate_ssim` imported from `utils`.
  - Ground truth matching is handled by `find_matching_gt(input_path, explicit_gt)`. It intelligently looks for `${stem}_gt.png`, `${stem}_clean_gt.png`, and explicitly maps known test prefixes (e.g., if "degraded_semicon" in stem, it uses `new_semicon_clean_gt.png`).
- **Inference Time**: Measured accurately using `time.perf_counter()` with proper `torch.cuda.synchronize()` before and after model inference.
- **Input/Output Format**: Reads `.npy` array or PIL image (converted to grayscale `"L"`). Scales input to floats `[0, 1]` and automatically pads height/width to multiples of 8. The output is cropped back, scaled to 0-255, and saved as a grayscale 8-bit `.png`.
- **Weights File**: Expects `model_files/best.pth`.
  - The weights file **does exist** in the repository and is 80.8 MB in size.

## Step 4: Environment Check
- **Python Version**: Python 3.12.7 (Miniconda)
- **Dependencies**: `torch`, `numpy`, `PIL` (pillow), and metrics libraries are **NOT installed** in this environment. 
- **Streamlit**: Was not installed, but I am installing it via `pip install streamlit` as permitted.
- **GPU Availability**: NVIDIA GeForce RTX 3050 Laptop GPU (6GB) is available on the system.

## Step 5: Baseline Run
Attempted to run `demo_single.py` as intended on an image from `input_custom/`:
```bash
python demo_single.py --input input_custom/degraded_semicon_gaussian.png
```
**Output / Error Traceback**:
```text
Traceback (most recent call last):
  File "C:\antigravity_cli\microchip-inspection-hackathon\demo_single.py", line 10, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```
*(No files were written due to the immediate failure. Following strict instructions, I did not attempt to install `numpy` or `torch` to fix the backend.)*

## Step 6: Sample Image Inventory
**Candidate images found in `input_custom/`:**
1. `000005.npy`, `000150.npy`, `000300.npy` - Do not have local GT files explicitly mapped.
2. `degraded_semicon_gaussian.png` - Matches with `new_semicon_clean_gt.png`
3. `degraded_semicon_lowres.png` - Matches with `new_semicon_clean_gt.png`
4. `degraded_semicon_speckle.png` - Matches with `new_semicon_clean_gt.png`
5. `real_semicon_die.png` (and `.npy`) - Matches with `real_semicon_die_gt.png`
6. `semicon_test_pattern.png` - No matching ground truth.

**Out-of-distribution (OOD) / New Images**: 
- `semicon_test_pattern.png`: A completely different image type with no corresponding GT in the folder.
- The `.npy` files (`000005`, `000150`, `000300`) might also serve as OOD depending on what the model was trained on, as they lack GT locally.

---
## Phase 1b Report

### STEP 1: Virtual environment
- Created virtual environment `.venv-ui`.
- Added `.venv-ui/` to `.git/info/exclude`.
- `git status` shows clean tree, venv ignored.
- Confirmed `sys.executable` points to `C:\antigravity_cli\microchip-inspection-hackathon\.venv-ui\Scripts\python.exe`.

### STEP 2: Install dependencies
- Attempted to install PyTorch with CUDA using the cu128 wheel (since Python 3.14 was installed, cu121 and cu124 wheels returned `No matching distribution found for torch`).
- The `torch-2.11.0+cu128` wheel is 2.7 GB. Due to network speed constraints, the download took longer than 10 minutes (only reaching ~15% completion).
- Per the hard rules ("If any install takes more than ~10 minutes or fails, stop and report the exact error. Don't experiment endlessly"), I aborted the installation.
- I was unable to freeze the requirements or verify the `torch.__version__` output.

### STEP 3: Push access
- Ran `$env:GIT_TERMINAL_PROMPT=0; git push -u origin ui-demo`.
- Result:
  ```
  branch 'ui-demo' set up to track 'origin/ui-demo'.
  Everything up-to-date
  ```
- Push access is **SUCCESSFUL** (cached credentials worked).

### STEP 4: Real baseline runs
- **Blocked**. Cannot run `demo_single.py` because PyTorch and dependencies failed to install within the 10-minute limit.

### STEP 5: Facts I need for the UI design
- **Resolution and shape**:
  - `degraded_semicon_gaussian.png`: Input 128x128. GT (`new_semicon_clean_gt.png`): 256x256.
  - `real_semicon_die.png`: Input 128x128. GT (`real_semicon_die_gt.png`): 256x256.
  - `semicon_test_pattern.png`: Input 128x128. No GT.
  - The model output shape is 2x the input, so it will be 256x256.
  - The GT matches the **output** size (256x256), not the input size. The UI will be able to display them side-by-side cleanly.
- **`output_path=None` behavior**: `run_restoration()` will still write the output file to `results/{inp_p.stem}_restored.png`. The returned dictionary contains the path string in the `"output"` key, but it does NOT contain the raw image array/tensor in memory. The UI will need to read the image from the saved path.
- **Model reloading**: `run_restoration()` reloads the model weights from disk on *every* call. The exact lines (68-73 in `demo_single.py`) are:
  ```python
  model = create_model().to(device)
  checkpoint_data = torch.load(ckpt_p, map_location=device)
  if isinstance(checkpoint_data, dict) and "model" in checkpoint_data:
      model.load_state_dict(checkpoint_data["model"])
  else:
      model.load_state_dict(checkpoint_data)
  model.eval()
  ```
- **In-distribution vs OOD claims**:
  - The `README.md` and `STATUS.md` state the model achieved 28.68 dB PSNR on "in-distribution data".
  - For `input_custom/` images, it says: "When evaluated on unseen real die and wafer samples (in `input_custom/`), SwinIR achieves +0.055 to +0.085 SSIM gain over Bicubic interpolation... Quality drops from the 28.68 dB claimed on in-distribution data down to 14.5–16.2 dB on out-of-distribution structures. This is due to a slight global substrate brightness shift."
  - For purely low-res without noise: "On pure low-res images with zero noise, SwinIR performs slightly worse than bicubic (-0.20 dB) because the network was trained on noisy pairs and expects noise artifacts."
  - Therefore, the images in `input_custom/` are officially considered **Out-of-Distribution (OOD)** by the documentation, and we should label them as such.
- **The `.npy` files (`000005`, `000150`, `000300`)**: These appear to be raw validation/test dataset samples. `demo_single.py` tries to match their GT by looking in `ROOT / "Train" / "train" / "train" / "GT"`. However, the `Train/` and `Test/` folders were deliberately git-ignored and do not exist in the repository. Thus, these `.npy` samples have no GT available anywhere in the repo.

---
## Phase 1c Report

### Setup Checks
- `git branch --show-current`: `ui-demo`
- `git status --short`:
  ```
   M results/degraded_semicon_gaussian_restored.png
  ?? UI_STATUS.md
  ```
  *(Note: I immediately restored the modified tracked file `results/degraded_semicon_gaussian_restored.png` to maintain a completely clean tree as required.)*
- `& $PY --version`: `Python 3.14.7`
- `& $PY -c "import torch..."`: `OK 2.14.0+cpu False`
- **Missing / Different Packages (vs requirements.txt)**:
  - `anyio`: installed 4.15.1, required 4.14.2
  - `certifi`: installed 2026.7.22, required 2022.12.7
  - `click`: installed 8.5.0, required 8.4.2
  - `huggingface_hub`: installed 2.0.0, required 1.27.0
  - `idna`: installed 3.20, required 3.4
  - `numpy`: installed 2.5.2, required 2.4.6
  - `packaging`: installed 26.3, required 24.1
  - `timm`: installed 1.0.30, required 1.0.28
  - `torch`: installed 2.14.0+cpu, required 2.11.0+cu128
  - `torchvision`: installed 0.29.0+cpu, required 0.26.0+cu128
  - `tqdm`: installed 4.70.1, required 4.66.5
  - **Missing**: `cuda-bindings`, `cuda-pathfinder`, `cuda-toolkit`, `httpcore`, `httpx`, `nvidia-*` packages, `semicon-hackathon`, `triton`.

### Baseline Runs
*(Run using `--output untracked_results/...` to prevent modifying tracked files)*

**a) degraded_semicon_gaussian.png (Run 1)**
```text
=================================================================
  SEMICONDUCTOR IMAGE RESTORATION (SwinIR 2x)
=================================================================
 Input File       : degraded_semicon_gaussian.png (128x128) [image]
 Output File      : degraded_semicon_gaussian_restored.png (256x256)
 Device Used      : CPU
 Inference Time   : 2.9126 seconds (2912.6 ms)
 Input Val Range  : [0.0000, 1.0000]
 Output Val Range : [-0.0123, 0.9222]
-----------------------------------------------------------------
 Ground Truth Match : new_semicon_clean_gt.png
 Bicubic Baseline   : PSNR = 14.45 dB  |  SSIM = 0.4215
 SwinIR Model       : PSNR = 14.52 dB  |  SSIM = 0.4815
 Model vs Baseline  : Delta PSNR = +0.07 dB | Delta SSIM = +0.0599
=================================================================
Total Wall Time: 7.5182818 seconds
```

**b) real_semicon_die.png**
```text
=================================================================
  SEMICONDUCTOR IMAGE RESTORATION (SwinIR 2x)
=================================================================
 Input File       : real_semicon_die.png (128x128) [image]
 Output File      : real_semicon_die_restored.png (256x256)
 Device Used      : CPU
 Inference Time   : 2.5958 seconds (2595.8 ms)
 Input Val Range  : [0.0627, 1.0000]
 Output Val Range : [0.0803, 0.9569]
-----------------------------------------------------------------
 Ground Truth Match : real_semicon_die_gt.png
 Bicubic Baseline   : PSNR = 15.45 dB  |  SSIM = 0.5781
 SwinIR Model       : PSNR = 15.37 dB  |  SSIM = 0.6335
 Model vs Baseline  : Delta PSNR = -0.08 dB | Delta SSIM = +0.0554
=================================================================
Total Wall Time: 7.2489501 seconds
```

**c) semicon_test_pattern.png (No GT expected)**
```text
=================================================================
  SEMICONDUCTOR IMAGE RESTORATION (SwinIR 2x)
=================================================================
 Input File       : semicon_test_pattern.png (128x128) [image]
 Output File      : semicon_test_pattern_restored.png (256x256)
 Device Used      : CPU
 Inference Time   : 2.6340 seconds (2634.0 ms)
 Input Val Range  : [0.0980, 0.9490]
 Output Val Range : [0.0796, 0.9664]
-----------------------------------------------------------------
 Ground Truth Match : None found (saved restored image without metrics)
=================================================================
Total Wall Time: 7.0803428 seconds
```

**d) degraded_semicon_speckle.png**
```text
=================================================================
  SEMICONDUCTOR IMAGE RESTORATION (SwinIR 2x)
=================================================================
 Input File       : degraded_semicon_speckle.png (128x128) [image]
 Output File      : degraded_semicon_speckle_restored.png (256x256)
 Device Used      : CPU
 Inference Time   : 2.6527 seconds (2652.7 ms)
 Input Val Range  : [0.0706, 1.0000]
 Output Val Range : [0.1029, 0.9262]
-----------------------------------------------------------------
 Ground Truth Match : new_semicon_clean_gt.png
 Bicubic Baseline   : PSNR = 15.93 dB  |  SSIM = 0.6048
 SwinIR Model       : PSNR = 16.18 dB  |  SSIM = 0.6901
 Model vs Baseline  : Delta PSNR = +0.25 dB | Delta SSIM = +0.0853
=================================================================
Total Wall Time: 7.2784013 seconds
```

**a) degraded_semicon_gaussian.png (Run 2)**
```text
=================================================================
  SEMICONDUCTOR IMAGE RESTORATION (SwinIR 2x)
=================================================================
 Input File       : degraded_semicon_gaussian.png (128x128) [image]
 Output File      : degraded_semicon_gaussian_restored.png (256x256)
 Device Used      : CPU
 Inference Time   : 2.6999 seconds (2699.9 ms)
 Input Val Range  : [0.0000, 1.0000]
 Output Val Range : [-0.0123, 0.9222]
-----------------------------------------------------------------
 Ground Truth Match : new_semicon_clean_gt.png
 Bicubic Baseline   : PSNR = 14.45 dB  |  SSIM = 0.4215
 SwinIR Model       : PSNR = 14.52 dB  |  SSIM = 0.4815
 Model vs Baseline  : Delta PSNR = +0.07 dB | Delta SSIM = +0.0599
=================================================================
Total Wall Time: 7.292603 seconds
```

### Checks
1. **Metrics Variance**: Yes, PSNR/SSIM differ across images (e.g. Gaussian: 14.52 dB / 0.4815; Real Die: 15.37 dB / 0.6335; Speckle: 16.18 dB / 0.6901).
2. **Reproducibility**: The second run of (a) produced the exact same PSNR (14.52 dB) and SSIM (0.4815). The inference time was slightly faster (2.6999s vs 2.9126s), indicating a minor warm-up effect even on CPU.
3. **Missing GT Handling**: The `semicon_test_pattern.png` run handled the missing GT cleanly, outputting `Ground Truth Match : None found (saved restored image without metrics)` without crashing.
4. **Untracked Outputs**: All outputs were written to `untracked_results/` which I subsequently added to `.git/info/exclude`.
5. **Output Verification**: The resulting files are exactly 256x256 pixels, mode `L` (grayscale), and exist on disk with sizes ranging from 20KB to 40KB.

---
## Phase 2 Report

### What was Built
- `app.py`: A Streamlit web application that serves as the UI layer for the existing SwinIR backend. It includes:
  - An input selector handling both pre-loaded examples and custom user uploads (with or without ground truth).
  - Proper label handling according to in-distribution and OOD claims from the original documentation.
  - A responsive three-column image viewer displaying the degraded input, restored output, and ground truth (enlarged via Nearest Neighbor to 512x512 to preserve raw pixel structures without introducing artifacting).
  - A dynamic metrics section that extracts live PSNR, SSIM, and inference time directly from `run_restoration()`.
- `.streamlit/config.toml`: Configured to run headlessly and disable usage stats, preventing email prompts during live demos.
- `requirements-ui.txt`: Added unpinned requirements list specifying the necessary packages for the UI, explicitly noting that CPU-only PyTorch is sufficient.
- `README-UI.md`: Documented the setup, features, included examples, and troubleshooting steps.

### Test Results Table
| Image | PSNR | SSIM | Inference Time (s) | Bicubic PSNR | Bicubic SSIM | GT Shown |
|---|---|---|---|---|---|---|
| `degraded_semicon_gaussian.png` | 14.5188 dB | 0.4815 | 2.67 s | 14.4458 dB | 0.4215 | Yes |
| `real_semicon_die.png` | 15.3651 dB | 0.6335 | 2.68 s | 15.4450 dB | 0.5781 | Yes |
| `degraded_semicon_speckle.png` | 16.1837 dB | 0.6901 | 2.66 s | 15.9287 dB | 0.6048 | Yes |
| `semicon_test_pattern.png` | N/A | N/A | 2.78 s | N/A | N/A | No |

*Note: The Gaussian numbers exactly match the baseline CLI checks from Phase 1c. All numbers differ organically across the examples.*

### How Tests Were Performed
1. **Launch Check**: Ran `python -m streamlit run app.py --server.headless true --server.port 8501` as a background process and successfully curled `http://localhost:8501/_stcore/health`, receiving an HTTP 200 `ok` response.
2. **End-to-End Click-Through**: Wrote and executed a script (`test_app.py`) utilizing Streamlit's `AppTest` framework (`streamlit.testing.v1.AppTest`). The script programmaticly selected the four required test images, clicked the "Restore" button, and scraped the rendered metrics and UI elements to verify correctness.
3. **Upload Path Test**: Tested the underlying upload function `save_uploaded_files` directly in the `test_app.py` script by passing mock `UploadedFile` objects and asserting that the files were correctly written to `ui_outputs/uploads/` on disk.
4. **Hardcode Check**: Ran PowerShell's `Select-String` against `app.py` to assert that no literal metric values (e.g., `14.5`, `0.48`, `0.63`) were hardcoded in the source code.

### Known Issues
- The model weights are reloaded from disk on every invocation of `run_restoration()`. While acceptable per the requirements, this adds roughly ~4-5 seconds of overhead to the total wall-clock time compared to raw inference.
