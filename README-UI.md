# Semiconductor Inspection Image Restoration (UI Demo)

This is a Streamlit user interface built on top of the original SwinIR restoration backend. The backend is invoked unchanged.

## Setup & Running

1. **Install Requirements**
   For a fresh machine, install the UI dependencies (CPU-only PyTorch is sufficient for inference):
   ```bash
   pip install -r requirements-ui.txt
   ```

2. **Run the App**
   ```bash
   python -m streamlit run app.py
   ```
   *Note: Ensure you invoke the correct Python path (e.g., `python`, `python3`, or your specific environment binary) when running the above command.*

## Features & Notes
- **First Click Slower**: The first inference run may be slightly slower due to PyTorch model initialization and warm-up.
- **CPU Runtime**: The restoration logic takes a few seconds per image on a standard CPU (a few seconds). 
- **Original Backend**: The UI directly calls the existing `demo_single.py` restoration logic without modifying the underlying models, metrics, or validation flow.

## Included Examples
The `input_custom/` folder provides the following pre-loaded examples:
- `degraded_semicon_gaussian.png`, `degraded_semicon_lowres.png`, `degraded_semicon_speckle.png`, `real_semicon_die.png`: These represent out-of-distribution (OOD) degraded semiconductor samples. to demonstrate structural similarity improvements on noisy unseen samples.
- `semicon_test_pattern.png`: An out-of-distribution test pattern with NO ground truth.
- `000005.npy`, `000150.npy`, `000300.npy`: Raw dataset samples with no explicit ground truth provided in the repository.

## Troubleshooting
- **Missing dependencies**: Ensure `streamlit` and all requirements in `requirements-ui.txt` are installed. If you encounter `ModuleNotFoundError: No module named 'torch'`, you need to install PyTorch.
- **Out of memory**: The 2x SwinIR super-resolution processes $128 \times 128 \rightarrow 256 \times 256$. While lightweight, ensure you have a few gigabytes of RAM available.
- **File Not Found**: Ensure you launch the app from the root of the repository so the `input_custom/` folder and `model_files/` directory are discoverable.

## Important Repository Rules
**Do NOT modify the following tracked files and folders in this branch:**
- `README.md`
- `demo_single.py`
- `src/` directory
- `model_files/` directory

The UI was built specifically to sit on top of the original backend without modifying its core functionality.
