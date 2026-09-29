import streamlit as st
import os
import time
import numpy as np
from PIL import Image
from pathlib import Path
import shutil

from demo_single import run_restoration, find_matching_gt

st.set_page_config(layout="wide", page_title="Semiconductor Restoration")

st.title("Semiconductor Inspection Image Restoration")
st.write("SwinIR denoising + 2x super-resolution")

def save_uploaded_files(input_file, gt_file=None):
    temp_dir = Path("ui_outputs/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = temp_dir / input_file.name
    with open(input_path, "wb") as f:
        f.write(input_file.getbuffer())
        
    gt_path = None
    if gt_file is not None:
        gt_path = temp_dir / gt_file.name
        with open(gt_path, "wb") as f:
            f.write(gt_file.getbuffer())
            
    return input_path, gt_path

def load_for_display(path):
    if str(path).endswith(".npy"):
        array = np.load(path).astype(np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array.squeeze(0)
        img = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    else:
        img = Image.open(path).convert("L")
    return img

def format_metric(val):
    if val is None:
        return "N/A: no ground truth"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)

def format_metric_delta(val, baseline):
    if val is None or baseline is None:
        return None
    return f"{val - baseline:+.4f}"

# Scan input_custom directory
input_custom_dir = Path("input_custom")
available_examples = []
if input_custom_dir.exists():
    for f in sorted(os.listdir(input_custom_dir)):
        # Skip ground truth files in the listing
        if "_gt" in f:
            continue
        available_examples.append(f)

def get_label(filename):
    if filename == "semicon_test_pattern.png":
        return f"{filename} - out-of-distribution / new image (no ground truth)"
    elif filename == "degraded_semicon_lowres.png":
        return f"{filename} - (out-of-distribution) Note: model can do worse than bicubic on noise-free low-res input."
    elif filename.endswith(".npy"):
        return f"{filename} - raw dataset sample, no ground truth in repo"
    else:
        return f"{filename} - (out-of-distribution)"

mode = st.radio("Mode", ["Choose an example", "Upload your own"], horizontal=True)

selected_input_path = None
selected_gt_path = None
is_ood = False

if mode == "Choose an example":
    if not available_examples:
        st.warning("No examples found in input_custom/ directory.")
    else:
        options_dict = {get_label(f): f for f in available_examples}
        selected_label = st.selectbox("Select Image", list(options_dict.keys()))
        selected_file = options_dict[selected_label]
        selected_input_path = input_custom_dir / selected_file
        
        # Check if it's OOD
        if "out-of-distribution" in selected_label:
            is_ood = True
            
else:
    input_upload = st.file_uploader("Upload Degraded Input", type=["png", "jpg", "jpeg", "npy"])
    gt_upload = st.file_uploader("Upload Ground Truth (Optional)", type=["png", "jpg", "jpeg", "npy"])
    
    if input_upload is not None:
        selected_input_path, selected_gt_path = save_uploaded_files(input_upload, gt_upload)

if st.button("Restore", type="primary", use_container_width=True):
    if selected_input_path is None:
        st.error("Please provide an input image first.")
    else:
        with st.spinner("Restoring image..."):
            try:
                # Prepare output directory
                out_dir = Path("ui_outputs/results")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{selected_input_path.stem}_restored.png"
                
                # Check for GT
                explicit_gt = str(selected_gt_path) if selected_gt_path else None
                
                start_time = time.time()
                result_dict = run_restoration(
                    input_path=str(selected_input_path),
                    output_path=str(out_path),
                    gt_path=explicit_gt
                )
                end_time = time.time()
                
                result_dict["total_wall_time"] = end_time - start_time
                
                # Find the matched GT path to display it
                matched_gt_path = None
                try:
                    matched_gt_path = find_matching_gt(selected_input_path, explicit_gt)
                except Exception:
                    pass
                    
                result_dict["matched_gt_path"] = matched_gt_path
                st.session_state["last_result"] = result_dict
                st.session_state["is_ood"] = is_ood
                
            except Exception as e:
                st.error(f"Error during restoration: {str(e)}")

# Display results if available in session state
if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    is_ood_flag = st.session_state.get("is_ood", False)
    
    if is_ood_flag:
        st.info("Notice: Metrics on out-of-distribution selections are expected to be lower than in-distribution results, as documented in the README.")
        
    st.markdown("### Images")
    col1, col2, col3 = st.columns(3)
    
    # Render sizes
    DISPLAY_SIZE = (512, 512)
    
    # 1. Input
    try:
        in_img = load_for_display(res["input"])
        orig_in_size = in_img.size
        in_img_disp = in_img.resize(DISPLAY_SIZE, Image.NEAREST)
        with col1:
            st.image(in_img_disp, caption=f"Degraded Input ({orig_in_size[0]}x{orig_in_size[1]}) - shown enlarged", use_container_width=True)
    except Exception as e:
        with col1:
            st.error(f"Failed to load input: {e}")
            
    # 2. Output
    try:
        out_img = load_for_display(res["output"])
        orig_out_size = out_img.size
        out_img_disp = out_img.resize(DISPLAY_SIZE, Image.NEAREST)
        with col2:
            st.image(out_img_disp, caption=f"Restored Output ({orig_out_size[0]}x{orig_out_size[1]}) - shown enlarged", use_container_width=True)
    except Exception as e:
        with col2:
            st.error(f"Failed to load output: {e}")
            
    # 3. Ground Truth
    with col3:
        if res.get("matched_gt_path") and Path(res["matched_gt_path"]).exists():
            try:
                gt_img = load_for_display(res["matched_gt_path"])
                orig_gt_size = gt_img.size
                gt_img_disp = gt_img.resize(DISPLAY_SIZE, Image.NEAREST)
                st.image(gt_img_disp, caption=f"Ground Truth ({orig_gt_size[0]}x{orig_gt_size[1]}) - shown enlarged", use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load ground truth: {e}")
        else:
            st.info("No ground truth available for this sample")
            
    # Metrics
    st.markdown("### Metrics")
    m1, m2, m3 = st.columns(3)
    
    psnr_val = res.get("psnr")
    ssim_val = res.get("ssim")
    bic_psnr = res.get("bicubic_psnr")
    bic_ssim = res.get("bicubic_ssim")
    
    with m1:
        st.metric(
            label="SwinIR PSNR (dB)", 
            value=format_metric(psnr_val), 
            delta=format_metric_delta(psnr_val, bic_psnr)
        )
    with m2:
        st.metric(
            label="SwinIR SSIM", 
            value=format_metric(ssim_val), 
            delta=format_metric_delta(ssim_val, bic_ssim)
        )
    with m3:
        inf_time = res.get("inference_time_s", 0)
        st.metric(
            label="Inference Time", 
            value=f"{inf_time:.4f} s"
        )
        
    st.markdown("---")
    sm1, sm2, sm3 = st.columns(3)
    with sm1:
        st.write(f"**Bicubic PSNR:** {format_metric(bic_psnr)}")
    with sm2:
        st.write(f"**Bicubic SSIM:** {format_metric(bic_ssim)}")
    with sm3:
        # read device from demo_single.py if possible, or fallback to CPU since we run purely inference wrapper
        # demo_single returns device in print but not explicitly in dict except implicitly through inference
        # Wait, run_restoration doesn't return the device in the dict. We can just check torch.cuda.is_available() ourselves here
        import torch
        dev_used = "CUDA" if torch.cuda.is_available() else "CPU"
        st.write(f"**Device Used:** {dev_used}")
        st.write(f"**Total time incl. model load:** {res.get('total_wall_time', 0):.4f} s")
        
    st.caption("Values computed live by run_restoration() for this exact run.")
