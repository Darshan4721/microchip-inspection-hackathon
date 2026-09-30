from streamlit.testing.v1 import AppTest
import os
from demo_single import run_restoration, find_matching_gt
from pathlib import Path
import sys

def test_stale_results_fix():
    print("Testing stale results fix (Task 1)...")
    at = AppTest.from_file("app.py", default_timeout=300).run()
    
    print("Testing default example (Task 7)...")
    default_val = at.selectbox[0].value
    print(f"Default selectbox value: {default_val}")
    if "degraded_semicon_gaussian.png" not in default_val:
        print("FAIL: Default is not degraded_semicon_gaussian.png")
        sys.exit(1)
    
    if "no ground truth" in default_val:
        print("FAIL: Default sample does not have ground truth")
        sys.exit(1)
        
    at.button[0].click().run()
    
    found_info = [elem.value for elem in at.info]
    print(f"Info boxes after restore: {found_info}")
    
    if any("Click Restore to run on this image" in i for i in found_info):
        print("FAIL: 'Click Restore to run on this image' should not be present after restoring.")
        sys.exit(1)
        
    options = at.selectbox[0].options
    other_option = None
    for opt in options:
        if opt != default_val:
            other_option = opt
            break
            
    print(f"Changing selection to: {other_option}")
    at.selectbox[0].set_value(other_option).run()
    
    found_info_after = [elem.value for elem in at.info]
    print(f"Info boxes after change: {found_info_after}")
    
    if not any("Click Restore to run on this image" in i for i in found_info_after):
        print("FAIL: 'Click Restore to run on this image' is MISSING after changing selection.")
        sys.exit(1)
    
    print("PASS: Stale results are correctly handled.\n")

def test_values_match_backend():
    print("Testing if UI values match backend values (Task 3)...")
    at = AppTest.from_file("app.py", default_timeout=300).run()
    
    target_option = None
    for opt in at.selectbox[0].options:
        if "degraded_semicon_gaussian.png" in opt:
            target_option = opt
            break
            
    at.selectbox[0].set_value(target_option).run()
    at.button[0].click().run()
    
    psnr_metric = at.metric[0].value
    ssim_metric = at.metric[1].value
    
    print(f"UI PSNR: {psnr_metric}")
    print(f"UI SSIM: {ssim_metric}")
    
    input_path = "input_custom/degraded_semicon_gaussian.png"
    out_dir = Path("ui_outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_backend_restored.png"
    
    res = run_restoration(input_path, str(out_path))
    backend_psnr = f"{res['psnr']:.4f}"
    backend_ssim = f"{res['ssim']:.4f}"
    
    print(f"Backend PSNR: {backend_psnr}")
    print(f"Backend SSIM: {backend_ssim}")
    
    # tolerance
    if abs(float(psnr_metric) - float(backend_psnr)) > 0.01 or abs(float(ssim_metric) - float(backend_ssim)) > 0.01:
        print("FAIL: UI metrics do not match backend metrics.")
        sys.exit(1)
    
    print("PASS: UI metrics match backend metrics.\n")

if __name__ == "__main__":
    test_stale_results_fix()
    test_values_match_backend()
    print("All smoke tests passed!")
