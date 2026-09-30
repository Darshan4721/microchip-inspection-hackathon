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

def test_mismatched_upload():
    print("Testing mismatched uploads (Task 1)...")
    from streamlit.testing.v1 import AppTest
    from PIL import Image
    import os
    import sys
    
    os.makedirs("ui_outputs/edge_cases", exist_ok=True)
    img1 = Image.new("L", (512, 512), color=128)
    img1.save("ui_outputs/edge_cases/test_512.png")
    img2 = Image.new("L", (100, 100), color=128)
    img2.save("ui_outputs/edge_cases/test_100.png")

    at = AppTest.from_file("app.py", default_timeout=300).run()
    at.radio[0].set_value("Upload your own").run()

    with open("ui_outputs/edge_cases/test_512.png", "rb") as f:
        at.file_uploader[0].set_value([("test_512.png", f.read(), "image/png")]).run()

    with open("ui_outputs/edge_cases/test_100.png", "rb") as f2:
        at.file_uploader[1].set_value([("test_100.png", f2.read(), "image/png")]).run()

    at.button[0].click().run()
    
    infos = [i.value for i in at.info]
    print("INFOS found:", infos)
    
    found_in = any("Inputs are resized to 128x128" in i for i in infos)
    found_gt = any("was resized to 256x256 using Bicubic interpolation" in i for i in infos)
    
    if not found_in or not found_gt:
        print("FAIL: Missing info messages for resized uploads!")
        sys.exit(1)
        
    print("PASS: Mismatched uploads correctly documented in UI.\n")

if __name__ == "__main__":
    test_mismatched_upload()
    test_stale_results_fix()
    test_values_match_backend()
    print("All smoke tests passed!")
