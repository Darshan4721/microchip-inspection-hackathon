from pathlib import Path
import time
import numpy as np
import torch
from demo_single import run_restoration

files = sorted(list(Path("test_batch_20").glob("*.*")))
assert len(files) == 20, f"Expected 20 files, found {len(files)}"

def benchmark_device(device_name):
    print(f"\n" + "=" * 65)
    print(f"  RUNNING 20-IMAGE SPEED BENCHMARK ON {device_name.upper()}")
    print("=" * 65)
    
    # Warmup on first sample
    _ = run_restoration(str(files[0]), output_path=f"results/speed_test/warmup.png", device=device_name)
    
    latencies = []
    for i, f in enumerate(files):
        out_dest = f"results/speed_test/{device_name}_{f.stem}.png"
        res = run_restoration(str(f), output_path=out_dest, device=device_name)
        lat = res["inference_time_s"]
        latencies.append(lat)
        print(f"[{i+1:02d}/20] {f.name:<20} -> {lat*1000:.2f} ms ({lat:.4f} s)")

    avg_time = np.mean(latencies)
    min_time = np.min(latencies)
    max_time = np.max(latencies)
    throughput = 1.0 / avg_time
    
    print("-" * 65)
    print(f"RESULTS FOR {device_name.upper()}:")
    print(f"  Average Time : {avg_time:.4f} s ({avg_time*1000:.1f} ms)")
    print(f"  Fastest Time : {min_time:.4f} s ({min_time*1000:.1f} ms)")
    print(f"  Slowest Time : {max_time:.4f} s ({max_time*1000:.1f} ms)")
    print(f"  Throughput   : {throughput:.2f} images/sec")
    print("=" * 65)
    return avg_time, min_time, max_time

# Run on GPU
gpu_avg, gpu_min, gpu_max = benchmark_device("cuda")

# Run on CPU
cpu_avg, cpu_min, cpu_max = benchmark_device("cpu")

print("\n" + "=" * 70)
print("FINAL SPEED SUMMARY TABLE (REAL MEASURED NUMBERS)")
print("=" * 70)
print(f"| Device | Hardware Name | Average Time | Fastest (Min) | Slowest (Max) |")
print(f"| :--- | :--- | :---: | :---: | :---: |")
print(f"| GPU | NVIDIA GeForce RTX 5060 Laptop GPU | {gpu_avg*1000:.1f} ms ({gpu_avg:.4f} s) | {gpu_min*1000:.1f} ms | {gpu_max*1000:.1f} ms |")
print(f"| CPU | Intel Host Processor | {cpu_avg*1000:.1f} ms ({cpu_avg:.4f} s) | {cpu_min*1000:.1f} ms | {cpu_max*1000:.1f} ms |")
print("=" * 70)
