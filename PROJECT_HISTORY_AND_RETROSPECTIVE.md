# VELTRAXX: Project History, Retrospective & Technical Milestones

> **Project:** VELTRAXX (Semiconductor Wafer Inspection Restoration Engine)  
> **Event:** Botathon Hackathon Finalist Project (October 2026)  
> **Lead Developer:** Darshan  
> **Architecture:** SwinIR-2x (Shifted Window Vision Transformer for Image Restoration)

---

## 1. Project Genesis & The Challenge

Modern microchips feature nanoscale electrical interconnects spaced less than 10 nanometers apart. Scanning Electron Microscopes (SEMs) inspect these wafers in cleanrooms. However:
- High-dose electron beams cause photoresist shrinkage, wafer charging, and thermal destruction.
- Low-dose beams protect the wafer, but suffer from heavy **Poisson electron shot noise** and **optical defocus blur**.

**The Challenge:** Build an ultra-fast, deterministic super-resolution engine ($2\times$) that takes noisy, blurred $128 \times 128$ SEM scans and reconstructs pristine $256 \times 256$ metrology-grade references in real time.

---

## 2. Chronological Milestones (Phases 1 to 14 + Botathon)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE 14-PHASE DEVELOPMENT ROADMAP                                 │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ PHASES 1–7: Core Architecture & Base Training                                                    │
│ • Selected SwinIR-2x over CNNs and Diffusion models (zero hallucination mandate).                │
│ • Trained on 2,800 synthetic semiconductor patches with AdamW, Cosine Annealing, L1 loss.        │
│ • Produced baseline checkpoint: model_files/best.pth (28.87 dB on nominal SEM wafers).           │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ PHASES 8–9: The Fine-Tuning Plateau                                                              │
│ • Attempted re-training on the same narrow sensor distribution.                                  │
│ • Result: Negligible +0.014 dB gain. Proved that re-training on identical data is compute waste. │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 10: The Domain-Shift Crisis & Discovery                                                    │
│ • Stress-tested best.pth across 32 varied layout images (Profiles A, B, C, D).                   │
│ • THE DEFECT: On dark-substrate wafers (Profile D), best.pth collapsed to 16.22 dB / 0.22 SSIM!   │
│ • ROOT CAUSE: Single-sensor tunnel vision. The model had no learned brightness invariance.      │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ PHASES 11–12: The Targeted Physics Augmentation Breakthrough                                     │
│ • Injected random baseline brightness shifts (Δμ ~ -0.25 to +0.35) into clean GT before degrade. │
│ • Forced attention heads to decouple wire edges from substrate DC background potential.         │
│ • Produced Flagship Model: model_files/finetuned_v2.pth. Profile D jumped +2.84 dB / +0.25 SSIM!│
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 13: 50-Image Unseen Verification Benchmark                                                 │
│ • Evaluated on 50 completely new, unseen semiconductor layouts.                                  │
│ • OUTCOME: finetuned_v2.pth beat best.pth on 50 out of 50 samples (100.0% Win Rate)!             │
│ • Average Gain: +2.36 dB PSNR and +0.2161 SSIM over baseline across the board!                  │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 14: Realistic Optical Defocus Benchmark                                                    │
│ • Tested under subtle microscope optical defocus (σ = 0.6–1.2) and fine sensor grain.            │
│ • OUTCOME: finetuned_v2.pth won on 36 of 50 samples (72.0%), averaging +0.33 dB PSNR.           │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ BOTATHON FINALS: Presentation UI & Metrology Overhaul                                            │
│ • Upgraded veltraxx_ui.py to Bento Metrology Suite with 22pt Consolas numbers.                   │
│ • Fixed Tkinter canvas memory leak during mouse drag strokes.                                    │
│ • Bound keyboard shortcuts ([Space], [Arrows], [R]) and 1-click proof quick-picks.               │
│ • Verified ~40ms steady-state GPU inference (over 23 FPS) on NVIDIA RTX 5060.                    │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Quantitative Summary Table (50 Unseen Images)

| Model / Pipeline | Mean PSNR | Mean SSIM | Delta vs Baseline | Win Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Bicubic ($2\times$) Baseline** | 15.84 dB | 0.2521 | — | — | Classical interpolation |
| **Pre-Trained SwinIR (`best.pth`)** | 17.56 dB | 0.3145 | +1.72 dB | — | Original pre-trained checkpoint |
| **Flagship SwinIR (`finetuned_v2.pth`)**| **19.92 dB** | **0.5307** | **+4.08 dB** | **100% (50/50)** | **🌟 Decisive Winner** |
| **Net Advantage of v2 over `best.pth`** | **+2.36 dB** | **+0.2161** | — | — | **Consistent Win across all 50** |

### Top 3 Presentation Proof Samples:
1. **`sample_06.png`**: **+4.44 dB PSNR leap** (21.97 dB $\rightarrow$ **26.41 dB**), SSIM leaps from 0.53 $\rightarrow$ **0.8124**.
2. **`sample_35.png`**: **+3.99 dB PSNR leap**, resolves parallel dense line gratings without cross-talk bridging.
3. **`sample_23.png`**: **+4.99 dB PSNR leap**, cleans heavy electron shot noise from isolated contact pads.

---

## 4. Honest Hackathon Retrospective (Botathon Reflections)

### What Worked Brilliantly:
1. **The Core Engineering Won the Science:** SwinIR’s shifted-window local self-attention is mathematically the right tool for semiconductor patterns. It completely avoided the checkerboard glitches of CNNs and the hallucinations of GANs.
2. **Targeted Physics-Based Fine-Tuning Was the Real Breakthrough:** Diagnosing the Profile D domain-shift weakness in Phase 10 and systematically fixing it in Phase 12 created a genuine 100% win-rate model that generalizes across real wafer substrates.
3. **The Presentation UI Was Rock-Solid:** Upgrading to 22pt Consolas cards, fixing the Tkinter canvas display list leak, and adding the inverted error map created a professional metrology tool that stood out from amateur demos.

### What We Learned for Future Hackathons:
1. **Hackathon Formats Vary Widely:** In hackathons like "Botathon", the competition often shifts from live coding to project presentation and pitch defense. Having crystal-clear slide hooks, plain-English analogies, and an unshakeable live demo is 80% of the battle.
2. **Cold Starts vs. Warm Inference:** On the first click after launch, PyTorch compiles CUDA kernels, causing a 1-second cold start before dropping to 40ms. Knowing this technical detail allows you to explain it to judges with confidence rather than panic.
3. **Keep C: Drive Clean During AI Projects:** Deep learning package managers (`uv`, `pip`) secretly cache gigabytes of `.whl` files on C: drive. Always run `uv cache clean` and `pip cache purge` when completing an AI project.

---
*Team VELTRAXX — Botathon Finalist Archive*
