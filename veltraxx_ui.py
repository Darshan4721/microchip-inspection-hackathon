"""
Whale Tracks — Semiconductor Image Restoration
Presentation Desktop Application
"""

import os
import sys
import time
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import torch
import torch.nn.functional as F

# Ensure local imports work
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import create_model
from src.utils import calculate_psnr, calculate_ssim

# Dark Mode Palette (Tailwind / Slate / Ice Blue)
BG_MAIN = "#0B0E14"          # Dark slate canvas
CARD_BG = "#131722"          # Surface card
CARD_BORDER = "#21283B"      # Subtle border
HEADER_BG = "#0F131D"        # Top bar
TEXT_PRIMARY = "#F1F5F9"     # High contrast white
TEXT_SECONDARY = "#94A3B8"   # Slate gray
TEXT_MUTED = "#64748B"       # Subtle slate
ACCENT_BLUE = "#38BDF8"      # Ice cyan / Sky 400
ACCENT_GREEN = "#10B981"     # Emerald 500
ACCENT_AMBER = "#F59E0B"     # Amber 500
ACCENT_RED = "#EF4444"       # Rose 500
BUTTON_BG = "#1E2538"        # Interactive button
BUTTON_HOVER = "#2D3748"
ACTIVE_CTA = "#0284C7"       # Vibrant Action CTA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLAGSHIP_CKPT = ROOT / "model_files" / "finetuned_v2.pth"
BASELINE_CKPT = ROOT / "model_files" / "best.pth"
SAMPLES_DIR = ROOT / "final_test_50" / "clean"


class VeltraxxApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VELTRAXX — Semiconductor Image Restoration Platform")
        self.root.geometry("1320x840")
        self.root.minsize(1180, 720)
        self.root.configure(bg=BG_MAIN)

        # Bring window to foreground
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(800, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()

        import queue
        self.ui_queue = queue.Queue()
        self.poll_queue()

        # Application State
        self.clean_gt_img = None       # PIL Image (256x256)
        self.clean_gt_arr = None       # float32 [0, 1] (256x256)
        self.damaged_img = None        # PIL Image (256x256 canvas resolution)
        self.restored_img = None       # PIL Image (256x256)
        self.heatmap_img = None        # PIL Image (256x256)

        self.has_manual_edits = False
        self.last_draw_x = None
        self.last_draw_y = None
        self.brush_size = 14
        self.erase_color_val = 0       # 0 = dark substrate, 255 = open line defect

        self.is_processing = False
        self.model = None
        self.active_model_name = "finetuned_v2"

        # Load samples list
        self.sample_files = sorted(list(SAMPLES_DIR.glob("sample_*.png")))
        if not self.sample_files:
            # Fallback to any pngs in final_test_50
            self.sample_files = sorted(list(ROOT.glob("final_test_50/**/*.png")))

        # Initialize Model in background
        self.init_model_thread()

        # Build UI Structure
        self.build_header()
        self.build_controls()
        self.build_quad_panels()
        self.build_status_bar()

        # Bind Global Keyboard Shortcuts
        self.root.bind("<space>", lambda e: self.on_restore_clicked())
        self.root.bind("<Left>", lambda e: self.cycle_sample(-1))
        self.root.bind("<Right>", lambda e: self.cycle_sample(1))
        self.root.bind("<r>", lambda e: self.reset_to_degraded())
        self.root.bind("<R>", lambda e: self.reset_to_degraded())

        # Load initial sample
        if self.sample_files:
            self.load_sample_by_path(self.sample_files[0])

    def poll_queue(self):
        try:
            while True:
                fn, args = self.ui_queue.get_nowait()
                fn(*args)
        except Exception:
            pass
        self.root.after(25, self.poll_queue)

    def dispatch_to_ui(self, fn, *args):
        self.ui_queue.put((fn, args))

    # ---------------------------------------------------------
    # Model Loading
    # ---------------------------------------------------------
    def init_model_thread(self):
        def _load():
            try:
                ckpt_path = FLAGSHIP_CKPT if FLAGSHIP_CKPT.exists() else BASELINE_CKPT
                m = create_model().to(DEVICE)
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                m.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
                m.eval()
                self.model = m
                self.dispatch_to_ui(self.on_model_ready)
            except Exception as e:
                print(f"Error loading model: {e}")
                self.dispatch_to_ui(lambda: self.status_pill.configure(text="⚠️ Model Load Failed", fg=ACCENT_RED))

        t = threading.Thread(target=_load, daemon=True)
        t.start()

    def on_model_ready(self):
        ckpt_name = "finetuned_v2.pth (Reflectance-Tuned)" if FLAGSHIP_CKPT.exists() else "best.pth"
        self.status_pill.configure(text=f"● Ready ({DEVICE.upper()})", fg=ACCENT_GREEN)
        self.model_info_lbl.configure(text=f"Engine: VELTRAXX SwinIR-2x | Model: {ckpt_name}")

    # ---------------------------------------------------------
    # Header Bar
    # ---------------------------------------------------------
    def build_header(self):
        header_frame = tk.Frame(self.root, bg=HEADER_BG, pady=10, padx=20, bd=0)
        header_frame.pack(fill="x", side="top")

        left_brand = tk.Frame(header_frame, bg=HEADER_BG)
        left_brand.pack(side="left")

        title_label = tk.Label(
            left_brand,
            text="VELTRAXX",
            font=("Segoe UI", 15, "bold"),
            fg=ACCENT_BLUE,
            bg=HEADER_BG
        )
        title_label.pack(side="left")

        sep = tk.Label(left_brand, text="|", font=("Segoe UI", 13), fg=TEXT_MUTED, bg=HEADER_BG, padx=10)
        sep.pack(side="left")

        sub_label = tk.Label(
            left_brand,
            text="Semiconductor Inspection Image Restoration Platform",
            font=("Segoe UI", 11, "bold"),
            fg=TEXT_SECONDARY,
            bg=HEADER_BG
        )
        sub_label.pack(side="left")

        # Right Telemetry
        right_telemetry = tk.Frame(header_frame, bg=HEADER_BG)
        right_telemetry.pack(side="right")

        self.model_info_lbl = tk.Label(
            right_telemetry,
            text="Engine: Loading Whale Tracks SwinIR weights...",
            font=("Segoe UI", 9),
            fg=TEXT_MUTED,
            bg=HEADER_BG,
            padx=10
        )
        self.model_info_lbl.pack(side="left")

        self.status_pill = tk.Label(
            right_telemetry,
            text="● Initializing...",
            font=("Segoe UI", 10, "bold"),
            fg=ACCENT_AMBER,
            bg="#171C2B",
            padx=12,
            pady=3,
            relief="solid",
            bd=1
        )
        self.status_pill.pack(side="left", padx=5)

    # ---------------------------------------------------------
    # Controls Toolbar (Bento Grid Style)
    # ---------------------------------------------------------
    def build_controls(self):
        ctrl_frame = tk.Frame(self.root, bg=BG_MAIN, pady=6, padx=16)
        ctrl_frame.pack(fill="x", side="top")

        # Section 1: Image Picker + Showcase Quick-Picks
        s1 = tk.LabelFrame(ctrl_frame, text=" 1. Ground Truth Source ", font=("Segoe UI", 9, "bold"), fg=TEXT_SECONDARY, bg=CARD_BG, bd=1, relief="solid", padx=8, pady=4)
        s1.pack(side="left", fill="y", padx=4)

        sample_names = [f.name for f in self.sample_files] if self.sample_files else ["No samples found"]
        self.sample_combo = ttk.Combobox(s1, values=sample_names, state="readonly", width=14, font=("Segoe UI", 9))
        if sample_names:
            self.sample_combo.current(0)
        self.sample_combo.bind("<<ComboboxSelected>>", self.on_sample_combo)
        self.sample_combo.pack(side="left", padx=3)

        btn_browse = tk.Button(s1, text="📁 Browse", command=self.on_browse_file, bg=BUTTON_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 8), relief="flat", padx=6, pady=2, cursor="hand2")
        btn_browse.pack(side="left", padx=3)

        # Quick-Pick Buttons for Flagship Samples (sample_06, sample_35, sample_23)
        qp_frame = tk.Frame(s1, bg=CARD_BG)
        qp_frame.pack(side="left", padx=(8, 2))

        tk.Label(qp_frame, text="Quick-Pick:", font=("Segoe UI", 8, "bold"), fg=ACCENT_AMBER, bg=CARD_BG).pack(side="left", padx=(0, 4))

        showcase_samples = [("sample_06", "S-06 (+4.4dB)"), ("sample_35", "S-35 (+4.0dB)"), ("sample_23", "S-23 (+5.0dB)")]
        for s_id, s_lbl in showcase_samples:
            btn_s = tk.Button(
                qp_frame,
                text=s_lbl,
                command=lambda name=s_id: self.select_sample_by_name(name),
                bg=BUTTON_BG,
                fg=TEXT_PRIMARY,
                font=("Segoe UI", 8, "bold"),
                relief="flat",
                padx=6,
                pady=1,
                cursor="hand2",
                activebackground=ACCENT_BLUE
            )
            btn_s.pack(side="left", padx=2)

        # Section 2: Optical Degradation Sliders (Defaults: Blur 1.0, Noise 6%)
        s2 = tk.LabelFrame(ctrl_frame, text=" 2. Optical Degradation Sliders ", font=("Segoe UI", 9, "bold"), fg=TEXT_SECONDARY, bg=CARD_BG, bd=1, relief="solid", padx=8, pady=4)
        s2.pack(side="left", fill="y", padx=4)

        # Defocus blur slider (default 1.0)
        blur_box = tk.Frame(s2, bg=CARD_BG)
        blur_box.pack(side="left", padx=4)
        self.blur_label = tk.Label(blur_box, text="Blur σ: 1.0", font=("Segoe UI", 8, "bold"), fg=TEXT_PRIMARY, bg=CARD_BG)
        self.blur_label.pack(anchor="w")
        self.blur_slider = tk.Scale(blur_box, from_=0.0, to=1.5, resolution=0.1, orient="horizontal", length=80, showvalue=0, command=self.on_slider_change, bg=CARD_BG, fg=TEXT_PRIMARY, highlightthickness=0, troughcolor=BG_MAIN)
        self.blur_slider.set(1.0)
        self.blur_slider.pack()

        # Noise grain slider (default 0.06 = 6%)
        noise_box = tk.Frame(s2, bg=CARD_BG)
        noise_box.pack(side="left", padx=4)
        self.noise_label = tk.Label(noise_box, text="Noise: 6%", font=("Segoe UI", 8, "bold"), fg=TEXT_PRIMARY, bg=CARD_BG)
        self.noise_label.pack(anchor="w")
        self.noise_slider = tk.Scale(noise_box, from_=0.0, to=0.12, resolution=0.01, orient="horizontal", length=80, showvalue=0, command=self.on_slider_change, bg=CARD_BG, fg=TEXT_PRIMARY, highlightthickness=0, troughcolor=BG_MAIN)
        self.noise_slider.set(0.06)
        self.noise_slider.pack()

        btn_box = tk.Frame(s2, bg=CARD_BG)
        btn_box.pack(side="left", padx=4)

        btn_apply_deg = tk.Button(btn_box, text="Apply Degradation", command=self.apply_slider_degradation, bg=BUTTON_BG, fg=ACCENT_BLUE, font=("Segoe UI", 8, "bold"), relief="flat", padx=6, pady=2, cursor="hand2")
        btn_apply_deg.pack(fill="x", pady=1)

        btn_preset = tk.Button(btn_box, text="Realistic Preset (1.0/6%)", command=self.set_realistic_preset, bg=BUTTON_BG, fg=ACCENT_GREEN, font=("Segoe UI", 8), relief="flat", padx=6, pady=1, cursor="hand2")
        btn_preset.pack(fill="x", pady=1)

        # Section 3: Interactive Defect Eraser Tool
        s3 = tk.LabelFrame(ctrl_frame, text=" 3. Interactive Eraser Tool ", font=("Segoe UI", 9, "bold"), fg=TEXT_SECONDARY, bg=CARD_BG, bd=1, relief="solid", padx=8, pady=4)
        s3.pack(side="left", fill="y", padx=4)

        tk.Label(s3, text="Brush:", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=CARD_BG).pack(side="left")
        self.brush_var = tk.StringVar(value="Med")
        for b_name, b_val in [("Sml", 8), ("Med", 16), ("Lrg", 28)]:
            b_btn = tk.Radiobutton(s3, text=b_name, value=b_name, variable=self.brush_var, command=lambda v=b_val: self.set_brush_size(v), bg=CARD_BG, fg=TEXT_PRIMARY, selectcolor=BUTTON_BG, activebackground=CARD_BG, font=("Segoe UI", 8))
            b_btn.pack(side="left")

        btn_reset_edits = tk.Button(s3, text="↺ Reset Edits [R]", command=self.reset_to_degraded, bg=BUTTON_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 8), relief="flat", padx=6, pady=2, cursor="hand2")
        btn_reset_edits.pack(side="left", padx=6)

        # Section 4: Primary Restore CTA
        s4 = tk.Frame(ctrl_frame, bg=BG_MAIN)
        s4.pack(side="right", fill="y", padx=4)

        self.btn_restore = tk.Button(
            s4,
            text="RESTORE IMAGE [SPACE]",
            command=self.on_restore_clicked,
            bg=ACTIVE_CTA,
            fg="#FFFFFF",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2",
            activebackground=ACCENT_BLUE
        )
        self.btn_restore.pack(fill="both", expand=True)

    def select_sample_by_name(self, target_stem):
        for idx, p in enumerate(self.sample_files):
            if target_stem in p.stem:
                self.sample_combo.current(idx)
                self.load_sample_by_path(p)
                return

    def cycle_sample(self, direction):
        if not self.sample_files:
            return
        curr_idx = self.sample_combo.current()
        if curr_idx < 0:
            curr_idx = 0
        new_idx = (curr_idx + direction) % len(self.sample_files)
        self.sample_combo.current(new_idx)
        self.load_sample_by_path(self.sample_files[new_idx])

    def set_brush_size(self, size):
        self.brush_size = size

    def set_realistic_preset(self):
        self.blur_slider.set(1.0)
        self.noise_slider.set(0.06)
        self.on_slider_change(None)
        self.apply_slider_degradation()

    def on_slider_change(self, _):
        b_val = self.blur_slider.get()
        n_val = self.noise_slider.get()
        self.blur_label.configure(text=f"Blur σ: {b_val:.1f}")
        self.noise_label.configure(text=f"Noise: {int(n_val*100)}%")

    # ---------------------------------------------------------
    # Main Quad Display (4 Panels Side-by-Side)
    # ---------------------------------------------------------
    def build_quad_panels(self):
        quad_frame = tk.Frame(self.root, bg=BG_MAIN, padx=14, pady=4)
        quad_frame.pack(fill="both", expand=True)

        for col in range(4):
            quad_frame.grid_columnconfigure(col, weight=1, uniform="panels")
        quad_frame.grid_rowconfigure(0, weight=1)

        # Panel 1: Clean Ground Truth
        self.panel1, self.canvas_gt, self.sub1 = self.create_display_card(
            quad_frame, col=0, title="1. Ground Truth Target", subtitle="Clean High-Resolution Semiconductor Image (256x256)", badge_text="Pristine SEM Reference", badge_color="#38BDF8"
        )

        # Panel 2: Damaged / Sensor Input (Interactive Canvas!)
        self.panel2, self.canvas_damaged, self.sub2 = self.create_interactive_card(
            quad_frame, col=1, title="2. Damaged / Sensor Input", subtitle="Click & Drag to Erase Defect Areas Directly", badge_text="Interactive Canvas", badge_color=ACCENT_AMBER
        )

        # Panel 3: Restored Output
        self.panel3, self.canvas_restored, self.sub3 = self.create_display_card(
            quad_frame, col=2, title="3. SwinIR 2x Restored", subtitle="High-Fidelity Reconstructed Layout Pattern", badge_text="Restored Output", badge_color=ACCENT_GREEN
        )

        # Panel 4: Difference / Error Map
        self.panel4, self.canvas_heatmap, self.sub4 = self.create_display_card(
            quad_frame, col=3,
            title="4. Restoration Error Map",
            subtitle="Restoration Error Map: Dark = close to ground truth, Bright = remaining difference",
            badge_text="Error Map", badge_color="#F43F5E"
        )

    def create_display_card(self, parent, col, title, subtitle, badge_text, badge_color):
        card = tk.Frame(parent, bg=CARD_BG, bd=1, relief="solid", padx=10, pady=8)
        card.grid(row=0, column=col, padx=5, pady=4, sticky="nsew")

        # Header inside card
        th_frame = tk.Frame(card, bg=CARD_BG)
        th_frame.pack(fill="x", pady=2)

        t_lbl = tk.Label(th_frame, text=title, font=("Segoe UI", 10, "bold"), fg=TEXT_PRIMARY, bg=CARD_BG)
        t_lbl.pack(side="left")

        badge = tk.Label(th_frame, text=f" {badge_text} ", font=("Segoe UI", 8, "bold"), fg=badge_color, bg="#1A2133", bd=1, relief="solid")
        badge.pack(side="right")

        sub_lbl = tk.Label(card, text=subtitle, font=("Segoe UI", 8), fg=TEXT_MUTED, bg=CARD_BG, wraplength=270, justify="left")
        sub_lbl.pack(anchor="w", pady=(0, 6))

        # Canvas container (256x256)
        cv_frame = tk.Frame(card, bg="#000000", bd=1, relief="solid")
        cv_frame.pack(expand=True)

        canvas = tk.Canvas(cv_frame, width=256, height=256, bg="#000000", highlightthickness=0)
        canvas.pack()

        # Footer stat placeholder
        stat_lbl = tk.Label(card, text="Awaiting Input", font=("Segoe UI", 8, "bold"), fg=TEXT_SECONDARY, bg=CARD_BG, pady=4)
        stat_lbl.pack(side="bottom")

        return card, canvas, stat_lbl

    def create_interactive_card(self, parent, col, title, subtitle, badge_text, badge_color):
        card = tk.Frame(parent, bg=CARD_BG, bd=1, relief="solid", padx=10, pady=8)
        card.grid(row=0, column=col, padx=5, pady=4, sticky="nsew")

        th_frame = tk.Frame(card, bg=CARD_BG)
        th_frame.pack(fill="x", pady=2)

        t_lbl = tk.Label(th_frame, text=title, font=("Segoe UI", 10, "bold"), fg=TEXT_PRIMARY, bg=CARD_BG)
        t_lbl.pack(side="left")

        badge = tk.Label(th_frame, text=f" {badge_text} ", font=("Segoe UI", 8, "bold"), fg=badge_color, bg="#282218", bd=1, relief="solid")
        badge.pack(side="right")

        sub_lbl = tk.Label(card, text=subtitle, font=("Segoe UI", 8, "bold"), fg=ACCENT_AMBER, bg=CARD_BG, wraplength=270, justify="left")
        sub_lbl.pack(anchor="w", pady=(0, 6))

        cv_frame = tk.Frame(card, bg="#000000", bd=1, relief="solid")
        cv_frame.pack(expand=True)

        canvas = tk.Canvas(cv_frame, width=256, height=256, bg="#000000", highlightthickness=0, cursor="crosshair")
        canvas.pack()

        # Mouse Bindings for interactive erasing
        canvas.bind("<Button-1>", self.on_canvas_press)
        canvas.bind("<B1-Motion>", self.on_canvas_drag)
        canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        stat_lbl = tk.Label(card, text="Eraser Ready: Drag mouse to damage", font=("Segoe UI", 8, "bold"), fg=ACCENT_AMBER, bg=CARD_BG, pady=4)
        stat_lbl.pack(side="bottom")

        return card, canvas, stat_lbl

    # ---------------------------------------------------------
    # Bottom Telemetry & Status Bar (20pt+ High-Visibility Metrology Cards)
    # ---------------------------------------------------------
    def build_status_bar(self):
        bar = tk.Frame(self.root, bg=HEADER_BG, pady=10, padx=20)
        bar.pack(fill="x", side="bottom")

        # Telemetry container with 3 large Bento Stat Cards
        stats_frame = tk.Frame(bar, bg=HEADER_BG)
        stats_frame.pack(side="left", fill="y")

        # PSNR Card
        psnr_card = tk.Frame(stats_frame, bg=CARD_BG, bd=1, relief="solid", padx=16, pady=4)
        psnr_card.pack(side="left", padx=8)
        tk.Label(psnr_card, text="PSNR RECONSTRUCTION", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=CARD_BG).pack(anchor="w")
        self.telemetry_psnr = tk.Label(psnr_card, text="— dB", font=("Consolas", 22, "bold"), fg=ACCENT_BLUE, bg=CARD_BG)
        self.telemetry_psnr.pack(anchor="w")

        # SSIM Card
        ssim_card = tk.Frame(stats_frame, bg=CARD_BG, bd=1, relief="solid", padx=16, pady=4)
        ssim_card.pack(side="left", padx=8)
        tk.Label(ssim_card, text="STRUCTURAL SSIM", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=CARD_BG).pack(anchor="w")
        self.telemetry_ssim = tk.Label(ssim_card, text="—", font=("Consolas", 22, "bold"), fg=ACCENT_GREEN, bg=CARD_BG)
        self.telemetry_ssim.pack(anchor="w")

        # Latency Card
        time_card = tk.Frame(stats_frame, bg=CARD_BG, bd=1, relief="solid", padx=16, pady=4)
        time_card.pack(side="left", padx=8)
        tk.Label(time_card, text="INFERENCE LATENCY", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=CARD_BG).pack(anchor="w")
        self.telemetry_time = tk.Label(time_card, text="— ms", font=("Consolas", 22, "bold"), fg=TEXT_PRIMARY, bg=CARD_BG)
        self.telemetry_time.pack(anchor="w")

        # Right side: Shortcuts legend & operational mode
        right_box = tk.Frame(bar, bg=HEADER_BG)
        right_box.pack(side="right", padx=10)

        self.telemetry_mode = tk.Label(right_box, text="Mode: Reference-Verified Optical Test", font=("Segoe UI", 9, "bold"), fg=TEXT_SECONDARY, bg=HEADER_BG)
        self.telemetry_mode.pack(anchor="e")

        shortcuts_lbl = tk.Label(
            right_box,
            text="Shortcuts: [Space] Restore  |  [← / →] Cycle Samples  |  [R] Reset",
            font=("Segoe UI", 9, "bold"),
            fg=ACCENT_AMBER,
            bg=HEADER_BG
        )
        shortcuts_lbl.pack(anchor="e", pady=(4, 0))

    # ---------------------------------------------------------
    # Interactive Drawing / Eraser Handlers (Zero-Leak Implementation)
    # ---------------------------------------------------------
    def on_canvas_press(self, event):
        self.last_draw_x = event.x
        self.last_draw_y = event.y
        self.erase_at(event.x, event.y)

    def on_canvas_drag(self, event):
        if self.last_draw_x is not None and self.last_draw_y is not None:
            self.erase_line(self.last_draw_x, self.last_draw_y, event.x, event.y)
        else:
            self.erase_at(event.x, event.y)
        self.last_draw_x = event.x
        self.last_draw_y = event.y

    def on_canvas_release(self, _):
        self.last_draw_x = None
        self.last_draw_y = None

    def erase_at(self, x, y):
        if self.damaged_img is None:
            return
        self.has_manual_edits = True
        r = self.brush_size // 2

        # Draw on PIL image
        draw = ImageDraw.Draw(self.damaged_img)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=self.erase_color_val)

        # Refresh canvas with ZERO memory leak (delete previous canvas item before adding new)
        self.tk_damaged = ImageTk.PhotoImage(self.damaged_img)
        self.canvas_damaged.delete("all")
        self.canvas_damaged.create_image(0, 0, anchor="nw", image=self.tk_damaged)
        self.sub2.configure(text="Manual Defect Added | Press Space to Restore", fg=ACCENT_AMBER)

    def erase_line(self, x0, y0, x1, y1):
        if self.damaged_img is None:
            return
        self.has_manual_edits = True
        r = self.brush_size // 2

        draw = ImageDraw.Draw(self.damaged_img)
        draw.line([x0, y0, x1, y1], fill=self.erase_color_val, width=self.brush_size)
        draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=self.erase_color_val)

        # Refresh canvas with ZERO memory leak (delete previous canvas item before adding new)
        self.tk_damaged = ImageTk.PhotoImage(self.damaged_img)
        self.canvas_damaged.delete("all")
        self.canvas_damaged.create_image(0, 0, anchor="nw", image=self.tk_damaged)
        self.sub2.configure(text="Manual Defect Added | Press Space to Restore", fg=ACCENT_AMBER)

    # ---------------------------------------------------------
    # Image Loading & Degradation
    # ---------------------------------------------------------
    def on_sample_combo(self, _):
        idx = self.sample_combo.current()
        if 0 <= idx < len(self.sample_files):
            self.load_sample_by_path(self.sample_files[idx])

    def on_browse_file(self):
        f = filedialog.askopenfilename(
            title="Select Semiconductor Ground Truth Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.npy"), ("All Files", "*.*")]
        )
        if f:
            self.load_sample_by_path(Path(f))

    def load_sample_by_path(self, path):
        try:
            if path.suffix.lower() == ".npy":
                arr = np.load(path)
                if arr.ndim == 3:
                    arr = arr[0]
                arr = np.clip(arr, 0.0, 1.0)
                img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
            else:
                img = Image.open(path).convert("L")

            # Center crop or resize to 256x256
            w, h = img.size
            if w != 256 or h != 256:
                if w >= 256 and h >= 256:
                    left = (w - 256) // 2
                    top = (h - 256) // 2
                    img = img.crop((left, top, left + 256, top + 256))
                else:
                    img = img.resize((256, 256), Image.BILINEAR)

            self.clean_gt_img = img
            self.clean_gt_arr = np.array(img).astype(np.float32) / 255.0

            # Render Clean GT (clean canvas display list)
            self.tk_gt = ImageTk.PhotoImage(self.clean_gt_img)
            self.canvas_gt.delete("all")
            self.canvas_gt.create_image(0, 0, anchor="nw", image=self.tk_gt)
            self.sub1.configure(text=f"File: {path.name} | 256x256", fg=TEXT_SECONDARY)

            # Apply initial moderate degradation (Default: Blur 1.0, Noise 6%)
            self.apply_slider_degradation()

            # Clear old restoration outputs
            self.canvas_restored.delete("all")
            self.canvas_heatmap.delete("all")
            self.sub3.configure(text="Click Restore or press Space to execute", fg=TEXT_MUTED)
            self.sub4.configure(text="Restoration Error Map: Dark = close to ground truth, Bright = remaining difference", fg=TEXT_MUTED)

            self.telemetry_psnr.configure(text="— dB")
            self.telemetry_ssim.configure(text="—")
            self.telemetry_time.configure(text="— ms")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def apply_slider_degradation(self):
        if self.clean_gt_arr is None:
            return

        sigma = float(self.blur_slider.get())
        noise_level = float(self.noise_slider.get())

        # 1. 2x downsampling (256x256 -> 128x128)
        lr = cv2.resize(self.clean_gt_arr, (128, 128), interpolation=cv2.INTER_AREA)

        # 2. Gaussian blur
        if sigma > 0.01:
            k = int(max(3, int(sigma * 4) | 1))
            blurred = cv2.GaussianBlur(lr, (k, k), sigmaX=sigma, sigmaY=sigma)
        else:
            blurred = lr

        # 3. Fine sensor grain noise (multiplicative speckle + additive sensor noise)
        if noise_level > 0.001:
            shape = 1.0 / max(noise_level ** 2, 1e-4)
            scale = noise_level ** 2
            speckle = np.random.gamma(shape=shape, scale=scale, size=lr.shape)
            sensor = np.random.normal(0, noise_level * 0.1, size=lr.shape)
            deg = np.clip(blurred * speckle + sensor, 0.0, 1.0)
        else:
            deg = blurred

        # Store clean 128x128 degraded array directly
        self.lr_degraded_arr = deg.astype(np.float32)

        # Up-render to 256x256 canvas for crisp interactive drawing
        deg_256 = cv2.resize(deg, (256, 256), interpolation=cv2.INTER_NEAREST)
        self.damaged_img = Image.fromarray((deg_256 * 255.0).astype(np.uint8), mode="L")
        self.has_manual_edits = False

        # Clean canvas display list to eliminate memory accumulation
        self.tk_damaged = ImageTk.PhotoImage(self.damaged_img)
        self.canvas_damaged.delete("all")
        self.canvas_damaged.create_image(0, 0, anchor="nw", image=self.tk_damaged)
        self.sub2.configure(text=f"Degraded: Blur σ={sigma:.1f} | Noise={int(noise_level*100)}%", fg=TEXT_SECONDARY)

    def reset_to_degraded(self):
        self.apply_slider_degradation()

    # ---------------------------------------------------------
    # Restoration Engine Execution
    # ---------------------------------------------------------
    def on_restore_clicked(self):
        if self.is_processing:
            return
        if self.damaged_img is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        if self.model is None:
            messagebox.showwarning("Warning", "Model is still initializing. Please wait a second.")
            return

        self.is_processing = True
        self.btn_restore.configure(text="PROCESSING...", bg="#64748B", state="disabled")
        self.status_pill.configure(text="SwinIR Restoring...", fg=ACCENT_AMBER)

        # Run inference in worker thread to prevent UI freezing
        t = threading.Thread(target=self.run_inference_worker, daemon=True)
        t.start()

    def run_inference_worker(self):
        try:
            start_t = time.perf_counter()

            # Prepare input tensor:
            # If no manual edits, use the pristine 128x128 degraded array directly (no double-downsampling distortion!)
            if not self.has_manual_edits and hasattr(self, "lr_degraded_arr") and self.lr_degraded_arr is not None:
                lr_input = self.lr_degraded_arr
            else:
                # When manual defect drawing occurred, downsample the edited 256x256 canvas
                dam_arr = np.array(self.damaged_img).astype(np.float32) / 255.0
                lr_input = cv2.resize(dam_arr, (128, 128), interpolation=cv2.INTER_AREA)

            inp_t = torch.from_numpy(lr_input).unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    out_t = self.model(inp_t)

            torch.cuda.synchronize() if DEVICE == "cuda" else None
            inference_ms = (time.perf_counter() - start_t) * 1000.0

            res_np = out_t.squeeze().clamp(0.0, 1.0).cpu().numpy()
            restored_pil = Image.fromarray((res_np * 255.0).round().astype(np.uint8), mode="L")

            # Calculate Difference / Error Map
            if self.has_manual_edits:
                # Highlight inpainting delta against input upscaled
                base_ref = cv2.resize(dam_arr, (256, 256), interpolation=cv2.INTER_LINEAR)
                diff = np.abs(res_np - base_ref)
            else:
                # Highlight delta against true clean Ground Truth
                diff = np.abs(res_np - self.clean_gt_arr)

            diff_scaled = np.clip(diff / 0.25, 0.0, 1.0)
            diff_u8 = (diff_scaled * 255.0).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(diff_u8, cv2.COLORMAP_INFERNO)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
            heatmap_pil = Image.fromarray(heatmap_rgb)

            # Compute PSNR/SSIM if Ground Truth is valid and untampered
            psnr_val = None
            ssim_val = None
            if not self.has_manual_edits and self.clean_gt_arr is not None:
                gt_t = torch.from_numpy(self.clean_gt_arr).unsqueeze(0).unsqueeze(0).to(DEVICE)
                psnr_val = calculate_psnr(out_t, gt_t)
                ssim_val = calculate_ssim(out_t, gt_t)

            self.dispatch_to_ui(self.update_restoration_results, restored_pil, heatmap_pil, inference_ms, psnr_val, ssim_val)
        except Exception as e:
            print(f"Inference error: {e}")
            self.dispatch_to_ui(lambda err=str(e): messagebox.showerror("Inference Error", err))
            self.dispatch_to_ui(self.reset_processing_state)

    def update_restoration_results(self, restored_pil, heatmap_pil, latency_ms, psnr, ssim):
        self.restored_img = restored_pil
        self.heatmap_img = heatmap_pil

        # Render Restored (Clean canvas display list)
        self.tk_restored = ImageTk.PhotoImage(restored_pil)
        self.canvas_restored.delete("all")
        self.canvas_restored.create_image(0, 0, anchor="nw", image=self.tk_restored)
        self.sub3.configure(text=f"Restored Output (256x256) | Latency: {latency_ms:.1f}ms", fg=ACCENT_GREEN)

        # Render Heatmap (Clean canvas display list)
        self.tk_heatmap = ImageTk.PhotoImage(heatmap_pil)
        self.canvas_heatmap.delete("all")
        self.canvas_heatmap.create_image(0, 0, anchor="nw", image=self.tk_heatmap)
        self.sub4.configure(text="Restoration Error Map: Dark = close to ground truth, Bright = remaining difference", fg="#F43F5E")

        # Telemetry Bar Updates (20pt+ Prominent Font)
        self.telemetry_time.configure(text=f"{latency_ms:.1f} ms")

        if self.has_manual_edits or psnr is None:
            self.telemetry_psnr.configure(text="N/A", fg=TEXT_MUTED)
            self.telemetry_ssim.configure(text="N/A", fg=TEXT_MUTED)
            self.telemetry_mode.configure(text="Mode: Interactive Hand-Drawn Defect Test (Reference-Free)", fg=ACCENT_AMBER)
        else:
            self.telemetry_psnr.configure(text=f"{psnr:.2f} dB", fg=ACCENT_BLUE)
            self.telemetry_ssim.configure(text=f"{ssim:.4f}", fg=ACCENT_GREEN)
            self.telemetry_mode.configure(text="Mode: Standard Optical Physical Verification", fg=TEXT_SECONDARY)

        self.reset_processing_state()
        self.status_pill.configure(text=f"Ready ({latency_ms:.0f}ms)", fg=ACCENT_GREEN)

    def reset_processing_state(self):
        self.is_processing = False
        self.btn_restore.configure(text="RESTORE IMAGE [SPACE]", bg=ACTIVE_CTA, state="normal")


def main():
    root = tk.Tk()
    app = VeltraxxApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
