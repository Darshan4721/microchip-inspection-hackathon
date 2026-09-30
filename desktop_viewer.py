import sys
import json
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "final_test_50" / "clean"
DEG_DIR = ROOT / "final_test_50" / "degraded"
BEST_DIR = ROOT / "final_test_50" / "restored_best"
V2_DIR = ROOT / "final_test_50" / "restored_finetuned"
JSON_PATH = ROOT / "final_test_50" / "results_phase14.json"

if not JSON_PATH.exists():
    print(f"Error: Results JSON not found at {JSON_PATH}. Run the benchmark script first.")
    sys.exit(1)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    DATA = json.load(f)

class RealisticInspectionViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("KLA Semiconductor Inspection Photo Viewer (Phase 14 Realistic)")
        self.root.geometry("1060x450")
        self.root.minsize(960, 400)
        self.root.configure(bg="#12141a")

        self.idx = 0

        # Top Control Bar
        top_bar = tk.Frame(root, bg="#12141a", pady=10)
        top_bar.pack(fill="x", padx=14)

        title_lbl = tk.Label(top_bar, text="🔬 Semiconductor Inspection Viewer", font=("Segoe UI", 12, "bold"), fg="#58a6ff", bg="#12141a")
        title_lbl.pack(side="left", padx=5)

        btn_prev = tk.Button(top_bar, text="◀ Prev", command=self.prev_img, bg="#21262d", fg="#c9d1d9", font=("Segoe UI", 9, "bold"), padx=12, pady=3, relief="flat", cursor="hand2")
        btn_prev.pack(side="left", padx=12)

        self.combo = ttk.Combobox(
            top_bar,
            values=[f"{d['sample_id']} | v2: {d['v2_p']:.2f} dB ({d['delta_p']:+.2f} dB)" for d in DATA],
            state="readonly",
            width=36,
            font=("Segoe UI", 9)
        )
        self.combo.current(0)
        self.combo.bind("<<ComboboxSelected>>", self.on_select)
        self.combo.pack(side="left", padx=5)

        btn_next = tk.Button(top_bar, text="Next ▶", command=self.next_img, bg="#21262d", fg="#c9d1d9", font=("Segoe UI", 9, "bold"), padx=12, pady=3, relief="flat", cursor="hand2")
        btn_next.pack(side="left", padx=10)

        self.stats_lbl = tk.Label(top_bar, text="", font=("Segoe UI", 10, "bold"), fg="#3fb950", bg="#12141a")
        self.stats_lbl.pack(side="right", padx=10)

        # Image Grid (4 columns)
        grid_frame = tk.Frame(root, bg="#12141a")
        grid_frame.pack(fill="both", expand=True, padx=14, pady=4)

        titles = [
            ("1. Damaged (128x128)", "#8b949e"),
            ("2. best.pth (256x256)", "#8b949e"),
            ("3. finetuned_v2 (256x256) 🌟", "#3fb950"),
            ("4. Ground Truth (256x256)", "#58a6ff")
        ]

        self.img_labels = []
        self.card_stat_labels = []

        for col, (title, color) in enumerate(titles):
            card = tk.Frame(grid_frame, bg="#161b22", bd=1, relief="solid")
            card.grid(row=0, column=col, padx=6, pady=4, sticky="nsew")
            grid_frame.grid_columnconfigure(col, weight=1)

            t_lbl = tk.Label(card, text=title, font=("Segoe UI", 9, "bold"), fg=color, bg="#161b22", pady=4)
            t_lbl.pack()

            img_lbl = tk.Label(card, bg="#000000")
            img_lbl.pack(padx=6, pady=4, expand=True)
            self.img_labels.append(img_lbl)

            s_lbl = tk.Label(card, text="-", font=("Segoe UI", 9), fg="#c9d1d9", bg="#161b22", pady=4)
            s_lbl.pack()
            self.card_stat_labels.append(s_lbl)

        # Bottom help
        bot_bar = tk.Frame(root, bg="#12141a", pady=6)
        bot_bar.pack(fill="x", padx=14)
        tk.Label(bot_bar, text="Controls: Left / Right arrow keys navigate samples. Close window to exit.", font=("Segoe UI", 8), fg="#6e7681", bg="#12141a").pack(side="left")

        root.bind("<Left>", lambda e: self.prev_img())
        root.bind("<Right>", lambda e: self.next_img())

        self.load_sample(0)

    def load_sample(self, idx):
        self.idx = max(0, min(idx, len(DATA) - 1))
        self.combo.current(self.idx)
        d = DATA[self.idx]
        sid = d["sample_id"]

        paths = [
            DEG_DIR / f"{sid}.png",
            BEST_DIR / f"{sid}.png",
            V2_DIR / f"{sid}.png",
            CLEAN_DIR / f"{sid}.png"
        ]

        self.photos = []
        target_size = (210, 210)

        for i, p in enumerate(paths):
            if p.exists():
                im = Image.open(p).convert("L")
                im_res = im.resize(target_size, Image.NEAREST if i == 0 else Image.BILINEAR)
                photo = ImageTk.PhotoImage(im_res)
                self.photos.append(photo)
                self.img_labels[i].configure(image=photo)
            else:
                self.img_labels[i].configure(text="Missing", image="")

        # Update card stats
        self.card_stat_labels[0].configure(text=f"Defocus σ: {d['sigma']:.2f} + Sensor Grain")
        self.card_stat_labels[1].configure(text=f"PSNR: {d['best_p']:.2f} dB | SSIM: {d['best_s']:.4f}")
        self.card_stat_labels[2].configure(text=f"PSNR: {d['v2_p']:.2f} dB | SSIM: {d['v2_s']:.4f}", fg="#3fb950")
        self.card_stat_labels[3].configure(text="Pristine SEM Ground Truth")

        d_str = f"v2 Gain: {d['delta_p']:+.2f} dB PSNR  |  {d['delta_s']:+.4f} SSIM"
        self.stats_lbl.configure(text=d_str)

    def prev_img(self):
        if self.idx > 0:
            self.load_sample(self.idx - 1)

    def next_img(self):
        if self.idx < len(DATA) - 1:
            self.load_sample(self.idx + 1)

    def on_select(self, event):
        self.load_sample(self.combo.current())

if __name__ == "__main__":
    root = tk.Tk()
    app = RealisticInspectionViewer(root)
    root.mainloop()
