from pathlib import Path
import random
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import TrainDataset
from model import create_model
from utils import calculate_psnr

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "Train" / "train" / "train"
DEGRADED_PATH = TRAIN_PATH / "NoisyLR"
GT_PATH = TRAIN_PATH / "GT"

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

BATCH_SIZE = 1
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
VALIDATION_SPLIT = 0.1
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

print("\nLoading dataset...")

dataset = TrainDataset(
    degraded_dir=DEGRADED_PATH,
    gt_dir=GT_PATH
)

sample_degraded, sample_gt = dataset[0]

print("Sample degraded shape:", sample_degraded.shape)
print("Sample GT shape:", sample_gt.shape)

validation_size = max(1, int(len(dataset) * VALIDATION_SPLIT))
training_size = len(dataset) - validation_size

train_dataset, val_dataset = random_split(
    dataset,
    [training_size, validation_size],
    generator=torch.Generator().manual_seed(SEED)
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda")
)

print("\nCreating lightweight SWISR model...")

model = create_model()
model = model.to(DEVICE)

criterion = nn.L1Loss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

use_amp = DEVICE == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)

best_psnr = -float("inf")
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    for batch_idx, (degraded, clean) in enumerate(train_loader):

        degraded = degraded.to(DEVICE, non_blocking=True)
        clean = clean.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda",
            enabled=use_amp
        ):

            restored = model(degraded)

            if batch_idx == 0:
                print("\nInput :", degraded.shape)
                print("Output:", restored.shape)
                print("GT    :", clean.shape)

            if restored.shape != clean.shape:
                raise RuntimeError(
                    f"Shape mismatch: restored={restored.shape}, gt={clean.shape}"
                )

            loss = criterion(restored, clean)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item()

        if batch_idx % 20 == 0:

            if DEVICE == "cuda":
                mem = torch.cuda.memory_allocated() / 1024**3
                print(
                    f"Epoch {epoch+1}/{EPOCHS} "
                    f"Batch {batch_idx}/{len(train_loader)} "
                    f"Loss {loss.item():.6f} "
                    f"GPU {mem:.2f} GB"
                )

            else:
                print(
                    f"Epoch {epoch+1}/{EPOCHS} "
                    f"Batch {batch_idx}/{len(train_loader)} "
                    f"Loss {loss.item():.6f}"
                )

    average_loss = total_loss / max(len(train_loader), 1)

    scheduler.step()

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------
    model.eval()

    total_psnr = 0.0

    with torch.no_grad():

        for degraded, clean in val_loader:

            degraded = degraded.to(DEVICE, non_blocking=True)
            clean = clean.to(DEVICE, non_blocking=True)

            restored = model(degraded)

            psnr = calculate_psnr(restored, clean)

            total_psnr += psnr

    average_psnr = total_psnr / max(len(val_loader), 1)

    print("\n" + "=" * 60)

    print(f"Epoch {epoch+1}/{EPOCHS}")

    print(f"Training Loss : {average_loss:.6f}")

    print(f"Validation PSNR: {average_psnr:.4f} dB")

    print(f"Learning Rate : {optimizer.param_groups[0]['lr']:.8f}")

    print("=" * 60)

    latest_path = CHECKPOINT_DIR / "latest.pth"

    torch.save(
        {
            "epoch": epoch + 1,

            "model": model.state_dict(),

            "optimizer": optimizer.state_dict(),

            "scheduler": scheduler.state_dict(),

            "best_psnr": best_psnr,
        },
        latest_path,
    )

    if average_psnr > best_psnr:

        best_psnr = average_psnr

        best_path = CHECKPOINT_DIR / "best.pth"

        torch.save(
            {
                "epoch": epoch + 1,

                "model": model.state_dict(),

                "optimizer": optimizer.state_dict(),

                "scheduler": scheduler.state_dict(),

                "best_psnr": best_psnr,
            },
            best_path,
        )

        print(f"New best model saved: {best_psnr:.4f} dB")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print("\nTraining completed.")

print(f"Best PSNR: {best_psnr:.4f} dB")