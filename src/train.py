from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import TrainDataset
from model import create_model
from utils import calculate_psnr


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "Train" / "train"

DEGRADED_PATH = TRAIN_PATH / "NoisyLR"
GT_PATH = TRAIN_PATH / "GT"

CHECKPOINT_DIR = ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(
    exist_ok=True
)


# =========================================================
# DEVICE
# =========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE}")


# =========================================================
# CONFIGURATION
# =========================================================

BATCH_SIZE = 4

EPOCHS = 20

LEARNING_RATE = 1e-5

NUM_WORKERS = 0

VALIDATION_SPLIT = 0.1

SEED = 42


# =========================================================
# RANDOM SEED
# =========================================================

random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# DATASET
# =========================================================

print("\nLoading dataset...")

dataset = TrainDataset(
    degraded_dir=DEGRADED_PATH,
    gt_dir=GT_PATH
)


# =========================================================
# CHECK ONE SAMPLE
# =========================================================

sample_degraded, sample_gt = dataset[0]

print(
    f"Sample degraded shape: "
    f"{sample_degraded.shape}"
)

print(
    f"Sample GT shape: "
    f"{sample_gt.shape}"
)

print(
    f"Sample degraded dtype: "
    f"{sample_degraded.dtype}"
)

print(
    f"Sample GT dtype: "
    f"{sample_gt.dtype}"
)


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

validation_size = max(
    1,
    int(len(dataset) * VALIDATION_SPLIT)
)

training_size = (
    len(dataset) - validation_size
)

train_dataset, val_dataset = random_split(
    dataset,
    [
        training_size,
        validation_size
    ],
    generator=torch.Generator().manual_seed(SEED)
)


print(
    f"\nTraining samples: "
    f"{len(train_dataset)}"
)

print(
    f"Validation samples: "
    f"{len(val_dataset)}"
)


# =========================================================
# DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda")
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda")
)


# =========================================================
# MODEL
# =========================================================

print("\nCreating SWISR model...")

model = create_model()

model = model.to(DEVICE)


# =========================================================
# LOSS
# =========================================================

criterion = nn.L1Loss()


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# =========================================================
# SCHEDULER
# =========================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# =========================================================
# MIXED PRECISION
# =========================================================

use_amp = DEVICE == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# =========================================================
# TRAINING
# =========================================================

best_psnr = -float("inf")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0


    # =====================================================
    # TRAIN
    # =====================================================

    for batch_idx, (
        degraded,
        clean
    ) in enumerate(train_loader):

        degraded = degraded.to(
            DEVICE,
            non_blocking=True
        )

        clean = clean.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        # -------------------------------------------------
        # Forward
        # -------------------------------------------------

        with torch.amp.autocast(
            "cuda",
            enabled=use_amp
        ):

            restored = model(
                degraded
            )


            # -------------------------------------------------
            # IMPORTANT SHAPE CHECK
            # -------------------------------------------------

            if batch_idx == 0:

                print(
                    f"\nDegraded shape: "
                    f"{degraded.shape}"
                )

                print(
                    f"Restored shape: "
                    f"{restored.shape}"
                )

                print(
                    f"GT shape: "
                    f"{clean.shape}"
                )


            # -------------------------------------------------
            # Loss
            # -------------------------------------------------

            if restored.shape != clean.shape:

                raise RuntimeError(
                    "\nShape mismatch!\n"
                    f"Restored: {restored.shape}\n"
                    f"Ground truth: {clean.shape}\n"
                    "The model output and GT must have "
                    "the same shape."
                )


            loss = criterion(
                restored,
                clean
            )


        # -------------------------------------------------
        # Backpropagation
        # -------------------------------------------------

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()


        total_loss += loss.item()


        # -------------------------------------------------
        # Print progress
        # -------------------------------------------------

        if batch_idx % 10 == 0:

            print(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS} | "
                f"Batch "
                f"{batch_idx}/{len(train_loader)} | "
                f"Loss "
                f"{loss.item():.6f}"
            )


    # =====================================================
    # AVERAGE LOSS
    # =====================================================

    average_loss = (
        total_loss /
        max(len(train_loader), 1)
    )


    # =====================================================
    # LEARNING RATE
    # =====================================================

    scheduler.step()


    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    total_psnr = 0.0


    with torch.no_grad():

        for degraded, clean in val_loader:

            degraded = degraded.to(
                DEVICE,
                non_blocking=True
            )

            clean = clean.to(
                DEVICE,
                non_blocking=True
            )


            restored = model(
                degraded
            )


            if restored.shape != clean.shape:

                raise RuntimeError(
                    "\nValidation shape mismatch!\n"
                    f"Restored: {restored.shape}\n"
                    f"GT: {clean.shape}"
                )


            psnr = calculate_psnr(
                restored,
                clean
            )

            total_psnr += psnr


    average_psnr = (
        total_psnr /
        max(len(val_loader), 1)
    )


    # =====================================================
    # PRINT EPOCH RESULTS
    # =====================================================

    print("\n" + "=" * 60)

    print(
        f"Epoch: "
        f"{epoch + 1}/{EPOCHS}"
    )

    print(
        f"Training Loss: "
        f"{average_loss:.6f}"
    )

    print(
        f"Validation PSNR: "
        f"{average_psnr:.4f} dB"
    )

    print(
        f"Learning Rate: "
        f"{optimizer.param_groups[0]['lr']:.8f}"
    )

    print("=" * 60)


    # =====================================================
    # SAVE LATEST
    # =====================================================

    latest_path = (
        CHECKPOINT_DIR /
        "latest.pth"
    )


    torch.save(
        {
            "epoch": epoch + 1,

            "model": model.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "scheduler":
                scheduler.state_dict(),

            "best_psnr":
                best_psnr
        },
        latest_path
    )


    # =====================================================
    # SAVE BEST
    # =====================================================

    if average_psnr > best_psnr:

        best_psnr = average_psnr


        best_path = (
            CHECKPOINT_DIR /
            "best.pth"
        )


        torch.save(
            {
                "epoch": epoch + 1,

                "model":
                    model.state_dict(),

                "optimizer":
                    optimizer.state_dict(),

                "scheduler":
                    scheduler.state_dict(),

                "best_psnr":
                    best_psnr
            },
            best_path
        )


        print(
            f"New best model saved: "
            f"{best_psnr:.4f} dB"
        )


# =========================================================
# FINISHED
# =========================================================

print(
    "\nTraining completed."
)

print(
    f"Best PSNR: "
    f"{best_psnr:.4f} dB"
)