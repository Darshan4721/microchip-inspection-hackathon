from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import TestDataset
from model import create_model
from utils import save_npy

ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = (
    ROOT /
    "Test" /
    "Test_NoisyLR" /
    "NoisyLR"
)

CHECKPOINT_PATH = (
    ROOT /
    "checkpoints" /
    "best.pth"
)

RESULT_PATH = (
    ROOT /
    "results"
)

RESULT_PATH.mkdir(
    exist_ok=True
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Device: {DEVICE}"
)

dataset = TestDataset(
    root=TEST_PATH
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)
model = create_model()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model"]
)

model = model.to(
    DEVICE
)

model.eval()


with torch.no_grad():

    for index, (
        noisy,
        filename
    ) in enumerate(loader):

        noisy = noisy.to(
            DEVICE
        )

        restored = model(
            noisy
        )

        output_name = (
            Path(filename[0]).stem
            + "_restored.npy"
        )

        output_path = (
            RESULT_PATH /
            output_name
        )

        save_npy(
            restored,
            output_path
        )

        print(
            f"[{index + 1}/{len(loader)}] "
            f"{filename[0]} "
            f"-> "
            f"{output_name}"
        )


print(
    "\nRestoration completed."
)

print(
    f"Results saved to: "
    f"{RESULT_PATH}"
)