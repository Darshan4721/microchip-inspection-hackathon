from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):

    def __init__(self, degraded_dir, gt_dir):

        self.degraded_dir = Path(degraded_dir)
        self.gt_dir = Path(gt_dir)

        self.files = sorted(
            self.degraded_dir.glob("*.npy")
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .npy files found in {self.degraded_dir}"
            )

        for file in self.files:

            gt_file = self.gt_dir / file.name

            if not gt_file.exists():
                raise RuntimeError(
                    f"Missing GT file: {gt_file}"
                )

        print(f"Found {len(self.files)} training pairs.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        degraded_path = self.files[index]
        gt_path = self.gt_dir / degraded_path.name

        degraded = np.load(degraded_path)
        ground_truth = np.load(gt_path)

        # NumPy -> Tensor
        degraded = torch.from_numpy(degraded).float()
        ground_truth = torch.from_numpy(ground_truth).float()

        # H,W -> C,H,W
        if degraded.ndim == 2:
            degraded = degraded.unsqueeze(0)

        if ground_truth.ndim == 2:
            ground_truth = ground_truth.unsqueeze(0)

        return degraded, ground_truth