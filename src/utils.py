import numpy as np
import torch


def calculate_psnr(pred, target):

    pred = torch.clamp(
        pred,
        0.0,
        1.0
    )

    target = torch.clamp(
        target,
        0.0,
        1.0
    )

    mse = torch.mean(
        (pred - target) ** 2
    )

    if mse == 0:
        return float("inf")

    psnr = 10 * torch.log10(
        1.0 / mse
    )

    return psnr.item()


def save_npy(tensor, path):

    tensor = tensor.detach().cpu()

    tensor = tensor.squeeze().numpy()

    tensor = np.clip(
        tensor,
        0.0,
        1.0
    )

    np.save(
        path,
        tensor.astype(np.float32)
    )


def load_checkpoint(
    model,
    optimizer,
    path,
    device
):

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    epoch = checkpoint.get(
        "epoch",
        0
    )

    return epoch