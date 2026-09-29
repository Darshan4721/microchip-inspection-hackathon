import numpy as np
import torch
import torch.nn.functional as F


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


def calculate_ssim(pred, target, window_size=11, sigma=1.5):
    """
    Computes Structural Similarity Index (SSIM) between two images.
    Supports torch.Tensor or numpy.ndarray in [0, 1].
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()

    while pred.ndim < 4:
        pred = pred.unsqueeze(0)
    while target.ndim < 4:
        target = target.unsqueeze(0)

    device = pred.device
    target = target.to(device)

    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size // 2)
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = (gauss / gauss.sum()).unsqueeze(1)
    kernel2d = (gauss @ gauss.T).unsqueeze(0).unsqueeze(0)

    mu1 = F.conv2d(pred, kernel2d, padding=window_size // 2)
    mu2 = F.conv2d(target, kernel2d, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel2d, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel2d, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel2d, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


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