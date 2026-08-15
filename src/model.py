import torch.nn as nn

from SwinIR.models.network_swinir import SwinIR


def create_model():

    model = SwinIR(
        upscale=2,

        in_chans=1,

        img_size=128,

        window_size=8,

        img_range=1.0,

        depths=[
            6,
            6,
            6,
            6,
            6,
            6
        ],

        embed_dim=180,

        num_heads=[
            6,
            6,
            6,
            6,
            6,
            6
        ],

        mlp_ratio=2,

        upsampler="pixelshuffle",

        resi_connection="1conv"
    )

    return model