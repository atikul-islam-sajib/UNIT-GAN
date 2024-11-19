import os
import sys
import torch
import argparse
import torch.nn as nn
import torch.optim as optim

sys.path.append("./src/")

from residualBlock import ResidualBlock
from encoder import Encoder
from generator import Generator
from discriminator import Discriminator
from gan_loss import GANLoss
from pixel_loss import PixelLoss


def helper(**kwargs):
    lr = kwargs["lr"]
    beta1 = kwargs["beta1"]
    beta2 = kwargs["beta2"]
    momentum = kwargs["momentum"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]

    shared_E = ResidualBlock(in_channels=256)
    shared_G = ResidualBlock(in_channels=256)

    E1 = Encoder(in_channels=3, sharedBlocks=shared_E)
    E2 = Encoder(in_channels=3, sharedBlocks=shared_E)

    G1 = Generator(in_channels=256, sharedBlocks=shared_G)
    G2 = Generator(in_channels=256, sharedBlocks=shared_G)

    D1 = Discriminator(in_channels=3)
    D2 = Discriminator(in_channels=3)

    if adam:
        optimizerG = optim.Adam(
            params=list(E1.parameters())
            + list(E2.parameters())
            + list(G1.parameters())
            + list(G2.parameters()),
            lr=lr,
            betas=(beta1, beta2),
        )
        optimizerD1 = optim.Adam(params=D1.parameters(), lr=lr, betas=(beta1, beta2))
        optimizerD2 = optim.Adam(params=D2.parameters(), lr=lr, betas=(beta1, beta2))

    elif SGD:
        optimizerG = optim.SGD(
            params=list(E1.parameters())
            + list(E2.parameters())
            + list(G1.parameters())
            + list(G2.parameters()),
            lr=lr,
            momentum=momentum,
        )
        optimizerD1 = optim.SGD(params=D1.parameters(), lr=lr, momentum=momentum)
        optimizerD2 = optim.SGD(params=D2.parameters(), lr=lr, momentum=momentum)

    criterion = GANLoss(reduction="mean")
    pixelLoss = PixelLoss(reduction="mean")

    return {
        "E1": E1,
        "E2": E2,
        "netG1": G1,
        "netG2": G2,
        "optimizerG": optimizerG,
        "optimizerD1": optimizerD1,
        "optimizerD2": optimizerD2,
        "criterion": criterion,
        "pixelLoss": pixelLoss,
    }
