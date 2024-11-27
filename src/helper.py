import os
import sys
import torch
import argparse
import traceback
import torch.nn as nn
import torch.optim as optim

sys.path.append("./src/")

from utils import config, load
from encoder import Encoder
from gan_loss import GANLoss
from generator import Generator
from pixel_loss import PixelLoss
from discriminator import Discriminator
from residualBlock import ResidualBlock

import warnings

warnings.filterwarnings("ignore")


def load_dataloader():
    processed_path = config()["path"]["processed_path"]

    if os.path.exists(processed_path):
        train_dataloader = os.path.join(processed_path, "train_dataloader.pkl")
        valid_dataloader = os.path.join(processed_path, "valid_dataloader.pkl")

        train_dataloader = load(filename=train_dataloader)
        valid_dataloader = load(filename=valid_dataloader)

        return {
            "train_dataloader": train_dataloader,
            "valid_dataloader": valid_dataloader,
        }


def helper(**kwargs):
    lr = kwargs["lr"]
    beta1 = kwargs["beta1"]
    beta2 = kwargs["beta2"]
    momentum = kwargs["momentum"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]

    shared_E = ResidualBlock(in_channels=256)
    shared_G = ResidualBlock(in_channels=256)

    E1 = Encoder(
        in_channels=config()["dataloader"]["image_channels"], sharedBlocks=shared_E
    )
    E2 = Encoder(
        in_channels=config()["dataloader"]["image_channels"], sharedBlocks=shared_E
    )

    G1 = Generator(in_channels=256, sharedBlocks=shared_G)
    G2 = Generator(in_channels=256, sharedBlocks=shared_G)

    D1 = Discriminator(in_channels=config()["dataloader"]["image_channels"])
    D2 = Discriminator(in_channels=config()["dataloader"]["image_channels"])

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

    try:
        dataset = load_dataloader()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

    return {
        "train_dataloader": dataset["train_dataloader"],
        "valid_dataloader": dataset["valid_dataloader"],
        "E1": E1,
        "E2": E2,
        "netG1": G1,
        "netG2": G2,
        "netD1": D1,
        "netD2": D2,
        "optimizerG": optimizerG,
        "optimizerD1": optimizerD1,
        "optimizerD2": optimizerD2,
        "criterion": criterion,
        "pixelLoss": pixelLoss,
    }


if __name__ == "__main__":
    init = helper(
        lr=2e-4,
        beta1=0.5,
        beta2=0.999,
        momentum=0.95,
        adam=True,
        SGD=False,
    )

    train_dataloader = init["train_dataloader"]
    valid_dataloader = init["valid_dataloader"]

    encoder1 = init["E1"]
    encoder2 = init["E2"]

    netG1 = init["netG1"]
    netG2 = init["netG2"]

    netD1 = init["netD1"]
    netD2 = init["netD2"]

    optimizerG = init["optimizerG"]
    optimizerD1 = init["optimizerD1"]
    optimizerD2 = init["optimizerD2"]

    criterion = init["criterion"]
    pixelLoss = init["pixelLoss"]

    assert (
        train_dataloader.__class__ == torch.utils.data.DataLoader
    ), "Train dataloader shoould be torch.utils.data.DataLoader".capitalize()
    assert (
        valid_dataloader.__class__ == torch.utils.data.DataLoader
    ), "Valid dataloader shoould be torch.utils.data.DataLoader".capitalize()

    assert (
        encoder1.__class__ == Encoder
    ), "Encoder object should be Encoder class".capitalize()
    assert (
        encoder2.__class__ == Encoder
    ), "Encoder object should be Encoder class".capitalize()

    assert (
        netG1.__class__ == Generator
    ), "Generator object should be Generator class".capitalize()
    assert (
        netG2.__class__ == Generator
    ), "Generator object should be Generator class".capitalize()

    assert (
        netD1.__class__ == Discriminator
    ), "netD1 object should be Discriminator class".capitalize()
    assert (
        netD2.__class__ == Discriminator
    ), "netD2 object should be Discriminator class".capitalize()

    assert (
        optimizerG.__class__ == optim.Adam
    ), "optimizerG object should be Adam class".capitalize()
    assert (
        optimizerD1.__class__ == optim.Adam
    ), "optimizerD1 object should be Adam class".capitalize()
    assert (
        optimizerD2.__class__ == optim.Adam
    ), "optimizerD2 object should be Adam class".capitalize()

    assert (
        criterion.__class__ == GANLoss
    ), "Criterion object should be GANLoss class".capitalize()
    assert (
        pixelLoss.__class__ == PixelLoss
    ), "pixelLoss object should be PixelLoss class".capitalize()
