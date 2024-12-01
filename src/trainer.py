import os
import sys
import torch
import argparse
import torch.optim as optim
import torch.nn as nn

sys.path.append("./src/")

from helper import helper
from encoder import Encoder
from gan_loss import GANLoss
from generator import Generator
from pixel_loss import PixelLoss
from kl_divergence import KLDivergence
from discriminator import Discriminator
from utils import config, dump, load, device_init, weight_init


class Trainer:
    def __init__(
        self,
        epochs: int = 500,
        lr: float = 2e-5,
        beta1: float = 0.5,
        beta2: float = 0.999,
        momentum: float = 0.95,
        adam: bool = True,
        SGD: bool = False,
        device: str = "cuda",
        l1_regularization: bool = False,
        l2_regularization: float = False,
        elasticNet_regularization: bool = False,
        verbose: bool = True,
        mlFlow: bool = True,
    ):
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.adam = adam
        self.SGD = SGD
        self.device = device
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.elasticNet_regularization = elasticNet_regularization
        self.verbose = verbose
        self.mlFlow = mlFlow

        self.device = device_init(device=self.device)

        self.init = helper(
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            momentum=self.momentum,
            adam=self.adam,
            SGD=self.SGD,
        )

        self.train_dataloader = self.init["train_dataloader"]
        self.valid_dataloader = self.init["valid_dataloader"]

        self.encoder1 = self.init["E1"].to(self.device)
        self.encoder2 = self.init["E2"].to(self.device)

        self.netG1 = self.init["netG1"].to(self.device)
        self.netG2 = self.init["netG2"].to(self.device)

        self.netD1 = self.init["netD1"].to(self.device)
        self.netD2 = self.init["netD2"].to(self.device)

        self.optimizerG = self.init["optimizerG"]
        self.optimizerD1 = self.init["optimizerD1"]
        self.optimizerD2 = self.init["optimizerD2"]

        self.criterion = self.init["criterion"]
        self.pixelLoss = self.init["pixelLoss"]
        self.kl_loss = self.init["kl_loss"]

        assert (
            self.train_dataloader.__class__ == torch.utils.data.DataLoader
        ), "Train dataloader shoould be torch.utils.data.DataLoader".capitalize()
        assert (
            self.valid_dataloader.__class__ == torch.utils.data.DataLoader
        ), "Valid dataloader shoould be torch.utils.data.DataLoader".capitalize()

        assert (
            self.encoder1.__class__ == Encoder
        ), "Encoder object should be Encoder class".capitalize()
        assert (
            self.encoder2.__class__ == Encoder
        ), "Encoder object should be Encoder class".capitalize()

        assert (
            self.netG1.__class__ == Generator
        ), "Generator object should be Generator class".capitalize()
        assert (
            self.netG2.__class__ == Generator
        ), "Generator object should be Generator class".capitalize()

        assert (
            self.netD1.__class__ == Discriminator
        ), "netD1 object should be Discriminator class".capitalize()
        assert (
            self.netD2.__class__ == Discriminator
        ), "netD2 object should be Discriminator class".capitalize()

        assert (
            self.optimizerG.__class__ == optim.Adam
        ), "optimizerG object should be Adam class".capitalize()
        assert (
            self.optimizerD1.__class__ == optim.Adam
        ), "optimizerD1 object should be Adam class".capitalize()
        assert (
            self.optimizerD2.__class__ == optim.Adam
        ), "optimizerD2 object should be Adam class".capitalize()

        assert (
            self.criterion.__class__ == GANLoss
        ), "Criterion object should be GANLoss class".capitalize()
        assert (
            self.pixelLoss.__class__ == PixelLoss
        ), "pixelLoss object should be PixelLoss class".capitalize()
        assert (
            self.kl_loss.__class__ == KLDivergence
        ), "KL Divergence object should be PixelLoss class".capitalize()

    def l1_regularizer(model):
        if model is not None:
            return sum(torch.norm(params, 1) for params in model.parameters())
        else:
            raise TypeError(
                "Model should be passed in the l1 regularizer".capitalize()()
            )


if __name__ == "__main__":
    trainer = Trainer()
