import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from residualBlock import ResidualBlock


class Generator(nn.Module):
    def __init__(self, in_channels: int = 256, sharedBlock: ResidualBlock = None):
        super(Generator, self).__init__()

        self.in_channels = in_channels

        if isinstance(sharedBlock, ResidualBlock):
            self.sharedBlock = sharedBlock
        else:
            raise ValueError(
                "shared_block must be an instance of ResidualBlock".capitalize()
            )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = self.sharedBlock(x)
            return x
        else:
            raise ValueError("Input should be the tensor type".capitalize())


if __name__ == "__main__":
    netG = Generator(in_channels=256)
