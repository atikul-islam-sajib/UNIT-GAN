import os
import sys
import math
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = int(math.pow(2, self.in_channels + self.in_channels))

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return 1
        else:
            raise ValueError("Input should be the tensor type".capitalize())


if __name__ == "__main__":
    netD = Discriminator(in_channels=3)
