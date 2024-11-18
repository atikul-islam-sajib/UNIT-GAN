import os
import sys
import math
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from residualBlock import ResidualBlock


class Generator(nn.Module):
    def __init__(self, in_channels: int = 256, sharedBlock: ResidualBlock = None):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = self.in_channels

        self.kernel_size = int(math.sqrt(math.sqrt(self.in_channels)))
        self.stride_size = int(math.sqrt(self.kernel_size))
        self.padding_size = self.stride_size // self.stride_size

        if isinstance(sharedBlock, ResidualBlock):
            self.sharedBlock = sharedBlock
        else:
            raise ValueError(
                "shared_block must be an instance of ResidualBlock".capitalize()
            )

        self.modelBlocks = []
        self.upsampleBlocks = []

        self.modelBlocks.append(
            nn.Sequential(
                *[ResidualBlock(in_channels=self.in_channels) for _ in range(3)]
            )
        )

        for _ in range(2):
            self.upsampleBlocks.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels // 2,
                    kernel_size=self.kernel_size,
                    stride=self.stride_size,
                    padding=self.padding_size,
                )
            )

            self.upsampleBlocks.append(
                nn.InstanceNorm2d(num_features=self.in_channels // 2)
            )
            self.upsampleBlocks.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

            self.in_channels //= 2

        self.modelBlocks.append(nn.Sequential(*self.upsampleBlocks))

        self.modelBlocks.append(
            nn.Sequential(
                nn.ReflectionPad2d(padding=self.kernel_size - 1),
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.kernel_size - 1,
                    kernel_size=self.kernel_size + 3,
                ),
                nn.Tanh(),
            )
        )

        self.modelBlocks = nn.Sequential(*self.modelBlocks)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = self.sharedBlock(x)
            return self.modelBlocks(x)
        else:
            raise ValueError("Input should be the tensor type".capitalize())


if __name__ == "__main__":
    netG = Generator(in_channels=256, sharedBlock=ResidualBlock(in_channels=256))
    print(netG(torch.randn(1, 256, 32, 32)).size())
