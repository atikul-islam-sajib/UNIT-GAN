import os
import sys
import math
import argparse
import torch
import torch.nn as nn


sys.path.append("./src/")

from residualBlock import ResidualBlock


class Encoder(nn.Module):

    def __init__(self, in_channels: int = 3, sharedBlocks=None):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = int(math.pow(2, self.in_channels + self.in_channels))
        self.kerenl_size = (self.in_channels * 2) + 1

        if not isinstance(sharedBlocks, ResidualBlock):
            raise ValueError(
                "shared_block must be an instance of ResidualBlock".capitalize()
            )

        self.sharedBlocks = sharedBlocks

        self.modelBlocks = list()
        self.downLayers = list()

        self.modelBlocks.append(
            nn.Sequential(
                nn.ReflectionPad2d(padding=self.in_channels),
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kerenl_size,
                ),
                nn.InstanceNorm2d(num_features=self.out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        )

        for _ in range(2):
            self.downLayers.append(
                nn.ReflectionPad2d(padding=self.in_channels // self.in_channels)
            )
            self.downLayers.append(
                nn.Conv2d(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels * 2,
                    kernel_size=self.kerenl_size // 2,
                    stride=(self.in_channels // self.in_channels) + 1,
                )
            )
            self.downLayers.append(
                nn.InstanceNorm2d(num_features=self.out_channels * 2)
            )
            self.downLayers.append(nn.ReLU())

            self.out_channels *= 2

        self.modelBlocks.append(nn.Sequential(*self.downLayers))

        self.modelBlocks.append(
            nn.Sequential(
                *[ResidualBlock(in_channels=self.out_channels) for _ in range(3)]
            )
        )

        self.modelBlocks = nn.Sequential(*self.modelBlocks)

    def reparameterization(self, mu: torch.Tensor):
        if isinstance(mu, torch.Tensor):
            z = torch.randn_like(mu)
            return mu + z
        else:
            raise ValueError("Input should be the tensor type".capitalize())

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = self.modelBlocks(x)
            mu = self.sharedBlocks(x)
            z = self.reparameterization(mu)

            return mu, z
        else:
            raise ValueError("Input should be the tensor type".capitalize())


if __name__ == "__main__":
    in_channels = 3
    shared_E = ResidualBlock(
        in_channels=int(in_channels * (math.pow(2, 8) - 1) / in_channels + 1)
    )

    encoder1 = Encoder(in_channels=3, sharedBlocks=shared_E)
    mu1, z1 = encoder1(torch.randn(1, 3, 128, 128))

    encoder2 = Encoder(in_channels=3, sharedBlocks=shared_E)
    mu2, z2 = encoder2(torch.randn(1, 3, 128, 128))

    print(f"mu1 shape: {mu1.shape}")
    print(f"z1 shape: {z1.shape}")
    print(f"mu2 shape: {mu2.shape}")
    print(f"z2 shape: {z2.shape}")
