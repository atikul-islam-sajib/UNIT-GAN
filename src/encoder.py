import os
import sys
import argparse
import torch
import torch.nn as nn


sys.path.append("./src/")

from residualBlock import ResidualBlock


class Encoder(nn.Module):

    def __init__(self, in_channels: int = 3, sharedBlocks=None):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = (self.in_channels * 21) + 1
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
    encoder1 = Encoder(in_channels=3, sharedBlocks=ResidualBlock(in_channels=256))
    print(encoder1)
    mu, z = encoder1(torch.randn(1, 3, 128, 128))

    print("mu:", mu.size())
    print("z:", z.size())
