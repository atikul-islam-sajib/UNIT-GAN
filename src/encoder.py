import os
import sys
import math
import argparse
import torch
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")

from utils import config
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
                nn.Conv2d(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels * 2,
                    kernel_size=self.kerenl_size // 2,
                    stride=(self.in_channels // self.in_channels) + 1,
                    padding=self.in_channels // self.in_channels,
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
    parser = argparse.ArgumentParser(description="Encoder for UNIT-GAN".title())
    parser.add_argument(
        "--in_channels", type=int, default=config()["dataloader"]["image_channels"], help="Number of input channels"
    )

    args = parser.parse_args()

    in_channels = args.in_channels

    batch_size = config()["dataloader"]["batch_size"]
    image_size = config()["dataloader"]["image_size"]

    shared_E = ResidualBlock(
        in_channels=int(in_channels * (math.pow(2, 8) - 1) / in_channels + 1)
    )

    encoder1 = Encoder(in_channels=in_channels, sharedBlocks=shared_E)
    encoder2 = Encoder(in_channels=in_channels, sharedBlocks=shared_E)

    mu1, z1 = encoder1(torch.randn(batch_size, in_channels, image_size, image_size))
    mu2, z2 = encoder2(torch.randn(batch_size, in_channels, image_size, image_size))

    assert (
        mu1.size() == mu2.size() == z1.size() == z2.size()
    ), "Shape mismatch(mu1, mu2) and (z1, z2)".capitalize()

    for filename in ["encoder1", "encoder2"]:
        draw_graph(
            model=encoder1,
            input_data=torch.randn(batch_size, in_channels, image_size, image_size),
        ).visual_graph.render(filename=f"./artifacts/files/{filename}", format="pdf")
