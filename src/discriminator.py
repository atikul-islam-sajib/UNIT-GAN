import os
import sys
import math
import torch
import argparse
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")

from utils import config

class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = int(math.pow(2, self.in_channels + self.in_channels))

        self.kernel_size = self.in_channels + 1
        self.stride_size = self.kernel_size // 2
        self.padding_size = self.stride_size // 2

        self.layers = list()

        for index in range(4):
            self.layers.append(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride_size,
                    padding=self.padding_size,
                )
            )

            if index != 0:
                self.layers.append(nn.InstanceNorm2d(num_features=self.out_channels))

            self.layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

            self.in_channels = self.out_channels
            self.out_channels = self.out_channels * 2

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels // self.in_channels,
                    kernel_size=self.kernel_size - 1,
                    stride=self.stride_size - 1,
                    padding=self.padding_size,
                )
            )
        )

        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.model(x)
        else:
            raise ValueError("Input should be the tensor type".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discriminator for the UNIT-GAN".title()
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Define the number of channels".capitalize(),
    )
    args = parser.parse_args()

    image_channels = args.in_channels

    batch_size = config()["dataloader"]["batch_size"]
    image_size = config()["dataloader"]["image_size"]

    netD = Discriminator(in_channels=image_channels)

    assert netD(
        torch.randn(batch_size, image_channels, image_size, image_size)
    ).size() == torch.Size(
        [
            batch_size,
            image_channels // image_channels,
            image_size // 16,
            image_size // 16,
        ]
    )

    draw_graph(
        model=netD,
        input_data=torch.randn(batch_size, image_channels, image_size, image_size),
    ).visual_graph.render(filename="./artifacts/files/netD", format="pdf")
