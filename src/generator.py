import os
import sys
import math
import torch
import argparse
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")

from utils import config
from residualBlock import ResidualBlock


class Generator(nn.Module):
    def __init__(self, in_channels: int = 256, sharedBlocks: ResidualBlock = None):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = self.in_channels

        self.kernel_size = int(math.sqrt(math.sqrt(self.in_channels)))
        self.stride_size = int(math.sqrt(self.kernel_size))
        self.padding_size = self.stride_size // self.stride_size

        if isinstance(sharedBlocks, ResidualBlock):
            self.sharedBlock = sharedBlocks
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
    parser = argparse.ArgumentParser(description="Generator for the UNIT-GAN".title())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=256,
        help="Number of input channels".capitalize(),
    )

    args = parser.parse_args()

    image_channels = args.in_channels

    batch_size = config()["dataloader"]["batch_size"]
    image_size = config()["dataloader"]["image_size"]

    shared_G = ResidualBlock(in_channels=image_channels)

    netG1 = Generator(in_channels=image_channels, sharedBlocks=shared_G)
    netG2 = Generator(in_channels=image_channels, sharedBlocks=shared_G)

    generatedImage1 = netG1(
        torch.randn(batch_size, image_channels, image_size, image_size)
    )
    generatedImage2 = netG2(
        torch.randn(batch_size, image_channels, image_size, image_size)
    )

    assert (
        generatedImage1.size() == generatedImage2.size()
    ), "Shape mismatch(generatedImage1, generatedImage2)".capitalize()

    for filename in ["netG1", "netG2"]:
        draw_graph(
            model=netG1,
            input_data=torch.randn(batch_size, image_channels, image_size, image_size),
        ).visual_graph.render(filename=f"./artifacts/files/{filename}", format="pdf")
