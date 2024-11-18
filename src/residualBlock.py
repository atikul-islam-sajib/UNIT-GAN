import os
import sys
import torch
import argparse
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int = 256):
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels

        self.reflectionpad2d = 1
        self.kernel_size = 3
        self.stride_size = 1

        self.layers = []

        for index in range(2):
            self.layers.append(nn.ReflectionPad2d(self.reflectionpad2d))

            self.layers.append(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride_size,
                )
            )

            self.layers.append(nn.InstanceNorm2d(num_features=self.out_channels))

            if index != 1:
                self.layers.append(nn.ReLU())

        self.residualBlock = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return x + self.residualBlock(x)
        else:
            raise ValueError("Input should be the tensor type".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual Block for UNIT-GAN".title())
    parser.add_argument(
        "--channels", type=int, default=256, help="Define the channels".capitalize()
    )

    args = parser.parse_args()

    image_channels = args.channels

    residual = nn.Sequential(
        *[ResidualBlock(in_channels=image_channels) for _ in range(3)]
    )

    batch_size = 1
    image_size = 32

    assert residual(torch.randn(1, 256, 32, 32)).size() == (
        batch_size,
        image_channels,
        image_size,
        image_size,
    ), "Residual block is not working correctly".capitalize()

    draw_graph(
        model=residual,
        input_data=torch.randn(batch_size, image_channels, image_size, image_size),
    ).visual_graph.render(filename="./artifacts/files/residualBlocks", format="pdf")

    print("Residual Block is stored in the folder {}".format("./artifacts/files/"))
