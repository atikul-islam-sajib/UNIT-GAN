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
    netD = Discriminator(in_channels=3)
    print(netD(torch.randn(1, 3, 128, 128)).size())
