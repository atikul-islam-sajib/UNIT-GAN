import os
import sys
import torch
import torch.nn as nn

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
    residual = ResidualBlock(in_channels=256)
    print(residual(torch.randn(1, 256, 32, 32)).size())
