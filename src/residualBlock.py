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
        self.stride_size = 2
        self.padding_size = 1

        self.layers = []

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError("Input should be the tensor type".capitalize())


if __name__ == "__main__":
    residual = ResidualBlock(in_channels=64)
