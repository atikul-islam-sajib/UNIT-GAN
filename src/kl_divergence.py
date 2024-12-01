import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()
        self.name = "KL Divergence".title()

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return torch.mean(torch.pow(x, 2))
        else:
            raise ValueError("X should be the type of torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KL Divergence for the UNIT-GAN".title()
    )
    parser.add_argument(
        "--kl", action="store_true", help="Define the loss function name".capitalize()
    )

    args = parser.parse_args()

    batch_size = config()["dataloader"]["batch_size"]

    loss = KLDivergence()

    predicted = torch.randn((batch_size, 64))

    assert (
        type(loss(predicted)) == torch.Tensor
    ), "Output should be the torch.Tensor".capitalize()
