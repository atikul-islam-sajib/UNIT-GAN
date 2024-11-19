import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class GANLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(GANLoss, self).__init__()

        self.name = "GANLoss for the UNIT-GAN"
        self.reduction = reduction
        self.loss = nn.MSELoss(reduction=self.reduction)

    def forward(self, predicted: torch.Tensor, actual: torch.Tensor):
        if isinstance(predicted, torch.Tensor) and isinstance(actual, torch.Tensor):
            return self.loss(predicted, actual)
        else:
            raise ValueError("Predicted and actual should be both tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GANLoss for the UNIT-GAN".title())
    parser.add_argument(
        "--reduction",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Define the reduction".capitalize(),
    )
    args = parser.parse_args()

    loss = GANLoss(reduction=args.reduction)

    actual = torch.tensor([1.0, 0.0, 1.0, 1.0])
    predicted = torch.tensor([1.0, 0.0, 1.0, 1.0])

    assert (
        type(loss(predicted, actual)) == torch.Tensor
    ), "Result should be in the tensor format".capitalize()
