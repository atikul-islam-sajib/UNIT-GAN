import os
import sys
import torch
import unittest
import torch.nn as nn

sys.path.append("./src/")

from residualBlock import ResidualBlock


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.residualBlock = ResidualBlock(in_channels=256)

    def test_residualBlocks(self):
        self.assertEqual(
            self.residualBlock(x=torch.randn(1, 256, 32, 32)).size(),
            (1, 256, 32, 32),
            "Single Residual block is not working properly".capitalize(),
        )

        residualBlock = nn.Sequential(
            *[ResidualBlock(in_channels=256) for _ in range(3)]
        )

        self.assertEqual(
            residualBlock(torch.randn(1, 256, 32, 32)).size(),
            (1, 256, 32, 32),
            "Multiple Residual blocks are not working properly".capitalize(),
        )


if __name__ == "__main__":
    unittest.main()
