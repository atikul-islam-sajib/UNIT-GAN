import os
import sys
import torch
import unittest
import torch.nn as nn

sys.path.append("./src/")

from encoder import Encoder
from residualBlock import ResidualBlock


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.residualBlock = ResidualBlock(in_channels=256)
        self.shared_E = ResidualBlock(in_channels=256)
        self.encoder1 = Encoder(in_channels=3, sharedBlocks=self.shared_E)
        self.encoder2 = Encoder(in_channels=3, sharedBlocks=self.shared_E)

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

    def test_encoder(self):
        batch_size = 1
        image_size = 128

        mu1, z1 = self.encoder1(torch.randn(batch_size, 3, image_size, image_size))
        mu2, z2 = self.encoder2(torch.randn(batch_size, 3, image_size, image_size))

        self.assertEqual(
            mu1.size(), mu2.size(), "Shape mismatch(mu1, mu2)".capitalize()
        )
        self.assertEqual(z1.size(), z2.size(), "Shape mismatch(z1, z2)".capitalize())


if __name__ == "__main__":
    unittest.main()
