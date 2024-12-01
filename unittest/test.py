import os
import sys
import torch
import unittest
import torch.nn as nn
import torch.optim as optim

sys.path.append("./src/")

from helper import helper
from encoder import Encoder
from gan_loss import GANLoss
from generator import Generator
from pixel_loss import PixelLoss
from kl_divergence import KLDivergence
from discriminator import Discriminator
from residualBlock import ResidualBlock


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.residualBlock = ResidualBlock(in_channels=256)
        self.shared_E = ResidualBlock(in_channels=256)
        self.encoder1 = Encoder(in_channels=3, sharedBlocks=self.shared_E)
        self.encoder2 = Encoder(in_channels=3, sharedBlocks=self.shared_E)

        self.shared_G = ResidualBlock(in_channels=256)
        self.netG1 = Generator(in_channels=256, sharedBlocks=self.shared_G)
        self.netG2 = Generator(in_channels=256, sharedBlocks=self.shared_G)

        self.netD = Discriminator(in_channels=3)

        self.init = helper(
            lr=2e-4,
            beta1=0.5,
            beta2=0.999,
            momentum=0.95,
            adam=True,
            SGD=False,
        )

        self.pixelLoss = PixelLoss(reduction="mean")
        self.criterion = GANLoss(reduction="mean")
        self.kl_loss = KLDivergence()

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

    def test_generator(self):
        batch_size = 1
        image_channels = 256
        image_size = 32

        generatedImage1 = self.netG1(
            torch.randn(batch_size, image_channels, image_size, image_size)
        )
        generatedImage2 = self.netG2(
            torch.randn(batch_size, image_channels, image_size, image_size)
        )

        self.assertEqual(
            generatedImage1.size(),
            generatedImage2.size(),
            "Shape mismatch(generatedImage1, generatedImage2)".capitalize(),
        )

        for filename in ["netG1", "netG2"]:
            self.assertTrue(
                os.path.exists(f"./artifacts/files/{filename}.pdf"),
                f"Graph for {filename} is not generated".capitalize(),
            )

    def test_discriminator(self):
        batch_size = 1
        image_channels = 3
        image_size = 128

        output1 = self.netD(
            torch.randn(batch_size, image_channels, image_size, image_size)
        )

        for filename in ["netD"]:
            self.assertTrue(
                os.path.exists(f"./artifacts/files/{filename}.pdf"),
                f"Graph for {filename} is not generated".capitalize(),
            )

        self.assertEqual(
            output1.size(),
            (
                batch_size,
                image_channels // image_channels,
                image_size // 16,
                image_size // 16,
            ),
            "Discriminator output is not as expected".capitalize(),
        )

    def test_helper_method(self):
        encoder1 = self.init["E1"]
        encoder2 = self.init["E2"]

        netG1 = self.init["netG1"]
        netG2 = self.init["netG2"]

        netD1 = self.init["netD1"]
        netD2 = self.init["netD2"]

        optimizerG = self.init["optimizerG"]
        optimizerD1 = self.init["optimizerD1"]
        optimizerD2 = self.init["optimizerD2"]

        criterion = self.init["criterion"]
        pixelLoss = self.init["pixelLoss"]

        assert (
            encoder1.__class__ == Encoder
        ), "Encoder object should be Encoder class".capitalize()
        assert (
            encoder2.__class__ == Encoder
        ), "Encoder object should be Encoder class".capitalize()

        assert (
            netG1.__class__ == Generator
        ), "Generator object should be Generator class".capitalize()
        assert (
            netG2.__class__ == Generator
        ), "Generator object should be Generator class".capitalize()

        assert (
            netD1.__class__ == Discriminator
        ), "netD1 object should be Discriminator class".capitalize()
        assert (
            netD2.__class__ == Discriminator
        ), "netD2 object should be Discriminator class".capitalize()

        assert (
            optimizerG.__class__ == optim.Adam
        ), "optimizerG object should be Adam class".capitalize()
        assert (
            optimizerD1.__class__ == optim.Adam
        ), "optimizerD1 object should be Adam class".capitalize()
        assert (
            optimizerD2.__class__ == optim.Adam
        ), "optimizerD2 object should be Adam class".capitalize()

        assert (
            criterion.__class__ == GANLoss
        ), "Criterion object should be GANLoss class".capitalize()
        assert (
            pixelLoss.__class__ == PixelLoss
        ), "pixelLoss object should be PixelLoss class".capitalize()

    def test_pixelLoss(self):
        actual = torch.tensor([1.0, 0.0, 1.0, 1.0])
        predicted = torch.tensor([1.0, 0.0, 1.0, 1.0])

        assert (
            type(self.pixelLoss(predicted, actual)) == torch.Tensor
        ), "Result should be in the tensor format".capitalize()

    def test_GANLoss(self):
        actual = torch.tensor([1.0, 0.0, 1.0, 1.0])
        predicted = torch.tensor([1.0, 0.0, 1.0, 1.0])

        assert (
            type(self.criterion(predicted, actual)) == torch.Tensor
        ), "Result should be in the tensor format".capitalize()

    def test_helper(self):
        init = helper(
            lr=2e-4,
            beta1=0.5,
            beta2=0.999,
            momentum=0.95,
            adam=True,
            SGD=False,
        )

        train_dataloader = init["train_dataloader"]
        valid_dataloader = init["valid_dataloader"]

        encoder1 = init["E1"]
        encoder2 = init["E2"]

        netG1 = init["netG1"]
        netG2 = init["netG2"]

        netD1 = init["netD1"]
        netD2 = init["netD2"]

        optimizerG = init["optimizerG"]
        optimizerD1 = init["optimizerD1"]
        optimizerD2 = init["optimizerD2"]

        criterion = init["criterion"]
        pixelLoss = init["pixelLoss"]
        kl_loss = init["kl_loss"]

        assert (
            train_dataloader.__class__ == torch.utils.data.DataLoader
        ), "Train dataloader shoould be torch.utils.data.DataLoader".capitalize()
        assert (
            valid_dataloader.__class__ == torch.utils.data.DataLoader
        ), "Valid dataloader shoould be torch.utils.data.DataLoader".capitalize()

        assert (
            encoder1.__class__ == Encoder
        ), "Encoder object should be Encoder class".capitalize()
        assert (
            encoder2.__class__ == Encoder
        ), "Encoder object should be Encoder class".capitalize()

        assert (
            netG1.__class__ == Generator
        ), "Generator object should be Generator class".capitalize()
        assert (
            netG2.__class__ == Generator
        ), "Generator object should be Generator class".capitalize()

        assert (
            netD1.__class__ == Discriminator
        ), "netD1 object should be Discriminator class".capitalize()
        assert (
            netD2.__class__ == Discriminator
        ), "netD2 object should be Discriminator class".capitalize()

        assert (
            optimizerG.__class__ == optim.Adam
        ), "optimizerG object should be Adam class".capitalize()
        assert (
            optimizerD1.__class__ == optim.Adam
        ), "optimizerD1 object should be Adam class".capitalize()
        assert (
            optimizerD2.__class__ == optim.Adam
        ), "optimizerD2 object should be Adam class".capitalize()

        assert (
            criterion.__class__ == GANLoss
        ), "Criterion object should be GANLoss class".capitalize()
        assert (
            pixelLoss.__class__ == PixelLoss
        ), "pixelLoss object should be PixelLoss class".capitalize()
        assert (
            kl_loss.__class__ == KLDivergence
        ), "KL Divergence object should be PixelLoss class".capitalize()


if __name__ == "__main__":
    unittest.main()
