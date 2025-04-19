import torch.nn as nn


class PatchBlock(nn.Module):
    """
    A basic convolutional block used in the PatchGAN discriminator.

    Applies a 2D convolution, followed by instance normalization and LeakyReLU.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param kernel_size: Kernel size (default is 4).
    :param stride: Stride of the convolution (default is 2).
    :param padding: Padding value (default is 1).
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of the PatchBlock.

        :param x: Input tensor.
        :return: Output tensor.
        """
        return self.model(x)


class PatchGanDiscriminator(nn.Module):
    """
    PatchGAN discriminator for GAN architectures.

    This discriminator classifies overlapping image patches as real or fake,
    which is effective for high-frequency detail learning.

    :param in_channels: Number of input image channels (e.g., 3 for RGB).
    :param channels: Base number of channels for the first conv layer.
    :param n_layers: Number of PatchBlock layers (controls receptive field).
    """
    def __init__(self, in_channels=3, channels=64, n_layers=3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.model.extend(nn.Sequential(
            *[PatchBlock(channels * 2 ** m, channels * 2 ** (m + 1)) for m in range(n_layers - 1)]
        ))

        self.model.append(PatchBlock(channels * 2 ** (n_layers - 1), channels * 2 ** n_layers, stride=1))
        self.model.append(nn.Conv2d(channels * 2 ** n_layers, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, input):
        """
        Forward pass through the PatchGAN discriminator.

        :param input: Input tensor of shape (B, C, H, W).
        :return: Patch-level discrimination scores.
        """
        return self.model(input)


class PixelGanDiscriminator(nn.Module):
    """
    PixelGAN discriminator that operates on individual pixels.

    Consists of 1x1 convolutions and is designed for pixel-wise discrimination.

    :param in_channels: Number of input image channels.
    :param channels: Base number of channels.
    :param n_layers: Number of PatchBlock layers (controls feature depth).
    """
    def __init__(self, in_channels=3, channels=64, n_layers=1):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.model.extend(nn.Sequential(
            *[PatchBlock(channels * 2 ** m, channels * 2 ** (m + 1), kernel_size=1, stride=1, padding=0)
              for m in range(n_layers)]
        ))

        self.model.append(nn.Conv2d(channels * 2 ** n_layers, 1, kernel_size=1, stride=1, padding=0))

    def forward(self, input):
        """
        Forward pass through the PixelGAN discriminator.

        :param input: Input tensor of shape (B, C, H, W).
        :return: Pixel-level discrimination scores.
        """
        return self.model(input)