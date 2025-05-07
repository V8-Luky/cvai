import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualGenerator(nn.Module):
    """
    Residual-based generator model for image-to-image translation tasks.

    This model is composed of:
    - An initial convolutional layer to expand input channels.
    - A set of downsampling layers.
    - A sequence of residual blocks.
    - A set of upsampling layers.
    - A final convolution to restore original input dimensions.

    :param in_channels: Number of input channels (e.g., 3 for RGB).
    :param channels: Base number of channels used in convolutions.
    :param n_blocks: Number of residual blocks used in the network.
    :param sampling_steps: Number of downsampling and upsampling layers.
    """
    def __init__(self, in_channels=3, channels=64, n_blocks=6, sampling_steps=2):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, channels, kernel_size=7, padding=3, stride=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        ]

        layers.extend(
            [ConvBlock(
                channels * 2 ** m,
                channels * 2 ** (m + 1),
                down=True,
                kernel_size=3,
                stride=2,
                padding=1,
            ) for m in range(sampling_steps)]
        )

        layers.extend(
            [ResidualBlock(channels * 2 ** sampling_steps)
              for _ in range(n_blocks)]
        )

        layers.extend(
            [ConvBlock(
                channels * 2 ** m,
                channels * 2 ** (m - 1),
                down=False,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ) for m in range(sampling_steps, 0, -1)]
        )

        layers.append(
            nn.Conv2d(channels, in_channels, kernel_size=7, padding=3, stride=1, padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        """
        Performs the forward pass through the generator.

        :param input: Input tensor of shape (B, C, H, W).
        :return: Output tensor of the same shape as the input.
        """
        return F.tanh(self.model(input))


class ResidualBlock(nn.Module):
    """
    A standard residual block consisting of two convolutional layers.

    :param dim: Number of input and output channels.
    :param padding_mode: Padding mode used in the convolution layers.
    """
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(dim, dim, kernel_size=3, padding=1, stride=1),
            ConvBlock(dim, dim, use_act=False, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        Applies the residual block to the input tensor.

        :param x: Input tensor.
        :return: Output tensor with residual connection applied.
        """
        return x + self.block(x)


class ConvBlock(nn.Module):
    """
    A convolutional block used for downsampling or upsampling operations.

    :param in_dim: Number of input channels.
    :param out_dim: Number of output channels.
    :param down: If True, performs downsampling using Conv2d; otherwise, performs upsampling using ConvTranspose2d.
    :param use_act: If True, activate output with ReLU; otherwise return unactivated.
    :param **kwargs: Arguments for Conv2d and ConvTranspose2d layers.
    """
    def __init__(self, in_dim, out_dim, down=True, use_act=True, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, padding_mode="reflect", **kwargs)
            if down else
            nn.ConvTranspose2d(in_dim, out_dim, **kwargs),
            nn.InstanceNorm2d(out_dim),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        """
        Applies the convolutional block transformation to the input tensor.

        :param x: Input tensor.
        :return: Transformed output tensor.
        """
        return self.block(x)


class UnetGenerator(nn.Module):
    """
    U-Net style generator architecture for image generation or translation tasks.

    This class constructs a symmetrical encoder-decoder network with skip connections,
    commonly used in tasks like image segmentation and style transfer.

    :param in_channels: Number of input channels (e.g., 3 for RGB images).
    :param features: Base number of feature maps in the network.
    :param sampling_steps: Number of downsampling/upsampling steps.
    """
    def __init__(self, in_channels=3, features=64, sampling_steps=5):
        super().__init__()
        block = InnermostUnetBlock(features * 8, features * 8)
        for _ in range(sampling_steps - 5):
            block = UnetBlock(features * 8, features * 8, submodule=block)
        block = UnetBlock(features * 4, features * 8, submodule=block)
        block = UnetBlock(features * 2, features * 4, submodule=block)
        block = UnetBlock(features * 1, features * 2, submodule=block)
        self.model = OutermostUnetBlock(in_channels, features, submodule=block)

    def forward(self, input):
        """
        Defines the forward pass of the generator.

        :param input: Input tensor of shape (B, C, H, W).
        :return: Output tensor of the same shape as input.
        """
        return self.model(input)


class OutermostUnetBlock(nn.Module):
    """
    The outermost block of the U-Net, handling input and output layers.

    :param dim_outer: Number of input and output channels.
    :param dim_inner: Number of internal feature channels.
    :param submodule: Inner U-Net block to nest within this layer.
    """
    def __init__(self, dim_outer, dim_inner, submodule: nn.Module):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(dim_outer, dim_inner, kernel_size=4, stride=2, padding=1),
            submodule,
            nn.ReLU(True),
            nn.ConvTranspose2d(dim_inner * 2, dim_outer, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Forward pass through the outermost U-Net block.

        :param x: Input tensor.
        :return: Output tensor.
        """
        return self.model(x)


class InnermostUnetBlock(nn.Module):
    """
    The innermost (deepest) block of the U-Net, with no nested submodules.

    :param dim_outer: Input and output channel size.
    :param dim_inner: Hidden channel size used for processing.
    """
    def __init__(self, dim_outer, dim_inner):
        super().__init__()

        self.model = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_outer, dim_inner, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim_inner, dim_outer, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(dim_outer)
        )

    def forward(self, x):
        """
        Forward pass through the innermost U-Net block.

        :param x: Input tensor.
        :return: Concatenated input and output tensor.
        """
        return torch.cat([x, self.model(x)], 1)


class UnetBlock(nn.Module):
    """
    A recursive U-Net block containing nested submodules.

    :param dim_outer: Number of input and output channels.
    :param dim_inner: Number of internal feature channels.
    :param submodule: A nested UnetBlock or InnermostUnetBlock.
    """
    def __init__(self, dim_outer, dim_inner, submodule: nn.Module):
        super().__init__()

        self.model = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_outer, dim_inner, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(dim_inner),
            submodule,
            nn.ReLU(True),
            nn.ConvTranspose2d(dim_inner * 2, dim_outer, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(dim_outer)
        )

    def forward(self, x):
        """
        Forward pass through a U-Net block with skip connection.

        :param x: Input tensor.
        :return: Concatenated input and processed tensor.
        """
        return torch.cat([x, self.model(x)], 1)
