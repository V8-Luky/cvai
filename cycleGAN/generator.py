import torch
import torch.nn as nn


class ResidualGenerator(nn.Module):
    def __init__(self, in_channels=3, channels=64, n_blocks=9, sampling_steps=3, padding_mode='reflect'):
        super().__init__()
        # To 64 channels
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=7, padding=3, padding_mode=padding_mode),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Downsampling
        self.model.extend(nn.Sequential(
            *[ConvBlock(
                channels * 2 ** m,
                channels * 2 ** (m + 1),
                down=True,
                padding_mode=padding_mode
            ) for m in range(sampling_steps)]
        ))

        # Residual connections
        self.model.extend(nn.Sequential(
            *[ResidualBlock(channels * 2 ** sampling_steps, padding_mode=padding_mode)
              for _ in range(n_blocks)]
        ))

        # Upsampling
        self.model.extend(nn.Sequential(
            *[ConvBlock(
                channels * 2 ** m,
                channels * 2 ** (m - 1),
                down=False,
                padding_mode=padding_mode
            ) for m in range(sampling_steps, 0, -1)]
        ))

        # To input channels
        self.model.extend(nn.Sequential(
            nn.Conv2d(channels, in_channels, kernel_size=7, padding=3, padding_mode=padding_mode),
            nn.Tanh()
        ))

    def forward(self, input):
        return self.model(input)


class ResidualBlock(nn.Module):
    def __init__(self, dim, padding_mode):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode=padding_mode),
        )

    def forward(self, x):
        return x + self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, padding_mode, down=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)
            if down else
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.InstanceNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UnetGenerator(nn.Module):
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
        return self.model(input)


class OutermostUnetBlock(nn.Module):
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
        return self.model(x)


class InnermostUnetBlock(nn.Module):
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
        return torch.cat([x, self.model(x)], 1)


class UnetBlock(nn.Module):
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
        return torch.cat([x, self.model(x)], 1)