import torch.nn as nn


class PatchBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.model(x)


class PatchGanDiscriminator(nn.Module):
    def __init__(self, in_channels=3, channels=64, n_layers=3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.model.extend(nn.Sequential(
            *[PatchBlock(channels * 2 ** m, channels * 2 ** (m + 1),) for m in range(n_layers - 1)]
        ))

        self.model.append(PatchBlock(channels * 2 ** (n_layers - 1), channels * 2 ** n_layers, stride=1))
        self.model.append(nn.Conv2d(channels * 2 ** n_layers, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, input):
        return self.model(input)


class PixelGanDiscriminator(nn.Module):
    def __init__(self, in_channels=3, channels=64):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, input):
        """Standard forward."""
        return self.model(input)