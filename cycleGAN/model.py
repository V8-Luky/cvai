from dataclasses import dataclass, field
import torch
import torch.nn as nn
from typing import Type
from collections.abc import Iterator

from .generator import UnetGenerator, ResidualGenerator
from .discriminator import PixelGanDiscriminator, PatchGanDiscriminator


@dataclass
class CycleGANConfig:
    """
    Configuration dataclass for initializing CycleGAN components.

    :param gen_type: Generator architecture to use (ResidualGenerator or UnetGenerator).
    :param gen_channels: Base number of channels for generator.
    :param gen_kwargs: Additional keyword arguments for generator initialization.
    :param disc_type: Discriminator architecture to use (PatchGan or PixelGan).
    :param disc_channels: Base number of channels for discriminator.
    :param disc_kwargs: Additional keyword arguments for discriminator initialization.
    """

    gen_type: Type[ResidualGenerator] | Type[UnetGenerator] = ResidualGenerator
    gen_channels: int = 64
    gen_kwargs: dict = field(default_factory=dict)

    disc_type: Type[PatchGanDiscriminator] | Type[PixelGanDiscriminator] = PatchGanDiscriminator
    disc_channels: int = 64
    disc_kwargs: dict = field(default_factory=dict)


class CycleGAN(nn.Module):
    """
    CycleGAN implementation supporting configurable generator/discriminator types.

    CycleGAN learns two mappings: A->B and B->A using adversarial loss and cycle-consistency loss.

    :param config: Configuration object with model settings.
    """

    def __init__(self, config: CycleGANConfig):
        super().__init__()
        self.config = config
        self.generator_a_to_b = config.gen_type(
            in_channels=3, channels=config.gen_channels, **config.gen_kwargs
        )
        self.generator_b_to_a = config.gen_type(
            in_channels=3, channels=config.gen_channels, **config.gen_kwargs
        )
        self.discriminator_a = config.disc_type(
            in_channels=3, channels=config.disc_channels, **config.disc_kwargs
        )
        self.discriminator_b = config.disc_type(
            in_channels=3, channels=config.disc_channels, **config.disc_kwargs
        )

        self.loss_gan = nn.MSELoss()
        self.loss_cycle = nn.L1Loss()
        self.loss_identity = nn.L1Loss()

    def forward(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the forward pass for CycleGAN using inputs from domain A and B.

        :param a: Real images from domain A.
        :param b: Real images from domain B.
        :return: Tuple of tensors of the fake images (fake_a, fake_b).
        """
        return self.generator_b_to_a(b), self.generator_a_to_b(a)

    def get_discriminator_params(self) -> Iterator[nn.Parameter]:
        """
        Returns an iterator over all discriminator parameters (both A and B).

        :return: Iterator over discriminator parameters.
        """
        yield from self.discriminator_a.parameters()
        yield from self.discriminator_b.parameters()

    def get_generator_params(self) -> Iterator[nn.Parameter]:
        """
        Returns an iterator over all generator parameters (both A->B and B->A).

        :return: Iterator over generator parameters.
        """
        yield from self.generator_a_to_b.parameters()
        yield from self.generator_b_to_a.parameters()

    def get_parameters(self) -> Iterator[nn.Parameter]:
        """
        Returns an iterator over all parameters of the CycleGAN model.

        :return: Iterator over all model parameters.
        """
        yield from self.get_generator_params()
        yield from self.get_discriminator_params()
