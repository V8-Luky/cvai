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
    :param lambda_a: Weight for cycle loss from A to B to A.
    :param lambda_b: Weight for cycle loss from B to A to B.
    :param lambda_identity: Weight for identity loss.
    """
    gen_type: Type[ResidualGenerator] | Type[UnetGenerator] = ResidualGenerator
    gen_channels: int = 64
    gen_kwargs: dict = field(default_factory=dict)

    disc_type: Type[PatchGanDiscriminator] | Type[PixelGanDiscriminator] = PatchGanDiscriminator
    disc_channels: int = 64
    disc_kwargs: dict = field(default_factory=dict)

    lambda_a: float = 10.0
    lambda_b: float = 10.0
    lambda_identity: float = 0.5


class CycleGAN(nn.Module):
    """
    CycleGAN implementation supporting configurable generator/discriminator types.

    CycleGAN learns two mappings: A->B and B->A using adversarial loss and cycle-consistency loss.

    :param config: Configuration object with model settings.
    """
    def __init__(self, config: CycleGANConfig):
        super().__init__()
        self.config = config
        self.generator_a_to_b = config.gen_type(in_channels=3, channels=config.gen_channels, **config.gen_kwargs)
        self.generator_b_to_a = config.gen_type(in_channels=3, channels=config.gen_channels, **config.gen_kwargs)
        self.discriminator_a = config.disc_type(in_channels=3, channels=config.disc_channels, **config.disc_kwargs)
        self.discriminator_b = config.disc_type(in_channels=3, channels=config.disc_channels, **config.disc_kwargs)

        self.loss_gan = nn.MSELoss()
        self.loss_cycle = nn.L1Loss()
        self.loss_identity = nn.L1Loss()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the forward pass for CycleGAN using inputs from domain A and B.

        :param a: Real images from domain A.
        :param b: Real images from domain B.
        :return: Tuple of tensors of the fake images (fake_a, fake_b).
        """
        return self.generator_b_to_a(b), self.generator_a_to_b(a)

    def generator_loss(
            self,
            real_a: torch.Tensor,
            real_b: torch.Tensor,
            fake_a: torch.Tensor,
            fake_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the total generator loss including adversarial, cycle consistency,
        and optional identity loss.

        :return: Total generator loss.
        """
        lambda_idt = self.config.lambda_identity
        lambda_a = self.config.lambda_a
        lambda_b = self.config.lambda_b

        disc_a = self.discriminator_a(fake_a)
        disc_b = self.discriminator_b(fake_b)
        loss_gen_a = self.loss_gan(disc_a, torch.ones_like(disc_a))
        loss_gen_b = self.loss_gan(disc_b, torch.ones_like(disc_b))

        rec_a = self.generator_b_to_a(fake_b)
        rec_b = self.generator_a_to_b(fake_a)
        loss_cycle_a = self.loss_cycle(rec_a, real_a) * lambda_a
        loss_cycle_b = self.loss_cycle(rec_b, real_b) * lambda_b

        if lambda_idt > 0.0:
            loss_idt_a = self.loss_identity(self.generator_b_to_a(real_a), real_a) * lambda_idt * lambda_a
            loss_idt_b = self.loss_identity(self.generator_a_to_b(real_b), real_b) * lambda_idt * lambda_b
        else:
            loss_idt_a = 0
            loss_idt_b = 0

        return loss_gen_a + loss_gen_b + loss_cycle_a + loss_cycle_b + loss_idt_a + loss_idt_b

    def discriminator_loss(
            self,
            real_a: torch.Tensor,
            real_b: torch.Tensor,
            fake_a: torch.Tensor,
            fake_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the total discriminator loss using real and fake samples
        for both domain A and B.

        :return: Total discriminator loss.
        """
        disc_real_a = self.discriminator_a(real_a)
        disc_fake_a = self.discriminator_a(fake_a.detach())
        loss_disc_real_a = self.loss_gan(disc_real_a, torch.ones_like(disc_real_a))
        loss_disc_fake_a = self.loss_gan(disc_fake_a, torch.zeros_like(disc_fake_a))
        loss_disc_a = loss_disc_real_a + loss_disc_fake_a

        disc_real_b = self.discriminator_b(real_b)
        disc_fake_b = self.discriminator_b(fake_b.detach())
        loss_disc_real_b = self.loss_gan(disc_real_b, torch.ones_like(disc_real_b))
        loss_disc_fake_b = self.loss_gan(disc_fake_b, torch.zeros_like(disc_fake_b))
        loss_disc_b = loss_disc_real_b + loss_disc_fake_b

        return (loss_disc_a + loss_disc_b) * 0.5

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
